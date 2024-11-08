import os
import sys
import time
import logging
from typing import Optional, Any, List, Dict, Set, Union, Tuple

import click
import pandas as pd

from snipe.api.enums import SigType
from snipe.api.snipe_sig import SnipeSig
from snipe.api.multisig_reference_QC import MultiSigReferenceQC
from snipe import __version__

import json
import lzstring
import hashlib
import multiprocessing
from joblib import Parallel, delayed

class MetadataSerializer:
    def __init__(self, logger: Optional[logging.Logger] = None, hash_algo: str = 'sha256'):
        self.hash_algo = hash_algo
        self.logger = logger or self._configure_default_logger()

    @staticmethod
    def _configure_default_logger() -> logging.Logger:
        logger = logging.getLogger('MetadataSerializer')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        try:
            json_str = json.dumps(metadata, sort_keys=True)
            compressor = lzstring.LZString()
            compressed = compressor.compressToBase64(json_str)
            return compressed
        except (TypeError, ValueError) as e:
            self.logger.error(f"Serialization failed: {e}")
            raise ValueError(f"Serialization failed: {e}") from e

    def deserialize_metadata(self, compressed_metadata: str) -> Dict[str, Any]:
        try:
            compressor = lzstring.LZString()
            json_str = compressor.decompressFromBase64(compressed_metadata)
            if json_str is None:
                raise ValueError("Decompression returned None.")
            metadata = json.loads(json_str)
            return metadata
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"Deserialization failed: {e}")
            raise ValueError(f"Deserialization failed: {e}") from e

    def compute_checksum(self, data: Dict[str, Any]) -> str:
        try:
            serialized = json.dumps(data, sort_keys=True)
            hash_obj = hashlib.new(self.hash_algo)
            hash_obj.update(serialized.encode('utf-8'))
            checksum = hash_obj.hexdigest()
            return checksum
        except (TypeError, ValueError, hashlib.Hash) as e:
            self.logger.error(f"Checksum computation failed: {e}")
            raise ValueError(f"Checksum computation failed: {e}") from e

    def export_and_verify_metadata(self, metadata: Dict[str, Any]) -> Tuple[str, str]:
        self.logger.debug("Exporting metadata...")
        metadata_str = self.serialize_metadata(metadata)
        checksum = self.compute_checksum(metadata)
        deserialized_metadata = self.deserialize_metadata(metadata_str)
        if deserialized_metadata != metadata:
            self.logger.error("Failed to serialize and deserialize metadata correctly.")
            sys.exit(1)
        self.logger.debug("Metadata serialized and deserialized successfully.")
        return metadata_str, checksum


def validate_sig_input(ctx, param, value: tuple) -> str:
    supported_extensions = ['.zip', '.sig']
    for path in value:
        if not os.path.exists(path):
            raise click.BadParameter(f"File not found: {path}")
        if not any(path.lower().endswith(ext) for ext in supported_extensions):
            raise click.BadParameter(f"Unsupported file format: {path}, supported formats: {', '.join(supported_extensions)}")
    return value


def validate_tsv_file(ctx, param, value: str) -> str:
    if not value.lower().endswith('.tsv'):
        raise click.BadParameter('Output file must have a .tsv extension.')
    return value


def split_chunks(lst: List[str], n: int) -> List[List[str]]:
    """Splits the list `lst` into `n` nearly equal chunks. If n is zero or greater than the list length, returns the entire list as one chunk."""
    if n <= 0:
        n = 1
    if n > len(lst):
        n = len(lst)
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_sample_task(args) -> Tuple[str, Dict[str, Any], Optional[str]]:
    (sample_path, sample_name, ref_sig_data, amplicon_sig_data, ychr_sig_data,
     varsigs_data, export_var, roi, predict_extra_folds, debug) = args

    # Reconstruct signatures from shared data
    try:
        reference_sig = ref_sig_data
        amplicon_sig = amplicon_sig_data
        ychr_sig = ychr_sig_data
        vars_snipesigs = varsigs_data

        # Configure logging for the worker
        logger = logging.getLogger(f'snipe_qc_worker_{sample_name}')
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)

        # Initialize QC instance
        qc_inst = MultiSigReferenceQC(
            reference_sig=reference_sig,
            amplicon_sig=amplicon_sig,
            ychr=ychr_sig,
            varsigs=vars_snipesigs,
            export_varsigs=export_var,
            enable_logging=debug
        )
        
        sample_sig = SnipeSig(sourmash_sig=sample_path, sig_type=SigType.SAMPLE, enable_logging=debug)
        if sample_sig.name == "":
            sample_sig.name = sample_name
            logger.warning(f"Sample name is empty. Setting to: `{sample_sig.name}`")

        sample_stats = qc_inst.process_sample(
            sample_sig=sample_sig,
            predict_extra_folds=predict_extra_folds if roi else None,
            advanced=True
        )
        sample_stats["file_path"] = sample_path
        return sample_sig.name, sample_stats, None
    except Exception as e:
        error_msg = f"Failed to process sample {sample_name}: {e}"
        return sample_name, {}, error_msg

@click.command()
@click.option('--ref', type=click.Path(exists=True), required=True, help='Reference genome signature file (required).')
@click.option('--sample', type=click.Path(exists=True), callback=validate_sig_input, multiple=True, default=None, help='Sample signature file. Can be provided multiple times.')
@click.option('--samples-from-file', type=click.Path(exists=True), help='File containing sample paths (one per line).')
@click.option('--amplicon', type=click.Path(exists=True), help='Amplicon signature file (optional).')
@click.option('--roi', is_flag=True, default=False, help='Calculate ROI for 1,2,5,9 folds.')
@click.option('--export-var', is_flag=True, default=False, help='Export signatures for variances')
@click.option('--ychr', type=click.Path(exists=True), help='Y chromosome signature file (overrides the reference ychr).')
@click.option('--debug', is_flag=True, default=False, help='Enable debugging and detailed logging.')
@click.option('-o', '--output', required=True, callback=validate_tsv_file, help='Output TSV file for QC results.')
@click.option('--var', 'vars', multiple=True, type=click.Path(exists=True), help='Extra signature file path to study variance of non-reference k-mers.')
@click.option('-c', '--cores', type=int, default=1, show_default=True, help='Number of CPU cores to use for parallel processing.')
@click.option('--metadata', type=str, default=None, help='Additional metadata in the format colname=value,colname=value,...')
@click.option('--metadata-from-file', type=click.Path(exists=True), help='File containing metadata information in TSV or CSV format.')
def qc(ref: str, sample: List[str], samples_from_file: Optional[str],
       amplicon: Optional[str], roi: bool, export_var: bool,
       ychr: Optional[str], debug: bool, output: str, vars: List[str], cores: int,
       metadata: Optional[str], metadata_from_file: Optional[str]):
    
    start_time = time.time()

    # Configure logging
    logger = logging.getLogger('snipe_qc')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    # Ensure that loggers propagate to the root logger
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger().propagate = True

    logger.info("Starting QC process.")

    # Collect sample paths from --sample and --samples-from-file
    samples_set: Set[str] = set()
    if sample:
        for _sample in sample:
            logger.debug(f"Adding sample from command-line: {_sample}")
            samples_set.add(_sample)
                

    if samples_from_file:
        logger.debug(f"Reading samples from file: {samples_from_file}")
        try:
            with open(samples_from_file, 'r', encoding='utf-8') as f:
                file_samples = {line.strip() for line in f if line.strip()}
            samples_set.update(file_samples)
            logger.debug(f"Collected {len(file_samples)} samples from file.")
        except Exception as e:
            logger.error(f"Failed to read samples from file {samples_from_file}: {e}")
            sys.exit(1)

    # Deduplicate and validate sample paths
    valid_samples = []
    for sample_path in samples_set:
        if os.path.exists(sample_path):
            valid_samples.append(os.path.abspath(sample_path))
        else:
            logger.warning(f"Sample file does not exist and will be skipped: {sample_path}")

    if not valid_samples:
        logger.error("No valid samples provided for QC.")
        sys.exit(1)

    logger.info(f"Total valid samples to process: {len(valid_samples)}")

    # Load reference signature
    logger.info(f"Loading reference signature from: {ref}")
    try:
        reference_sig = SnipeSig(sourmash_sig=ref, sig_type=SigType.GENOME, enable_logging=debug)
        logger.debug(f"Loaded reference signature: {reference_sig.name}")
    except Exception as e:
        logger.error(f"Failed to load reference signature from {ref}: {e}")
        sys.exit(1)

    # Load amplicon signature if provided
    amplicon_sig = None
    if amplicon:
        logger.info(f"Loading amplicon signature from: {amplicon}")
        try:
            amplicon_sig = SnipeSig(sourmash_sig=amplicon, sig_type=SigType.AMPLICON, enable_logging=debug)
            logger.debug(f"Loaded amplicon signature: {amplicon_sig.name}")
        except Exception as e:
            logger.error(f"Failed to load amplicon signature from {amplicon}: {e}")
            sys.exit(1)
    
    # Load Y chromosome signature if provided
    ychr_sig = None
    if ychr:
        logger.info(f"Loading Y chromosome signature from: {ychr}")
        try:
            ychr_sig = SnipeSig(sourmash_sig=ychr, sig_type=SigType.GENOME, enable_logging=debug)
            logger.debug(f"Loaded Y chromosome signature: {ychr_sig.name}")
        except Exception as e:
            logger.error(f"Failed to load Y chromosome signature from {ychr}: {e}")
            sys.exit(1)

    # Prepare variance signatures if provided
    vars_paths = []
    vars_snipesigs = []
    if vars:
        logger.debug(f"Loading {len(vars)} variance signature(s).")
        for path in vars:
            if not os.path.exists(path):
                logger.error(f"Variance signature file does not exist: {path}")
                sys.exit(1)
            vars_paths.append(os.path.abspath(path))
            try:
                var_sig = SnipeSig(sourmash_sig=path, sig_type=SigType.AMPLICON, enable_logging=debug)
                vars_snipesigs.append(var_sig)
                logger.debug(f"Loaded variance signature: {var_sig.name}")
            except Exception as e:
                logger.error(f"Failed to load variance signature from {path}: {e}")
                    
        logger.debug(f"Variance signature paths: {vars_paths}")
    

    export_metadata = {
            "scale": reference_sig.scale,
            "ksize": reference_sig.ksize,
            "reference": {
                "name": reference_sig.name,
                "md5sum": reference_sig.md5sum,
                "filename": os.path.basename(ref)
            },
            "amplicon": {
                "name": amplicon_sig.name,
                "md5sum": amplicon_sig.md5sum,
                "filename": os.path.basename(amplicon)
            } if amplicon_sig else {
                "name": "",
                "md5sum": "",
                "filename": ""
            },
            "ychr": {
                "name": ychr_sig.name,
                "md5sum": ychr_sig.md5sum,
                "filename": ychr
            } if ychr_sig else {
                "name": "",
                "md5sum": "",
                "filename": ""
            },
            "variance": [
                {
                    "name": var.name,
                    "md5sum": var.md5sum,
                    "filename": os.path.basename(path)
                } for var, path in zip(vars_snipesigs, vars_paths)
            ] if vars_snipesigs else []
        }
                
    # Instantiate MetadataSerializer
    METADATA = MetadataSerializer(
        logger=logger,
        hash_algo='sha256',
    )
        
    metadata_str, metadata_md5sum = METADATA.export_and_verify_metadata(
        metadata=export_metadata
    )
    
    predict_extra_folds = [1, 2, 5, 9]
    
    sample_to_stats = {}
    failed_samples = []

    tasks = []
    for sample_path in valid_samples:
        sample_name = os.path.basename(sample_path).split('.')[0]
        task_args = (
            sample_path,
            sample_name,
            reference_sig,
            amplicon_sig,
            ychr_sig,
            vars_snipesigs,
            export_var,
            roi,
            predict_extra_folds,
            debug
        )
        tasks.append(task_args)

    # Define a wrapper function for joblib
    def joblib_process_sample(args):
        return process_sample_task(args)

    # Use joblib for parallel processing
    results = Parallel(n_jobs=cores)(
        delayed(joblib_process_sample)(args) for args in tasks
    )

    # Collect results
    for name, stats, error_msg in results:
        if error_msg:
            failed_samples.append(name)
            logger.error(error_msg)
        else:
            sample_to_stats[name] = stats
            logger.debug(f"Successfully processed sample: {name}")

    # Separate successful and failed results
    succeeded = list(sample_to_stats.keys())
    failed = len(failed_samples)

    # Handle complete failure
    if len(succeeded) == 0:
        logger.error("All samples failed during QC processing. Output TSV will not be generated.")
        sys.exit(1)

    # write total success and failure
    logger.info("Successfully processed samples: %d", len(succeeded))

    # Prepare the command-line invocation for comments
    command_invocation = ' '.join(sys.argv)

    # Create pandas DataFrame for succeeded samples
    df = pd.DataFrame(sample_to_stats.values())

    # Reorder columns to have 'sample' and 'file_path' first, if they exist
    cols = list(df.columns)
    reordered_cols = []
    for col in ['sample', 'file_path']:
        if col in cols:
            reordered_cols.append(col)
            cols.remove(col)
    reordered_cols += cols
    df = df[reordered_cols]

    # Process --metadata option
    metadata_dict = {}
    if metadata:
        try:
            for item in metadata.split(','):
                key, value = item.split('=', 1)
                metadata_dict[key.strip()] = value.strip()
            logger.debug(f"Parsed metadata from command line: {metadata_dict}")
        except Exception as e:
            logger.error(f"Failed to parse --metadata: {e}")
            sys.exit(1)

    # Process --metadata-from-file option
    if metadata_from_file:
        if not os.path.exists(metadata_from_file):
            logger.error(f"Metadata file does not exist: {metadata_from_file}")
            sys.exit(1)
        try:
            # Read metadata file
            metadata_df = pd.read_csv(metadata_from_file, sep=None, engine='python', header=None)
            # Validate columns
            if metadata_df.shape[1] != 3:
                logger.error("Metadata file must have three columns: sample_path, metadata_col, value")
                sys.exit(1)
            metadata_samples = metadata_df.iloc[:, 0].tolist()
            metadata_cols = metadata_df.iloc[:, 1].tolist()
            metadata_values = metadata_df.iloc[:, 2].tolist()
            # Build a nested dictionary: {sample_name: {col1: val1, col2: val2, ...}}
            metadata_sample_dict = {}
            for sample_path, col_name, value in zip(metadata_samples, metadata_cols, metadata_values):
                sample_basename = os.path.basename(sample_path)
                if sample_basename not in metadata_sample_dict:
                    metadata_sample_dict[sample_basename] = {}
                metadata_sample_dict[sample_basename][col_name] = value
            logger.debug(f"Parsed metadata from file: {metadata_sample_dict}")
        except Exception as e:
            logger.error(f"Failed to read or parse metadata file {metadata_from_file}: {e}")
            sys.exit(1)
    else:
        metadata_sample_dict = {}

    # Apply metadata to DataFrame
    if metadata_dict:
        # Apply global metadata to all samples
        for key, value in metadata_dict.items():
            df[key] = value

    # Apply per-sample metadata
    if metadata_sample_dict:
        # Apply per-sample metadata
        df["tmp_basename"] = df["filename"].apply(os.path.basename)
        # Map metadata to samples
        for sample_basename, meta_dict in metadata_sample_dict.items():
            for key, value in meta_dict.items():
                df.loc[df["tmp_basename"] == sample_basename, key] = value
        df.drop(columns=["tmp_basename"], inplace=True)


    # sanitize file_path and filename
    df["filename"] = df["file_path"].apply(os.path.basename)
    # drop file_path
    df.drop(columns=["file_path"], inplace=True)

    """
    coverage: 5 decimal points
    xploidy, ycoverage, CV, mean,median,chr-: 2 decimal points
    mapping index, predicted contamination, error index, 3 decimal points
    """
    
    floating_5 = ["coverage"]
    floating_2 = ["Ploidy", "chrY Coverage", "CV", "mean", "median", "chr-", 'Relative total abundance', 'fraction of total abundance']
    floating_3 = ["Mapping index", "contamination", "error", "k-mer-to-bases ratio"]

    # for any float columns, round to 4 decimal places
    for col in df.columns:
        if (df[col].dtype == float) and (df[col].eq(0).all()):
            df[col] = df[col].astype(int)
        if any([x in col for x in floating_5]):
            df[col] = df[col].round(5)
        elif any([x in col for x in floating_2]):
            df[col] = df[col].round(2)
        elif any([x in col for x in floating_3]):
            df[col] = df[col].round(3)
    
    # make sure all integer columns are converted to int
    df = df.apply(lambda col: col.apply(lambda x: int(x) if isinstance(x, float) and x.is_integer() else x))
    df_zero_uniqe_hashes = df[df["Total unique k-mers"] == 0]
    df = df[(df["Total unique k-mers"] > 0)]
    if len(df_zero_uniqe_hashes):
        logger.warning(f"Empty sigs not processed: {len(df_zero_uniqe_hashes)}: {', '.join(df_zero_uniqe_hashes['filename'])}")
    
    # fill na by zero
    df.fillna(0, inplace=True)
    
    try:
        with open(output, 'w', encoding='utf-8') as f:
            header_dict = {"snipe-version": __version__, "metadata": metadata_str}
            f.write(f"#{json.dumps(header_dict)}\n")
            df.to_csv(f, sep='\t', index=False)
        logger.info(f"QC results successfully exported to {output}")
    except Exception as e:
        logger.error(f"Failed to export QC results to {output}: {e}")
        sys.exit(1)

    # Report failed samples if any
    if failed:
        logger.warning(f"The following {len(failed_samples)} sample(s) failed during QC processing:")
        for sample in failed_samples:
            logger.warning(f"- {sample}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"QC process completed in {elapsed_time:.2f} seconds.")
