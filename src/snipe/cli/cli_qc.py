import os
import sys
import time
import logging
from typing import Optional, Any, List, Dict, Set, Union, Tuple

import click
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from snipe.api.enums import SigType
from snipe.api.snipe_sig import SnipeSig
from snipe.api.multisig_reference_QC import MultiSigReferenceQC
# from snipe.api.metadata_manager import MetadataSerializer
import json
import lzstring
import hashlib

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
        self.logger.info("Exporting metadata...")
        metadata_str = self.serialize_metadata(metadata)
        checksum = self.compute_checksum(metadata)
        deserialized_metadata = self.deserialize_metadata(metadata_str)
        if deserialized_metadata != metadata:
            self.logger.error("Failed to serialize and deserialize metadata correctly.")
            sys.exit(1)
        self.logger.info("Metadata serialized and deserialized successfully.")
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
    """Splits the list `lst` into `n` nearly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_subset(
    subset: List[str],
    ref: str,
    amplicon: Optional[str],
    ychr: Optional[str],
    vars: List[str],
    export_var: bool,
    roi: bool,
    debug: bool
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Worker function to process a subset of samples.
    
    Args:
        subset (List[str]): List of sample file paths to process.
        ref (str): Reference signature file path.
        amplicon (Optional[str]): Amplicon signature file path.
        ychr (Optional[str]): Y chromosome signature file path.
        vars (List[str]): List of variance signature file paths.
        export_var (bool): Flag to export variance signatures.
        roi (bool): Flag to calculate ROI.
        debug (bool): Flag to enable debug logging.
    
    Returns:
        Tuple[Dict[str, Any], List[str]]: A tuple containing:
            - A dictionary mapping sample names to their QC statistics.
            - A list of sample names that failed to process.
    """
    # Configure logging for the worker
    subset_logger = logging.getLogger('snipe_qc_worker')
    subset_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not subset_logger.hasHandlers():
        subset_logger.addHandler(handler)
    
    subset_logger.debug(f"Worker started with {len(subset)} samples.")
    
    # Load reference signature
    try:
        reference_sig = SnipeSig(sourmash_sig=ref, sig_type=SigType.GENOME, enable_logging=debug)
        subset_logger.debug(f"Loaded reference signature: {reference_sig.name}")
    except Exception as e:
        subset_logger.error(f"Failed to load reference signature from {ref}: {e}")
        return {}, subset  # All samples in this subset fail
    
    # Load amplicon signature if provided
    amplicon_sig = None
    if amplicon:
        try:
            amplicon_sig = SnipeSig(sourmash_sig=amplicon, sig_type=SigType.AMPLICON, enable_logging=debug)
            subset_logger.debug(f"Loaded amplicon signature: {amplicon_sig.name}")
        except Exception as e:
            subset_logger.error(f"Failed to load amplicon signature from {amplicon}: {e}")
            return {}, subset  # All samples in this subset fail
    
    # Load Y chromosome signature if provided
    ychr_sig = None
    if ychr:
        try:
            ychr_sig = SnipeSig(sourmash_sig=ychr, sig_type=SigType.GENOME, enable_logging=debug)
            subset_logger.debug(f"Loaded Y chromosome signature: {ychr_sig.name}")
        except Exception as e:
            subset_logger.error(f"Failed to load Y chromosome signature from {ychr}: {e}")
            return {}, subset  # All samples in this subset fail
    
    # Load variance signatures if provided
    vars_snipesigs = []
    if vars:
        subset_logger.debug(f"Loading {len(vars)} variance signature(s).")
        for path in vars:
            if not os.path.exists(path):
                subset_logger.error(f"Variance signature file does not exist: {path}")
                return {}, subset  # All samples in this subset fail
            try:
                var_sig = SnipeSig(sourmash_sig=path, sig_type=SigType.AMPLICON, enable_logging=debug)
                vars_snipesigs.append(var_sig)
                subset_logger.debug(f"Loaded variance signature: {var_sig.name}")
            except Exception as e:
                subset_logger.error(f"Failed to load variance signature from {path}: {e}")
                return {}, subset  # All samples in this subset fail
    
    # Initialize QC instance
    try:
        qc_inst = MultiSigReferenceQC(
            reference_sig=reference_sig,
            amplicon_sig=amplicon_sig,
            ychr=ychr_sig if ychr_sig else None,
            varsigs=vars_snipesigs if vars_snipesigs else None,
            export_varsigs=export_var,
            enable_logging=debug
        )
    except Exception as e:
        subset_logger.error(f"Failed to initialize MultiSigReferenceQC: {e}")
        return {}, subset  # All samples in this subset fail
    
    predict_extra_folds = [1, 2, 5, 9]
    
    subset_stats = {}
    subset_failed = []
    for sample_path in subset:
        sample_sig = SnipeSig(sourmash_sig=sample_path, sig_type=SigType.SAMPLE, enable_logging=debug)
        subset_logger.debug(f"DELME Processing sample: {sample_sig.name}")
        if sample_sig.name == "":
            _newname = os.path.basename(sample_path).split('.')[0]
            sample_sig.name = _newname
            subset_logger.warning(f"Sample name is empty. Setting to: `{sample_sig.name}`")
            
        try:
            sample_stats = qc_inst.process_sample(
                sample_sig=sample_sig,
                predict_extra_folds=predict_extra_folds if roi else None,
                advanced=True
            )
            #! override the internal sig file path with the actual used file path.
            sample_stats["file_path"] =  sample_path
            subset_stats[sample_sig.name] = sample_stats
            subset_logger.debug(f"Successfully processed sample: {sample_sig.name}")
        except Exception as e:
            subset_failed.append(sample_sig.name)
            subset_logger.error(f"Failed to process sample {sample_sig.name}: {e}")
            continue
    
    subset_logger.debug(f"Worker completed. Success: {len(subset_stats)}, Failed: {len(subset_failed)}")
    return subset_stats, subset_failed


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

    """
        Perform quality control (QC) on multiple samples against a reference genome.

        This command calculates various QC metrics for each provided sample, optionally including advanced metrics and ROI (Return on investement) predictions. Results are aggregated and exported to a TSV file.

        ## Usage

        ```bash
        snipe qc [OPTIONS]
        ```

        ## Options

        - `--ref PATH` **[required]**  
        Reference genome signature file.

        - `--sample PATH`  
        Sample signature file. Can be provided multiple times.

        - `--samples-from-file PATH`  
        File containing sample paths (one per line).

        - `--amplicon PATH`  
        Amplicon signature file (optional).

        - `--roi`  
        Calculate ROI for 1x, 2x, 5x, and 9x coverage folds.

        - `--ychr PATH`  
        Y chromosome signature file (overrides the reference ychr).

        - `--debug`  
        Enable debugging and detailed logging.

        - `-o`, `--output PATH` **[required]**  
        Output TSV file for QC results.

        - `--var PATH`  
        Variance signature file path. Can be used multiple times.
        
        - `--export-var`
        Export signatures for sample hashes found in the variance signature.
        
        - `-c`, `--cores INT`
        Number of CPU cores to use for parallel processing. Default: 1.
        
        - `--metadata STR`
        Additional metadata in the format `colname=value,colname=value,...`. Applies to all samples.
        
        - `--metadata-from-file PATH`
        File containing metadata information in TSV or CSV format.  Each row should have `sample_path,metadata_col,value`.

        ## Examples

        ### Performing QC on Multiple Samples

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig -o qc_results.tsv
        ```

        ### Performing QC with Samples Listed in a File

        ```bash
        snipe qc --ref reference.sig --samples-from-file samples.txt -o qc_results.tsv
        ```

        *Contents of `samples.txt`:*

        ```
        sample1.sig
        sample2.sig
        sample3.sig
        ```

        ### Performing QC with an Amplicon Signature

        ```bash
        snipe qc --ref reference.sig --amplicon amplicon.sig --sample sample1.sig -o qc_results.tsv
        ```

        ### Including Advanced QC Metrics and ROI Calculations

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --advanced --roi -o qc_results.tsv
        ```

        ### Using Multiple Variance Signatures

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --var var1.sig --var var2.sig -o qc_results.tsv
        ```

        ### Overriding the Y Chromosome Signature

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --ychr custom_y.sig -o qc_results.tsv
        ```

        ### Combining Multiple Options

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig --amplicon amplicon.sig --var var1.sig --var var2.sig --advanced --roi -o qc_results.tsv
        ```

        ## Detailed Use Cases

        ### Use Case 1: Basic QC on Single Sample

        **Objective:** Perform QC on a single sample against a reference genome without any advanced metrics or ROI.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig -o qc_basic.tsv
        ```

        **Explanation:**

        - `--ref reference.sig`: Specifies the reference genome signature file.
        - `--sample sample1.sig`: Specifies the sample signature file.
        - `-o qc_basic.tsv`: Specifies the output TSV file for QC results.

        **Expected Output:**

        A TSV file named `qc_basic.tsv` containing basic QC metrics for `sample1.sig`.

        ### Use Case 2: QC on Multiple Samples with ROI

        **Objective:** Perform QC on multiple samples and calculate Regions of Interest (ROI) for each.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig --roi -o qc_roi.tsv
        ```

        **Explanation:**

        - `--ref reference.sig`: Reference genome signature file.
        - `--sample sample1.sig` & `--sample sample2.sig`: Multiple sample signature files.
        - `--roi`: Enables ROI calculations.
        - `-o qc_roi.tsv`: Output file for QC results.

        **Expected Output:**

        A TSV file named `qc_roi.tsv` containing QC metrics along with ROI predictions for `sample1.sig` and `sample2.sig`.

        ### Use Case 3: Advanced QC with Amplicon and Variance Signatures

        **Objective:** Perform advanced QC on a sample using an amplicon signature and multiple variance signatures.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --amplicon amplicon.sig --sample sample1.sig --var var1.sig --var var2.sig --advanced -o qc_advanced.tsv
        ```

        **Explanation:**

        - `--ref reference.sig`: Reference genome signature file.
        - `--amplicon amplicon.sig`: Amplicon signature file.
        - `--sample sample1.sig`: Sample signature file.
        - `--var var1.sig` & `--var var2.sig`: Variance signature files.
        - `--export-var`: Export signatures for variances.
        - `--metadata`: Additional metadata in the format `metadata1=value,metadata2=value,...`.
        - `--metadata-from-file`: File containing metadata information in TSV or CSV format per signature.
        - `-o qc_advanced.tsv`: Output file for QC results.

        **Expected Output:**

        A TSV file named `qc_advanced.tsv` containing comprehensive QC metrics, including advanced metrics and analyses based on the amplicon and variance signatures for `sample1.sig`.

        ### Use Case 4: Overwriting Existing Output File

        **Objective:** Perform QC and overwrite an existing output TSV file.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig -o qc_results.tsv
        ```

        **Explanation:**

        - If `qc_results.tsv` already exists, the command will **fail** to prevent accidental overwriting. To overwrite, use the `--force` flag (assuming you've implemented it; if not, you may need to adjust the `qc` command to include a `--force` option).

        **Adjusted Command with `--force` (if implemented):**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig -o qc_results.tsv --force
        ```

        **Expected Output:**

        The existing `qc_results.tsv` file will be overwritten with the new QC results for `sample1.sig`.

        ### Use Case 5: Using a Custom Y Chromosome Signature

        **Objective:** Override the default Y chromosome signature with a custom one during QC.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --ychr custom_y.sig -o qc_custom_y.tsv
        ```

        **Explanation:**

        - `--ychr custom_y.sig`: Specifies a custom Y chromosome signature file to override the default.

        **Expected Output:**

        A TSV file named `qc_custom_y.tsv` containing QC metrics for `sample1.sig` with analyses based on the custom Y chromosome signature.

        ### Use Case 6: Reading Sample Paths from a File

        **Objective:** Perform QC on multiple samples listed in a text file.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --samples-from-file samples.txt -o qc_from_file.tsv
        ```

        **Explanation:**

        - `--samples-from-file samples.txt`: Specifies a file containing sample paths, one per line.

        **Contents of `samples.txt`:**

        ```
        sample1.sig
        sample2.sig
        sample3.sig
        ```

        **Expected Output:**

        A TSV file named `qc_from_file.tsv` containing QC metrics for `sample1.sig`, `sample2.sig`, and `sample3.sig`.

        ### Use Case 7: Combining Multiple Options for Comprehensive QC

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig --amplicon amplicon.sig --var var1.sig --var var2.sig --advanced --roi -o qc_comprehensive.tsv
        ```

        **Explanation:**

        - `--ref reference.zip`: Reference genome signature file.
        - `--sample sample1.zip` & `--sample sample2.sig`: Multiple sample signature files.
        - `--amplicon amplicon.zip`: Amplicon signature file.
        - `--var var1.zip` & `--var var2.zip`: Variance signature files.
        - `--advanced`: Includes advanced QC metrics.
        - `--roi`: Enables ROI calculations.
        - `-o qc_comprehensive.tsv`: Output file for QC results.

        **Expected Output:**

        A TSV file named `qc_comprehensive.tsv` containing comprehensive QC metrics, including advanced analyses, ROI predictions, and data from amplicon and variance signatures for both `sample1.sig` and `sample2.sig`.
    """
    

    start_time = time.time()

    # Configure logging
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

    if cores > 1 and len(valid_samples) > 1:
        logger.info(f"Parallel processing enabled with {cores} cores.")
        # Split valid_samples into chunks
        chunks = split_chunks(valid_samples, cores)
        logger.debug(f"Splitting samples into {len(chunks)} chunks for parallel processing.")

        # Prepare arguments for each worker
        worker_args = [
            (
                chunk,
                ref,
                amplicon,
                ychr,
                vars,
                export_var,
                roi,
                debug
            )
            for chunk in chunks
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            # Map each chunk to the process_subset function
            futures = {executor.submit(process_subset, *args): idx for idx, args in enumerate(worker_args)}
            # Initialize tqdm with total number of samples
            with tqdm(total=len(valid_samples), desc="Processing samples in parallel") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        subset_stats, subset_failed = future.result()
                        sample_to_stats.update(subset_stats)
                        failed_samples.extend(subset_failed)
                        # Update the progress bar by the number of samples processed in this subset
                        processed_count = len(subset_stats) + len(subset_failed)
                        pbar.update(processed_count)
                    except Exception as e:
                        logger.error(f"A worker failed with exception: {e}")
                        # Retrieve the subset that caused the exception
                        subset_idx = futures[future]
                        subset = worker_args[subset_idx][0]
                        # Update the progress bar by the number of samples in the failed subset
                        pbar.update(len(subset))
                        failed_samples.extend(subset)
                        continue
    else:
        logger.info("Parallel processing not enabled. Processing samples sequentially.")
        # Sequential processing as original
        qc_instance = MultiSigReferenceQC(
                reference_sig=reference_sig,
                amplicon_sig=amplicon_sig,
                ychr=ychr_sig if ychr_sig else None,
                varsigs=vars_snipesigs if vars_snipesigs else None,
                export_varsigs=export_var,
                enable_logging=debug
            )
        
        with tqdm(total=len(valid_samples), desc="Processing samples") as pbar:
            for sample_path in valid_samples:
                sample_sig = SnipeSig(sourmash_sig=sample_path, sig_type=SigType.SAMPLE, enable_logging=debug)
                qc_instance.logger.debug(f"DELME Processing sample: {sample_sig.name}")
                if sample_sig.name == "":
                    _newname = os.path.basename(sample_path).split('.')[0]
                    sample_sig.name = _newname
                    qc_instance.logger.warning(f"Sample name is empty. Setting to: `{sample_sig.name}`")
                
                try:
                    sample_stats = qc_instance.process_sample(
                        sample_sig=sample_sig,
                        predict_extra_folds=predict_extra_folds if roi else None,
                        advanced=True
                    )
                    sample_to_stats[sample_sig.name] = sample_stats
                except Exception as e:
                    failed_samples.append(sample_sig.name)
                    qc_instance.logger.error(f"Failed to process sample {sample_sig.name}: {e}")
                finally:
                    pbar.update(1)

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
            
    
    # santize file_path and filename
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
    floating_3 = ["Mapping index", "contamination", "error"]
    
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
        
    
    try:
        with open(output, 'w', encoding='utf-8') as f:
            header_dict = {"sha256": metadata_md5sum, "metadata": metadata_str}
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
