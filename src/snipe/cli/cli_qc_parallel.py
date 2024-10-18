import os
import sys
import time
import logging
from typing import Optional, Any, List, Dict, Set, Union

import click
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from snipe.api.enums import SigType
from snipe.api.sketch import SnipeSketch
from snipe.api.snipe_sig import SnipeSig
from snipe.api.reference_QC import ReferenceQC

import signal


def validate_sig_input(ctx, param, value: tuple) -> str:
    supported_extensions = ['.zip', '.sig']
    for path in value:
        if not os.path.exists(path):
            raise click.BadParameter(f"File not found: {path}")
        if not any(path.lower().endswith(ext) for ext in supported_extensions):
            raise click.BadParameter(f"Unsupported file format: {path}, supported formats: {', '.join(supported_extensions)}")
    return value

def process_sample(sample_path: str, reference_sig: SnipeSig, amplicon_sig: Optional[SnipeSig],
                  advanced: bool, roi: bool, debug: bool,
                  ychr_sig: Optional[SnipeSig] = None,
                  var_sigs: Optional[List[SnipeSig]] = None) -> Dict[str, Any]:
    """
    Process a single sample for QC.

    Parameters:
    - sample_path (str): Path to the sample signature file.
    - ref_path (str): Path to the reference signature file.
    - amplicon_path (Optional[str]): Path to the amplicon signature file.
    - advanced (bool): Flag to include advanced metrics.
    - roi (bool): Flag to calculate ROI.
    - debug (bool): Flag to enable debugging.
    - vars_paths (Optional[Union[List[str], List[SnipeSig]]]): List of paths to variable signature files or SnipeSig objects.

    Returns:
    - Dict[str, Any]: QC results for the sample.
    """
    # Configure worker-specific logging
    logger = logging.getLogger(f'snipe_qc_worker_{os.path.basename(sample_path)}')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    sample_name = os.path.splitext(os.path.basename(sample_path))[0]
    try:
        # Load sample signature
        sample_sig = SnipeSig(sourmash_sig=sample_path, sig_type=SigType.SAMPLE, enable_logging=debug)
        logger.debug(f"Loaded sample signature: {sample_sig.name}")

        # Instantiate ReferenceQC
        qc_instance = ReferenceQC(
            sample_sig=sample_sig,
            reference_sig=reference_sig,
            amplicon_sig=amplicon_sig,
            enable_logging=debug
        )
        
        # Load variable signatures if provided
        if var_sigs:
            qc_instance.logger.debug(f"Loading {len(var_sigs)} variable signature(s).")
            vars_dict: Dict[str, SnipeSig] = {sig.name: sig for sig in var_sigs}
            qc_instance.logger.debug(f"vars_dict: {vars_dict}")
            vars_order: List[str] = []
                            
            qc_instance.logger.debug("Loading variable signature(s) from SnipeSig objects.")
            for snipe_name in vars_dict.keys():
                vars_order.append(snipe_name)
                qc_instance.logger.debug(f"Loaded variable signature: {snipe_name}")
                
            # log keys of vars_dict and vars_order
            qc_instance.logger.debug(f"vars_dict keys: {vars_dict.keys()}")
            qc_instance.logger.debug(f"vars_order: {vars_order}")
            
            qc_instance.nonref_consume_from_vars(vars=vars_dict, vars_order=vars_order)
            
        # No else block needed; variables are optional

        # Calculate chromosome metrics
        # genome_chr_to_sig: Dict[str, SnipeSig] = qc_instance.load_genome_sig_to_dict(zip_file_path = ref_path)
        chr_to_sig = reference_sig.chr_to_sig.copy()
        if ychr_sig:
            chr_to_sig['y'] = ychr_sig
        
        qc_instance.calculate_chromosome_metrics(chr_to_sig)
        qc_instance.calculate_sex_chrs_metrics(chr_to_sig)
        # Get aggregated stats
        aggregated_stats = qc_instance.get_aggregated_stats(include_advanced=advanced)

        # Initialize result dict
        result = {
            "sample": sample_name,
            "file_path": os.path.abspath(sample_path),
        }
        # Add aggregated stats
        result.update(aggregated_stats)

        # Calculate ROI if requested
        if roi:
            logger.debug(f"Calculating ROI for sample: {sample_name}")
            for fold in [1, 2, 5, 9]:
                try:
                    predicted_coverage = qc_instance.predict_coverage(extra_fold=fold)
                    result[f"Predicted_Coverage_Fold_{fold}x"] = predicted_coverage
                    logger.debug(f"Fold {fold}x: Predicted Coverage = {predicted_coverage}")
                except RuntimeError as e:
                    logger.error(f"ROI calculation failed for sample {sample_name} at fold {fold}x: {e}")
                    result[f"Predicted_Coverage_Fold_{fold}x"] = None

        return result

    except Exception as e:
        logger.error(f"QC failed for sample {sample_path}: {e}")
        return {
            "sample": sample_name,
            "file_path": os.path.abspath(sample_path),
            "QC_Error": str(e)
        }
        
def validate_tsv_file(ctx, param, value: str) -> str:
    if not value.lower().endswith('.tsv'):
        raise click.BadParameter('Output file must have a .tsv extension.')
    return value


@click.command()
@click.option('--ref', type=click.Path(exists=True), required=True, help='Reference genome signature file (required).')
@click.option('--sample', type=click.Path(exists=True), callback=validate_sig_input, multiple=True, help='Sample signature file. Can be provided multiple times.')
@click.option('--samples-from-file', type=click.Path(exists=True), help='File containing sample paths (one per line).')
@click.option('--amplicon', type=click.Path(exists=True), help='Amplicon signature file (optional).')
@click.option('--roi', is_flag=True, default=False, help='Calculate ROI for 1,2,5,9 folds.')
@click.option('--cores', '-c', default=4, type=int, show_default=True, help='Number of CPU cores to use for parallel processing.')
@click.option('--advanced', is_flag=True, default=False, help='Include advanced QC metrics.')
@click.option('--ychr', type=click.Path(exists=True), help='Y chromosome signature file (overrides the reference ychr).')
@click.option('--debug', is_flag=True, default=False, help='Enable debugging and detailed logging.')
@click.option('-o', '--output', required=True, callback=validate_tsv_file, help='Output TSV file for QC results.')
@click.option('--var', 'vars', multiple=True, type=click.Path(exists=True), help='Variable signature file path. Can be used multiple times.')
def parallel_qc(ref: str, sample: List[str], samples_from_file: Optional[str],
       amplicon: Optional[str], roi: bool, cores: int, advanced: bool, 
       ychr: Optional[str], debug: bool, output: str, vars: List[str]):
    """
        Parallelized version of the `qc` command (not optimized for memory).
    """
    
    
    start_time = time.time()

    # Configure logging
    logger = logging.getLogger('snipe_qc_parallel')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    logger.info("Starting QC process.")
    
    # warning of high memory usage if used large var files
    if vars:
        logger.warning("Using large variable signature files may result in high memory usage.")


    # Collect sample paths from --sample and --samples-from-file
    samples_set: Set[str] = set()
    if sample:
        samples_set.update(sample)

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

    # Prepare variable signatures if provided
    vars_paths = []
    vars_snipesigs = []
    if vars:
        logger.info(f"Loading {len(vars)} variable signature(s).")
        for path in vars:
            if not os.path.exists(path):
                logger.error(f"Variable signature file does not exist: {path}")
                sys.exit(1)
            vars_paths.append(os.path.abspath(path))
            try:
                var_sig = SnipeSig(sourmash_sig=path, sig_type=SigType.AMPLICON, enable_logging=debug)
                vars_snipesigs.append(var_sig)
                logger.debug(f"Loaded variable signature: {var_sig.name}")
            except Exception as e:
                logger.error(f"Failed to load variable signature from {path}: {e}")
                
        logger.debug(f"Variable signature paths: {vars_paths}")

    # Prepare arguments for process_sample function    
    dict_process_args = []
    for sample_path in valid_samples:
        dict_process_args.append({
            "sample_path": sample_path,
            "reference_sig": reference_sig,
            "amplicon_sig": amplicon_sig,
            "advanced": advanced,
            "roi": roi,
            "debug": debug,
            "ychr_sig": ychr_sig,
            "var_sigs": vars_snipesigs #vars_paths
        })
        
    results = []
    
    # Define a handler for forceful shutdown
    executor = None
    def shutdown(signum, frame):
        logger.warning("Shutdown signal received. Terminating all worker processes...")
        global executor
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        os._exit(1)  # Forcefully terminate the program

    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as exec_instance:
            executor = exec_instance  # Assign the executor instance
            futures = {
                executor.submit(process_sample, **args): args for args in dict_process_args
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
                sample = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.error(f"Sample {sample['sample_path']} generated an exception: {exc}")
                    results.append({
                        "sample": os.path.splitext(os.path.basename(sample['sample_path']))[0],
                        "file_path": sample['sample_path'],
                        "QC_Error": str(exc)
                    })
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Shutting down...")
        os._exit(1)  # Forcefully terminate the program
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        os._exit(1)  # Forcefully terminate the program
    finally:
        executor = None  # Reset executor

    # Separate successful and failed results
    succeeded = [res for res in results if "QC_Error" not in res]
    failed = [res for res in results if "QC_Error" in res]

    # Handle complete failure
    if len(succeeded) == 0:
        logger.error("All samples failed during QC processing. Output TSV will not be generated.")
        sys.exit(1)
        
    # write total success and failure
    logger.info("Successfully processed samples: %d", len(succeeded))

    # Prepare the command-line invocation for comments
    command_invocation = ' '.join(sys.argv)

    # Create pandas DataFrame for succeeded samples
    df = pd.DataFrame(succeeded)

    # Reorder columns to have 'sample' and 'file_path' first, if they exist
    cols = list(df.columns)
    reordered_cols = []
    for col in ['sample', 'file_path']:
        if col in cols:
            reordered_cols.append(col)
            cols.remove(col)
    reordered_cols += cols
    df = df[reordered_cols]

    # Export to TSV with comments
    try:
        with open(output, 'w', encoding='utf-8') as f:
            # Write comment with command invocation
            f.write(f"# Command: {command_invocation}\n")
            # Write the DataFrame to the file
            df.to_csv(f, sep='\t', index=False)
        logger.info(f"QC results successfully exported to {output}")
    except Exception as e:
        logger.error(f"Failed to export QC results to {output}: {e}")
        sys.exit(1)

    # Report failed samples if any
    if failed:
        failed_samples = [res['sample'] for res in failed]
        logger.warning(f"The following {len(failed_samples)} sample(s) failed during QC processing:")
        for sample in failed_samples:
            logger.warning(f"- {sample}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"QC process completed in {elapsed_time:.2f} seconds.")