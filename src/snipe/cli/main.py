import os
import sys
import time
import logging
from typing import Optional, Any, List, Dict, Set

import click
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from snipe.api.enums import SigType
from snipe.api.sketch import SnipeSketch
from snipe.api.snipe_sig import SnipeSig
from snipe.api.reference_QC import ReferenceQC

# pylint: disable=logging-fstring-interpolation

def validate_zip_file(ctx, param, value: str) -> str:
    """
    Validate that the output file has a .zip extension.
    """
    if not value.lower().endswith('.zip'):
        raise click.BadParameter('Output file must have a .zip extension.')
    return value


def validate_tsv_file(ctx, param, value: str) -> str:
    """
    Validate that the output file has a .tsv extension.
    """
    if not value.lower().endswith('.tsv'):
        raise click.BadParameter('Output file must have a .tsv extension.')
    return value


@click.group()
def cli():
    """
    Snipe CLI Tool

    Use this tool to perform various sketching and quality control operations on genomic data.
    """
    pass


@cli.command()
@click.option('-s', '--sample', type=click.Path(exists=True), help='Sample FASTA file.')
@click.option('-r', '--ref', type=click.Path(exists=True), help='Reference genome FASTA file.')
@click.option('-a', '--amplicon', type=click.Path(exists=True), help='Amplicon FASTA file.')
@click.option('--ychr', type=click.Path(exists=True), help='Y chromosome signature file (required for --ref and --amplicon).')
@click.option('-n', '--name', required=True, help='Signature name.')
@click.option('-o', '--output-file', required=True, callback=validate_zip_file, help='Output file with .zip extension.')
@click.option('-b', '--batch-size', default=100000, type=int, show_default=True, help='Batch size for sample sketching.')
@click.option('-c', '--cores', default=4, type=int, show_default=True, help='Number of CPU cores to use.')
@click.option('-k', '--ksize', default=51, type=int, show_default=True, help='K-mer size.')
@click.option('-f', '--force', is_flag=True, help='Overwrite existing output file.')
@click.option('--scale', default=10000, type=int, show_default=True, help='sourmash scale factor.')
@click.option('--debug', is_flag=True, default=False, hidden=True, help='Enable debugging.')
@click.pass_context
def sketch(ctx, sample: Optional[str], ref: Optional[str], amplicon: Optional[str],
           ychr: Optional[str], name: str, output_file: str, batch_size: int,
           cores: int, ksize: int, scale: int, force: bool, debug: bool):
    """
    Perform sketching operations.

    You must specify exactly one of --sample, --ref, or --amplicon.
    """
    # Ensure mutual exclusivity
    samples = [sample, ref, amplicon]
    provided = [s for s in samples if s]
    if len(provided) != 1:
        click.echo('Error: Exactly one of --sample, --ref, or --amplicon must be provided.')
        sys.exit(1)

    # Handle existing output file
    if os.path.exists(output_file):
        if force:
            try:
                os.remove(output_file)
                click.echo(f"Overwriting existing output file {output_file}.")
            except Exception as e:
                click.echo(f"Failed to remove existing file {output_file}: {e}")
                sys.exit(1)
        else:
            click.echo(f"Output file {output_file} already exists. Please provide a new output file or use --force to overwrite.")
            sys.exit(1)

    # Determine the mode and set SigType accordingly
    if sample:
        mode = 'sample'
        sig_type = SigType.SAMPLE
        input_file = sample
    elif ref:
        mode = 'ref'
        sig_type = SigType.GENOME
        input_file = ref
    elif amplicon:
        mode = 'amplicon'
        sig_type = SigType.AMPLICON
        input_file = amplicon
    else:
        # This should not happen due to mutual exclusivity check
        raise click.UsageError('You must specify one of --sample, --ref, or --amplicon.')

    # Instantiate and execute SnipeSketch with logging based on debug flag
    try:
        sketcher = SnipeSketch(enable_logging=debug)
    except Exception as e:
        click.echo(f'Error instantiating SnipeSketch: {e}')
        sys.exit(1)

    # Sketching start time
    start_time = time.time()

    try:
        if sig_type == SigType.SAMPLE:
            # Modify sample name for clarity
            snipe_modification_to_name = f"{name}-snipesample"
            # Perform sample sketching
            sample_signature = sketcher.sample_sketch(
                filename=sample,
                sample_name=snipe_modification_to_name,
                num_processes=cores,
                batch_size=batch_size,
                ksize=ksize,
                scale=scale,
            )
            # Export the signature to a ZIP file
            SnipeSketch.export_sigs_to_zip(
                [sample_signature],
                output_file,
            )

        elif sig_type == SigType.GENOME:
            # Print sketching parameters for user information
            click.echo(f"Sketching genome with ksize={ksize}, scale={scale}, cores={cores}")

            snipe_modification_to_name = f"{name}-snipegenome"
            # Perform genome sketching
            genome_sig, chr_to_sig = sketcher.parallel_genome_sketching(
                fasta_file=ref,
                cores=cores,
                ksize=ksize,
                scale=scale,
                assigned_genome_name=snipe_modification_to_name,
            )

            # Rename genome signature for consistency
            genome_sig.name = snipe_modification_to_name

            # If Y chromosome is provided, perform amplicon sketching
            if ychr:
                if not sig_type == SigType.GENOME:
                    click.echo("Error: --ychr is only applicable with --ref.")
                    sys.exit(1)
                snipe_ychr_name = "sex-y"
                y_chr_sig = sketcher.amplicon_sketching(
                    fasta_file=ychr,
                    ksize=ksize, 
                    scale=scale,
                    amplicon_name=snipe_ychr_name,
                )
                chr_to_sig[snipe_ychr_name] = y_chr_sig

            # Log the detected chromosomes
            autosomal = [name for name in chr_to_sig.keys() if "autosome" in name.lower()]
            sex = [name for name in chr_to_sig.keys() if "sex" in name.lower()]

            click.echo("Autodetected chromosomes:")
            for i, chr_name in enumerate(chr_to_sig.keys(), 1):
                print(f"{chr_name}", end="\t")
                if i % 6 == 0:
                    click.echo()
            if len(chr_to_sig) % 6 != 0:
                click.echo()  # For newline after the last line

            click.echo(f"Autosomal chromosomes: {len(autosomal)}, Sex chromosomes: {len(sex)}")
            
            # Export genome and chromosome signatures to a ZIP file
            SnipeSketch.export_sigs_to_zip(
                [genome_sig] + list(chr_to_sig.values()),
                output_file,
            )

        elif sig_type == SigType.AMPLICON:
            snipe_modification_to_name = f"{name}-snipeamplicon"
            # Perform amplicon sketching
            amplicon_signature = sketcher.amplicon_sketching(
                fasta_file=amplicon,
                ksize=ksize,
                scale=scale,
                amplicon_name=snipe_modification_to_name,
            )
            # Export the signature to a ZIP file
            SnipeSketch.export_sigs_to_zip(
                [amplicon_signature],
                output_file,
            )

    except Exception as e:
        click.echo(f"Error during sketching: {e}")
        sys.exit(1)

    # Sketching end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    click.echo(f"Sketching completed in {elapsed_time:.2f} seconds.")


# Define the top-level process_sample function
def process_sample(sample_path: str, ref_path: str, amplicon_path: Optional[str],
                  advanced: bool, roi: bool, debug: bool) -> Dict[str, Any]:
    """
    Process a single sample for QC.

    Parameters:
    - sample_path (str): Path to the sample signature file.
    - ref_path (str): Path to the reference signature file.
    - amplicon_path (Optional[str]): Path to the amplicon signature file.
    - advanced (bool): Flag to include advanced metrics.
    - roi (bool): Flag to calculate ROI.
    - debug (bool): Flag to enable debugging.

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

        # Load reference signature
        reference_sig = SnipeSig(sourmash_sig=ref_path, sig_type=SigType.GENOME, enable_logging=debug)
        logger.debug(f"Loaded reference signature: {reference_sig.name}")

        # Load amplicon signature if provided
        amplicon_sig = None
        if amplicon_path:
            amplicon_sig = SnipeSig(sourmash_sig=amplicon_path, sig_type=SigType.AMPLICON, enable_logging=debug)
            logger.debug(f"Loaded amplicon signature: {amplicon_sig.name}")

        # Instantiate ReferenceQC
        qc_instance = ReferenceQC(
            sample_sig=sample_sig,
            reference_sig=reference_sig,
            amplicon_sig=amplicon_sig,
            enable_logging=debug
        )
        
        # calculate chromosome metrics
        qc_instance.calculate_chromosome_metrics()

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


@cli.command()
@click.option('--ref', type=click.Path(exists=True), required=True, help='Reference genome signature file (required).')
@click.option('--sample', type=click.Path(exists=True), multiple=True, help='Sample signature file. Can be provided multiple times.')
@click.option('--samples-from-file', type=click.Path(exists=True), help='File containing sample paths (one per line).')
@click.option('--amplicon', type=click.Path(exists=True), help='Amplicon signature file (optional).')
@click.option('--roi', is_flag=True, default=False, help='Calculate ROI for 1,2,5,9 folds.')
@click.option('--cores', '-c', default=4, type=int, show_default=True, help='Number of CPU cores to use for parallel processing.')
@click.option('--advanced', is_flag=True, default=False, help='Include advanced QC metrics.')
@click.option('--debug', is_flag=True, default=False, help='Enable debugging and detailed logging.')
@click.option('-o', '--output', required=True, callback=validate_tsv_file, help='Output TSV file for QC results.')
def qc(ref: str, sample: List[str], samples_from_file: Optional[str],
       amplicon: Optional[str], roi: bool, cores: int, advanced: bool,
       debug: bool, output: str):
    """
    Perform quality control (QC) on multiple samples against a reference genome.

    This command calculates various QC metrics for each provided sample, optionally including advanced metrics
    and ROI predictions. Results are aggregated and exported to a TSV file.
    """
    start_time = time.time()

    # Configure logging
    logger = logging.getLogger('snipe_qc')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    logger.info("Starting QC process.")

    # Collect sample paths from --sample and --samples-from-file
    samples_set: Set[str] = set(sample)  # Start with samples provided via --sample

    if samples_from_file:
        logger.debug(f"Reading samples from file: {samples_from_file}")
        try:
            with open(samples_from_file, encoding='utf-8') as f:
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

    # Prepare arguments for parallel processing
    process_args = [
        (sample_path, ref, amplicon, advanced, roi, debug)
        for sample_path in valid_samples
    ]

    # Process samples in parallel with progress bar
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_sample, *args): args[0] for args in process_args
        }
        # Iterate over completed futures with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
            sample = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logger.error(f"Sample {sample} generated an exception: {exc}")
                results.append({
                    "sample": os.path.splitext(os.path.basename(sample))[0],
                    "file_path": sample,
                    "QC_Error": str(exc)
                })

    # Create pandas DataFrame
    logger.info("Aggregating results into DataFrame.")
    df = pd.DataFrame(results)

    # Reorder columns to have 'sample' and 'file_path' first, if they exist
    cols = list(df.columns)
    reordered_cols = []
    for col in ['sample', 'file_path']:
        if col in cols:
            reordered_cols.append(col)
            cols.remove(col)
    reordered_cols += cols
    df = df[reordered_cols]

    # Export to TSV
    try:
        df.to_csv(output, sep='\t', index=False)
        logger.info(f"QC results successfully exported to {output}")
    except Exception as e:
        logger.error(f"Failed to export QC results to {output}: {e}")
        sys.exit(1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"QC process completed in {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    cli()
