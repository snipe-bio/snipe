import os
import sys
import time
import logging
from typing import Optional, Any

import click

from snipe.api.enums import SigType
from snipe.api.sketch import SnipeSketch


def validate_zip_file(ctx, param, value: str) -> str:
    """
    Validate that the output file has a .zip extension.

    Args:
        ctx: Click context.
        param: Click parameter.
        value (str): The value of the parameter.

    Raises:
        click.BadParameter: If the file does not have a .zip extension.

    Returns:
        str: The validated file path.
    """
    if not value.lower().endswith('.zip'):
        raise click.BadParameter('Output file must have a .zip extension.')
    return value


def ensure_mutually_exclusive(ctx, param, value: Any) -> Any:
    """
    Ensure that only one of --sample, --ref, or --amplicon is provided.

    Args:
        ctx: Click context.
        param: Click parameter.
        value (Any): The value of the parameter.

    Raises:
        click.UsageError: If more than one or none of the mutually exclusive options are provided.

    Returns:
        Any: The validated value.
    """
    sample, ref, amplicon = ctx.params.get('sample'), ctx.params.get('ref'), ctx.params.get('amplicon')
    if sum([bool(sample), bool(ref), bool(amplicon)]) > 1:
        raise click.UsageError('Only one of --sample, --ref, or --amplicon can be used at a time.')
    if not any([bool(sample), bool(ref), bool(amplicon)]):
        raise click.UsageError('You must specify one of --sample, --ref, or --amplicon.')
    return value


@click.group()
def cli():
    """
    Snipe CLI Tool

    Use this tool to perform various sketching operations on genomic data.
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
    ensure_mutually_exclusive(ctx, None, None)

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
            autosomal = [name for name in chr_to_sig.keys() if "autosome" in name]
            sex = [name for name in chr_to_sig.keys() if "sex" in name]

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


# Add sketch command to cli
cli.add_command(sketch)


# Example placeholder for future 'qc' command
@cli.group()
def qc():
    """
    Perform quality control operations.
    """
    pass


@qc.command()
def run_qc():
    """
    Run quality control checks.
    """
    click.echo('Quality control functionality is not yet implemented.')
    # Future implementation goes here


if __name__ == '__main__':
    cli()
