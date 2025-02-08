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


def validate_zip_file(ctx, param, value: str) -> str:
    if not value.lower().endswith('.zip'):
        raise click.BadParameter('Output file must have a .zip extension.')
    return value



@click.command()
@click.option('-s', '--sample', type=click.Path(exists=True), help='Sample FASTA file.')
@click.option('-r', '--ref', type=click.Path(exists=True), help='Reference genome FASTA file.')
@click.option('-a', '--amplicon', type=click.Path(exists=True), help='Amplicon FASTA file.')
@click.option('--ychr', type=click.Path(exists=True), help='Y chromosome FASTA file (overrides the reference ychr).')
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
    Sketch genomic data to generate signatures for QC.

    Only one input type (`--sample`, `--ref`, or `--amplicon`) can be specified at a time.

    ## Usage

    ```bash
    snipe sketch [OPTIONS]
    ```

    ## Options

    - `-s`, `--sample PATH`  
      Sample FASTA file.

    - `-r`, `--ref PATH`  
      Reference genome FASTA file.

    - `-a`, `--amplicon PATH`  
      Amplicon FASTA file.

    - `--ychr PATH`  
      Y chromosome FASTA file (overrides the reference ychr).

    - `-n`, `--name TEXT` **[required]**  
      Signature name.

    - `-o`, `--output-file PATH` **[required]**  
      Output file with `.zip` extension.

    - `-b`, `--batch-size INTEGER` **[default: 100000]**  
      Batch size for sample sketching.

    - `-c`, `--cores INTEGER` **[default: 4]**  
      Number of CPU cores to use.

    - `-k`, `--ksize INTEGER` **[default: 51]**  
      K-mer size.

    - `-f`, `--force`  
      Overwrite existing output file.

    - `--scale INTEGER` **[default: 10000]**  
      Sourmash scale factor.

    - `--debug`  
      Enable debugging.

    ## Examples

    ### Sketching a Reference Genome

    Generate a sketch from a reference genome FASTA file.

    ```bash
    snipe sketch -r reference.fasta -n genome_ref -o genome_output.zip --cores 8 --debug
    ```

    ### Sketching a Sample FASTA File

    Generate a sketch from a sample FASTA file.

    ```bash
    snipe sketch -s sample.fasta -n sample_name -o sample_output.zip
    ```

    ### Sketching an Amplicon FASTA File with Custom Batch Size

    ```bash
    snipe sketch -a amplicon.fasta -n amplicon_name -o amplicon_output.zip -b 50000
    ```

    ### Overwriting an Existing Output File

    ```bash
    snipe sketch -r reference.fasta -n genome_ref -o genome_output.zip --force
    ```

    ### Using a Custom Y Chromosome File

    ```bash
    snipe sketch -r reference.fasta -n genome_ref -o genome_output.zip --ychr custom_y.fasta
    ```
    """
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
            sample_signature, total_bases, total_valid_kmers = sketcher.sample_sketch(
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
            
            click.echo(
                f"Sample {snipe_modification_to_name} sketching completed with {total_bases} bases "
                f"and {total_valid_kmers} valid k-mers."
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
            mitochondrial = [name for name in chr_to_sig.keys() if "mito" in name.lower()]

            click.echo("Autodetected chromosomes:")
            for i, chr_name in enumerate(chr_to_sig.keys(), 1):
                print(f"{chr_name}", end="\t")
                if i % 6 == 0:
                    click.echo()
            if len(chr_to_sig) % 6 != 0:
                click.echo()  # For newline after the last line

            click.echo(f"Autosomal chromosomes: {len(autosomal)}, Sex chromosomes: {len(sex)}, Mitochondrial: {len(mitochondrial)}")
            
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
