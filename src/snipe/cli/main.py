import os
import sys
from snipe.api import SigType, SnipeSig, ReferenceQC
from snipe.api.sketch import SnipeSketch
import click
import sourmash
import time

def validate_zip_file(ctx, param, value):
    if not value.lower().endswith('.zip'):
        raise click.BadParameter('Output file must have a .zip extension.')
    return value

def ensure_mutually_exclusive(ctx, param, value):
    """Ensure that only one of --sample, --ref, or --amplicon is provided."""
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

@click.command()
@click.option('-s', '--sample', type=click.Path(exists=True), help='Sample fasta file.')
@click.option('-r', '--ref', type=click.Path(exists=True), help='Reference genome fasta file.')
@click.option('-a', '--amplicon', type=click.Path(exists=True), help='Amplicon fasta file.')
@click.option('--ychr', type=click.Path(exists=True), help='Y chromosome signature file (required for --ref and --amplicon).')
@click.option('-n', '--name', required=True, help='Signature name.')
@click.option('-o', '--output-file', required=True, callback=validate_zip_file, help='Output file with .zip extension.')
@click.option('-b', '--batch-size', default=100000, type=int, show_default=True, help='Batch size for sample sketching.')
@click.option('-c', '--cores', default=4, type=int, show_default=True, help='Number of CPU cores to use.')
@click.option('-k', '--ksize', default=51, type=int, show_default=True, help='K-mer size.')
@click.option('-f', '--force', is_flag=True, help='Overwrite existing output file.')
@click.option('--scale', default=10000, type=int, show_default=True, help='sourmash scale factor.')
@click.pass_context
def sketch(ctx, sample, ref, amplicon, ychr, name, output_file, batch_size, cores, ksize, scale, force):
    """
    Perform sketching operations.

    You must specify exactly one of --sample, --ref, or --amplicon.
    """
    # Ensure mutual exclusivity
    ensure_mutually_exclusive(ctx, None, None)
    
    if force:
        if os.path.exists(output_file):
            os.remove(output_file)
            click.echo(f"Overwriting existing output file {output_file}.")    
    else:
        if os.path.exists(output_file):
            click.echo(f"Output file {output_file} already exists. Please provide a new output file.")
            sys.exit(1)

    # Determine the mode and set SigType accordingly
    sig_type: SigType = None
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
        raise click.UsageError(
            'You must specify one of --sample, --ref, or --amplicon.'
        )

    # Instantiate and execute SnipeSketch
    try:
        sketcher = SnipeSketch(enable_logging=False)
    except Exception as e:
        click.echo(f'Error instantiating SnipeSketch: {e}')
        sys.exit(1)
    
    # sketching start time
    start_time = time.time()

    if sig_type == SigType.SAMPLE:
        snipe_modification_to_name = f"{name}-snipesample"
        # sample_name, filename, num_processes, batch_size, ksize, scale
        sample_signature = sketcher.sample_sketch(
            filename=sample,
            sample_name=snipe_modification_to_name,
            num_processes=cores,
            batch_size=batch_size,
            ksize=ksize,
            scale=scale,
        )
        
        SnipeSketch.export_sigs_to_zip(
            [sample_signature],
            output_file,
        )

    elif sig_type == SigType.GENOME:
        # print skethcing parameters
        click.echo(f"Sketching genome with ksize={ksize}, scale={scale}, cores={cores}")
        
        snipe_modification_to_name = f"{name}-snipegenome"
        genome_sig, chr_to_sig = sketcher.parallel_genome_sketching(
            fasta_file = ref,
            cores=cores,
            ksize=ksize,
            scale=scale, 
            assigned_genome_name=snipe_modification_to_name,
        )
        
        # make sure genome name is snipe_modification_to_name
        genome_sig.name = snipe_modification_to_name
        # if y chr
        if ychr:
            snipe_ychr_name = "sex-y"
            y_chr_sig = sketcher.amplicon_sketching(
                fasta_file = ychr,
                ksize = ksize, 
                scale = scale,
                amplicon_name = snipe_ychr_name,
            )
            chr_to_sig[snipe_ychr_name] = y_chr_sig
            
        # log the user the detected chromosomes (nice view, 3 per row)
        click.echo("Autodetected chromosomes:")
        count_autosomals = 0
        count_sex = 0
        for i, chr_name in enumerate(chr_to_sig.keys()):
            if "autosome" in chr_name: count_autosomals += 1
            elif "sex" in chr_name: count_sex += 1
            print(f"{chr_name}", end="\t")
            if (i + 1) % 6 == 0:
                click.echo()
        click.echo(f"\nAutosomal chromosomes: {count_autosomals}, Sex chromosomes: {count_sex}")

        SnipeSketch.export_sigs_to_zip(
            [genome_sig, *chr_to_sig.values()],
            output_file,
        )
    
    elif sig_type == SigType.AMPLICON:
        snipe_modification_to_name = f"{name}-snipeamplicon"
        amplicon_signature = sketcher.amplicon_sketching(
            fasta_file=amplicon,
            ksize=ksize,
            scale=scale,
            amplicon_name=name,
        )
        
        SnipeSketch.export_sigs_to_zip(
            [amplicon_signature],
            output_file,
        )
        
    # sketching end time
    end_time = time.time()
    click.echo(f"Sketching completed in {end_time - start_time:.2f} seconds.")
        



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
