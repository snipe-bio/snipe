import click
from . import __version__
from . import Signature, SigType
import os

class SpecialHelpOrder(click.Group):
    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super(SpecialHelpOrder, self).__init__(*args, **kwargs)

    def get_help(self, ctx):
        orig_list_commands = self.list_commands
        self.list_commands = self.list_commands_for_help
        help_text = super(SpecialHelpOrder, self).get_help(ctx)
        self.list_commands = orig_list_commands
        return help_text

    def list_commands_for_help(self, ctx):
        """Reorder the list of commands when listing the help."""
        commands = super(SpecialHelpOrder, self).list_commands(ctx)
        return sorted(
            commands,
            key=lambda command: self.help_priorities.get(command, 1)
        )

    def command(self, *args, **kwargs):
        """Capture a priority for listing command names in help."""
        help_priority = kwargs.pop('help_priority', 1)

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            self.help_priorities[cmd.name] = help_priority
            return cmd

        return decorator


@click.group(cls=SpecialHelpOrder)
@click.version_option(version=__version__, prog_name="snipe")
@click.option("-q", "--quiet", default=False, is_flag=True)
@click.pass_context
def cli(ctx, quiet):
    pass

def validate_samples(ctx, param, value):
    validated_files = []
    for entry in value:
        if os.path.isdir(entry):
            # If entry is a directory, list and validate only .sig files
            for file in os.listdir(entry):
                full_path = os.path.join(entry, file)
                if full_path.endswith('.sig'):
                    validated_files.append(full_path)
        elif os.path.isfile(entry):
            # Check if the file is a regular file with paths listed
            try:
                with open(entry, 'r') as f:
                    lines = f.readlines()
                    # Assuming if first line ends with '.sig', it's a list file
                    if lines and lines[0].strip().endswith('.sig'):
                        for line in lines:
                            line = line.strip()
                            if os.path.exists(line) and line.endswith('.sig'):
                                validated_files.append(line)
                            else:
                                raise click.BadParameter(f"File {line} does not exist or is not a .sig file")
                    else:
                        # Direct .sig file
                        if entry.endswith('.sig'):
                            validated_files.append(entry)
                        else:
                            raise click.BadParameter("Only .sig files are allowed")
            except UnicodeDecodeError:
                # File is not a text file, thus handle as a .sig file
                if entry.endswith('.sig'):
                    validated_files.append(entry)
                else:
                    raise click.BadParameter("Only .sig files are allowed")
        else:
            raise click.BadParameter(f"Unsupported file or path: {entry}")
    return validated_files


@cli.command(name="align", help_priority=2)
@click.option('--genome', '-g', type=click.Path(exists=True), help='Reference signature', required=True)
@click.option('--amplicon', '-a', multiple=True, type=click.Path(exists=True), help='Amplicon signature')
@click.option('--ksize', '-k', type=int, default=51, help='K-mer size for the signature')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.argument('samples', nargs=-1, callback=validate_samples)
@click.pass_context
def cli_align(ctx, samples, genome, amplicon, output, ksize):
    print("Snipe will align the samples to the reference and amplicon signatures")
    print("Context:", ctx.obj)
    print("Samples:", samples)
    print("Genome:", genome)
    print("Amplicon:", amplicon)
    
    # 1. Loading the reference genome signature
    snipe_genome = Signature(ksize, SigType.GENOME)
    snipe_genome.load_from_path(path=genome)
    
    # 2. Loading all amplicons
    amplicons = []
    for amplicon_path in amplicon:
        snipe_amplicon = Signature(ksize, SigType.AMPLICON)
        snipe_amplicon.load_from_path(path=amplicon_path)
        amplicons.append(snipe_amplicon)
    
    # 3. Aligning the samples
    all_samples = {}
    for sample in samples:
        snipe_sample = Signature(ksize, SigType.SAMPLE)
        snipe_sample.load_from_path(path=sample)
        snipe_sample.add_reference_signature(snipe_genome)
        for amplicon in amplicons:
            snipe_sample.add_amplicon_signature(amplicon)
        
        all_samples[snipe_sample.name] = snipe_sample
        
    
    # debug
    # for name, sample in all_samples.items():
    #     print(sample.)
        


if __name__ == "__main__":
    cli()
