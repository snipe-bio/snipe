import click
import importlib.metadata

from snipe.cli.cli_qc import qc as cli_qc
from snipe.cli.cli_sketch import sketch as cli_sketch
from snipe.cli.cli_ops import ops as cli_ops
from snipe.cli.cli_qc_parallel import parallel_qc

# pylint: disable=logging-fstring-interpolation

@click.group()
def cli():
    """
    Snipe CLI Tool

    Commands:
    - `sketch`: Perform sketching operations on genomic data.
    - `qc`: Execute quality control (QC) on multiple samples against a reference genome.
    - `ops`: Perform various operations on sketches.
    """
    pass


cli.add_command(cli_qc)
cli.add_command(cli_sketch)
cli.add_command(cli_ops)

# Load plugin commands
def load_plugins():
    try:
        entry_points = importlib.metadata.entry_points()
        if hasattr(entry_points, 'select'):
            plugins = entry_points.select(group='snipe.plugins')
        else:
            plugins = entry_points.get('snipe.plugins', [])
        
        for entry_point in plugins:
            plugin = entry_point.load()
            cli.add_command(plugin)
    
    except Exception as e:
        click.echo(f"Error loading plugins: {e}", err=True)


load_plugins()

if __name__ == '__main__':
    cli()
