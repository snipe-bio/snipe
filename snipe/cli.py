import click
from . import __version__
from .utils import Signature


class SpecialHelpOrder(click.Group):
    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super(SpecialHelpOrder, self).__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super(SpecialHelpOrder, self).get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super(SpecialHelpOrder, self).list_commands(ctx)
        return (
            c[1]
            for c in sorted(
                (self.help_priorities.get(command, 1), command) for command in commands
            )
        )

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop("help_priority", 1)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator


@click.group(cls=SpecialHelpOrder)
@click.version_option(version=__version__, prog_name="snipe")
@click.option("-q", "--quiet", default=False, is_flag=True)
@click.pass_context
def cli(ctx, quiet):
    pass


@cli.command(name="cluster", help_priority=1)
@click.option(
    "-m",
    required=False,
    type=click.FloatRange(0, 1, clamp=False),
    default=0.0,
    show_default=True,
    help="cluster sequences with (containment > cutoff)",
)
@click.pass_context
def cli_cluster(ctx, m):
    sig = Signature(sig_path='/home/mabuelanin/2023-QC-manuscript/sigs/SAMN08874620_SRX11760742.sig', k_size=51)
    print(sig.median_abundance())


if __name__ == "__main__":
    cli()
