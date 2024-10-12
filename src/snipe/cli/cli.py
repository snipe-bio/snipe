"""CLI module for Snipe."""

import click
from snipe.api.api import SnipeAPI

@click.group()
def main():
    """Snipe CLI provides commands to interact with the Snipe API."""
    pass

@main.command()
@click.argument('key')
@click.argument('value')
def set(key: str, value: str) -> None:
    """Set a key-value pair in the Snipe API.

    Args:
        key (str): The key to set.
        value (str): The value to associate with the key.
    """
    api = SnipeAPI()
    api.set_data(key, value)
    click.echo(f"Set {key} = {value}")

@main.command()
@click.argument('key')
def get(key: str) -> None:
    """Get the value for a key from the Snipe API.

    Args:
        key (str): The key to retrieve the value for.
    """
    api = SnipeAPI()
    value = api.get_data(key)
    if value:
        click.echo(f"{key} = {value}")
    else:
        click.echo(f"{key} not found.")

if __name__ == "__main__":
    main()
