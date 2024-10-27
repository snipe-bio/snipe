# cli_ops.py

import os
import sys
import logging
from typing import List
from collections import defaultdict
import click
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from snipe.api.enums import SigType
from snipe.api.snipe_sig import SnipeSig

# Configure the root logger to CRITICAL to suppress unwanted logs by default
logging.basicConfig(level=logging.CRITICAL)


def validate_sig_output(ctx, param, value: str) -> str:
    """
    Validate that the output file has a supported extension.

    Supported extensions: .zip, .sig

    Args:
        ctx: Click context.
        param: Click parameter.
        value (str): The value passed to the parameter.

    Returns:
        str: The validated output file path.

    Raises:
        click.BadParameter: If the file extension is not supported.
    """
    supported_extensions = ['.zip', '.sig']
    if not value.endswith(tuple(supported_extensions)):
        raise click.BadParameter(f"Output file must have one of the following extensions: {supported_extensions}")
    return value


def validate_sig_path(ctx, param, value: str) -> str:
    """
    Validate that the provided signature file path exists.

    Args:
        ctx: Click context.
        param: Click parameter.
        value (str): The value passed to the parameter.

    Returns:
        str: The validated signature file path.

    Raises:
        click.BadParameter: If the file does not exist.
    """
    if not os.path.isfile(value):
        raise click.BadParameter(f"Signature file does not exist: {value}")
    return value


def parse_operation_order(ctx, **kwargs):
    """
    Parse the order of operations based on the command line arguments.

    Args:
        ctx: Click context.
        kwargs: Command options.

    Returns:
        List[tuple]: A list of tuples containing operation names and their corresponding values.
    """
    operations = []
    argv = sys.argv[1:]  # Exclude the script name

    # Define a mapping of option names to operation identifiers
    option_order = {
        '--reset-abundance': ('reset_abundance', None),
        '--trim-singletons': ('trim_singletons', None),
        '--min-abund': ('keep_min_abundance', 'min_abund'),
        '--max-abund': ('keep_max_abundance', 'max_abund'),
        '--trim-below-median': ('trim_below_median', None),
    }

    # Iterate through argv to capture the order of operations
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg in option_order:
            op_name, param = option_order[arg]
            if param and kwargs.get(param) is not None:
                operations.append((op_name, kwargs[param]))
            elif param is None and kwargs.get(op_name.replace('_', '-')):
                # For flag-type operations
                operations.append((op_name, None))
            skip_next = param is not None
        else:
            # Handle options with values, e.g., --min-abund=5
            for opt, (op_name, param) in option_order.items():
                if arg.startswith(opt + '='):
                    value = arg.split('=', 1)[1]
                    if param:
                        operations.append((op_name, value))
    return operations


def apply_operations(signatures: List[SnipeSig], operations: List[tuple], logger: logging.Logger):
    """
    Apply the list of operations to the signatures in the specified order.

    Args:
        signatures (List[SnipeSig]): List of SnipeSig instances.
        operations (List[tuple]): List of operations to apply.
        logger (logging.Logger): Logger for logging messages.
    """
    for op, value in operations:
        logger.debug(f"Applying operation: {op} with value: {value}")
        try:
            if op == 'reset_abundance':
                for sig in signatures:
                    sig.reset_abundance(new_abundance=1)
                    logger.debug(f"Reset abundance for signature: {sig.name}")
            elif op == 'trim_singletons':
                for sig in signatures:
                    sig.trim_singletons()
                    logger.debug(f"Trimmed singletons for signature: {sig.name}")
            elif op == 'keep_min_abundance':
                min_abund = int(value)
                for sig in signatures:
                    sig.keep_min_abundance(min_abund)
                    logger.debug(f"Kept hashes with abundance >= {min_abund} for signature: {sig.name}")
            elif op == 'keep_max_abundance':
                max_abund = int(value)
                for sig in signatures:
                    sig.keep_max_abundance(max_abund)
                    logger.debug(f"Kept hashes with abundance <= {max_abund} for signature: {sig.name}")
            elif op == 'trim_below_median':
                for sig in signatures:
                    sig.trim_below_median()
                    logger.debug(f"Trimmed hashes below median abundance for signature: {sig.name}")
            else:
                logger.error(f"Unknown operation: {op}")
                click.echo(f"Error: Unknown operation '{op}'.", err=True)
                sys.exit(1)
        except ValueError as ve:
            logger.error(f"Value error during operation '{op}': {ve}")
            click.echo(f"Error: {ve}", err=True)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error during operation '{op}': {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


def load_signatures(sig_paths: List[str], logger: logging.Logger, allow_duplicates: bool = False) -> List[SnipeSig]:
    """
    Load SnipeSig signatures from the provided file paths.

    Args:
        sig_paths (List[str]): List of file paths to load signatures from.
        logger (logging.Logger): Logger for logging messages.
        allow_duplicates (bool): Flag to allow loading duplicate signature files.

    Returns:
        List[SnipeSig]: List of loaded SnipeSig instances.

    Raises:
        SystemExit: If loading a signature fails.
    """
    signatures = []
    loaded_paths = set()
    for path in sig_paths:
        try:
            logger.debug(f"Loading signature from: {path}")
            sig = SnipeSig(
                sourmash_sig=path,
                sig_type=SigType.SAMPLE,
                enable_logging=logger.isEnabledFor(logging.DEBUG)
            )
            signatures.append(sig)
            loaded_paths.add(path)
            logger.debug(f"Loaded signature: {sig.name}")
        except Exception as e:
            logger.error(f"Failed to load signature from {path}: {e}")
            click.echo(f"Error: Failed to load signature from {path}: {e}", err=True)
            sys.exit(1)
    return signatures


def common_options(func):
    """
    Decorator to add common options to all ops subcommands.

    Args:
        func: The Click command function to decorate.

    Returns:
        The decorated function with added options.
    """
    func = click.argument(
        'sig_files',
        type=click.Path(exists=True),
        nargs=-1,
        callback=lambda ctx, param, value: [validate_sig_path(ctx, param, p) for p in value]
    )(func)
    func = click.option(
        '--sigs-from-file',
        type=click.Path(exists=True),
        help='File containing signature paths (one per line).'
    )(func)
    func = click.option(
        '--reset-abundance',
        is_flag=True,
        default=False,
        help='Reset abundance for all input signatures to 1.'
    )(func)
    func = click.option(
        '--trim-singletons',
        is_flag=True,
        default=False,
        help='Trim singletons from all input signatures.'
    )(func)
    func = click.option(
        '--min-abund',
        type=int,
        help='Keep hashes with abundance >= this value.'
    )(func)
    func = click.option(
        '--max-abund',
        type=int,
        help='Keep hashes with abundance <= this value.'
    )(func)
    func = click.option(
        '--trim-below-median',
        is_flag=True,
        default=False,
        help='Trim hashes below the median abundance.'
    )(func)
    func = click.option(
        '-o', '--output-file',
        required=True,
        callback=validate_sig_output,
        help='Output file with .zip or .sig extension.'
    )(func)
    func = click.option(
        '--name', '-n',
        type=str,
        default=None,
        required=True,
        help='Name for the output signature.'
    )(func)
    return func


@click.group()
def ops():
    """
    Perform operations on SnipeSig signatures.

    Subcommands:
        1. `sum`        Merge multiple signatures by summing their abundances.
        2. `intersect`  Compute the intersection of multiple signatures.
        3. `union`      Compute the union of multiple signatures.
        4. `subtract`   Subtract one signature from another.
        5. `common`     Extract hashes common to all input signatures.

    Use 'snipe ops <subcommand> --help' for more information on a command.
    """
    pass


@ops.command()
@common_options
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable debugging and detailed logging.'
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Overwrite the output file if it already exists.'
)
@click.pass_context
def sum(ctx, sig_files, sigs_from_file, reset_abundance, trim_singletons,
        min_abund, max_abund, trim_below_median, output_file, name, debug, force):
    """
    Merge multiple signatures by summing their abundances.

    This command loads multiple signature files, applies specified operations
    (like resetting abundances), and then sums them to create a new signature
    where the abundance of each hash is the sum of its abundances across all
    input signatures.

    **Example:**

        snipe ops sum sample1.sig.zip sample2.sig.zip -o summed.sig.zip --reset-abundance

    This command will:
      1. Load `sample1.sig.zip` and `sample2.sig.zip`.
      2. Reset the abundance of each hash in both signatures to 1.
      3. Sum the signatures, resulting in `summed.sig.zip` where each hash has an abundance of 2.
    """
    # Setup logging
    logger = logging.getLogger('ops.sum')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # Collect all signature paths
    all_sig_paths = list(sig_files)
    if sigs_from_file:
        try:
            with open(sigs_from_file, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        if not os.path.isfile(line):
                            logger.error(f"Signature file does not exist: {line}")
                            click.echo(f"Error: Signature file does not exist: {line}", err=True)
                            sys.exit(1)
                        all_sig_paths.append(line)
        except Exception as e:
            logger.error(f"Failed to read signatures from file {sigs_from_file}: {e}")
            click.echo(f"Error: Failed to read signatures from file {sigs_from_file}: {e}", err=True)
            sys.exit(1)

    if not all_sig_paths:
        logger.error("No signature files provided. Use positional arguments or --sigs-from-file.")
        click.echo("Error: No signature files provided. Use positional arguments or --sigs-from-file.", err=True)
        sys.exit(1)

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures with duplicates allowed
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=True)

    if not signatures:
        logger.error("No signatures loaded. Exiting.")
        click.echo("Error: No signatures loaded. Exiting.", err=True)
        sys.exit(1)

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                                       min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    # Sum signatures
    try:
        summed_signature = SnipeSig.sum_signatures(
            signatures,
            name=name or "summed_signature",
            filename=None,
            enable_logging=debug
        )
        logger.debug(f"Summed signature created with name: {summed_signature.name}")
    except Exception as e:
        logger.error(f"Failed to sum signatures: {e}")
        click.echo(f"Error: Failed to sum signatures: {e}", err=True)
        sys.exit(1)

    # Export the summed signature
    try:
        summed_signature.export(output_file)
        click.echo(f"Summed signature exported to {output_file}")
        logger.info(f"Summed signature exported to {output_file}")
    except FileExistsError:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to export summed signature: {e}")
        click.echo(f"Error: Failed to export summed signature: {e}", err=True)
        sys.exit(1)


@ops.command()
@common_options
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable debugging and detailed logging.'
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Overwrite the output file if it already exists.'
)
@click.pass_context
def intersect(ctx, sig_files, sigs_from_file, reset_abundance, trim_singletons,
              min_abund, max_abund, trim_below_median, output_file, name, debug, force):
    """
    Intersect multiple sigs and retain abundance of first one.

    This command identifies hashes that are present in **all** input signatures and
    retains their abundance from the first signature.

    **Example:**

        snipe ops intersect sample1.sig.zip sample2.sig.zip -o intersection.sig.zip

    This command will:
      1. Load `sample1.sig.zip` and `sample2.sig.zip`.
      2. Apply any specified operations.
      3. Retain only hashes common to both signatures.
      4. Use the abundance values from `sample1.sig.zip` for the common hashes.
      5. Export the resulting intersection to `intersection.sig.zip`.
    """
    # Setup logging
    logger = logging.getLogger('ops.intersect')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # Collect all signature paths
    all_sig_paths = list(sig_files)
    if sigs_from_file:
        try:
            with open(sigs_from_file, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        if not os.path.isfile(line):
                            logger.error(f"Signature file does not exist: {line}")
                            click.echo(f"Error: Signature file does not exist: {line}", err=True)
                            sys.exit(1)
                        all_sig_paths.append(line)
        except Exception as e:
            logger.error(f"Failed to read signatures from file {sigs_from_file}: {e}")
            click.echo(f"Error: Failed to read signatures from file {sigs_from_file}: {e}", err=True)
            sys.exit(1)

    if not all_sig_paths:
        logger.error("No signature files provided. Use positional arguments or --sigs-from-file.")
        click.echo("Error: No signature files provided. Use positional arguments or --sigs-from-file.", err=True)
        sys.exit(1)

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if not signatures:
        logger.error("No signatures loaded. Exiting.")
        click.echo("Error: No signatures loaded. Exiting.", err=True)
        sys.exit(1)

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                                       min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    # Compute intersection
    try:
        common_signature = SnipeSig.common_hashes(
            signatures,
            name=name or "common_hashes_signature",
            filename=None,
            enable_logging=debug
        )
        logger.debug(f"Common signature created with name: {common_signature.name}")
    except Exception as e:
        logger.error(f"Failed to compute intersection of signatures: {e}")
        click.echo(f"Error: Failed to compute intersection of signatures: {e}", err=True)
        sys.exit(1)

    # Export the common signature
    try:
        common_signature.export(output_file)
        click.echo(f"Common signature exported to {output_file}")
        logger.info(f"Common signature exported to {output_file}")
    except FileExistsError:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to export common signature: {e}")
        click.echo(f"Error: Failed to export common signature: {e}", err=True)
        sys.exit(1)


@ops.command()
@common_options
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable debugging and detailed logging.'
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Overwrite the output file if it already exists.'
)
@click.pass_context
def subtract(ctx, sig_files, sigs_from_file, reset_abundance, trim_singletons,
             min_abund, max_abund, trim_below_median, output_file, name, debug, force):
    """
    Subtract one signature from another.

    This command removes the hashes present in the second signature from the first
    signature. The resulting signature will contain only the hashes that are unique
    to the first signature.

    **Example:**

        snipe ops subtract sample1.sig.zip sample2.sig.zip -o subtracted.sig.zip

    This command will:
      1. Load `sample1.sig.zip` and `sample2.sig.zip`.
      2. Apply any specified operations.
      3. Subtract the hashes of `sample2.sig.zip` from `sample1.sig.zip`.
      4. Export the resulting signature to `subtracted.sig.zip`.
    """
    # Setup logging
    logger = logging.getLogger('ops.subtract')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # Collect all signature paths
    all_sig_paths = list(sig_files)
    if sigs_from_file:
        try:
            with open(sigs_from_file, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        if not os.path.isfile(line):
                            logger.error(f"Signature file does not exist: {line}")
                            click.echo(f"Error: Signature file does not exist: {line}", err=True)
                            sys.exit(1)
                        all_sig_paths.append(line)
        except Exception as e:
            logger.error(f"Failed to read signatures from file {sigs_from_file}: {e}")
            click.echo(f"Error: Failed to read signatures from file {sigs_from_file}: {e}", err=True)
            sys.exit(1)

    if len(all_sig_paths) != 2:
        logger.error("Subtract command requires exactly two signature files: <signature1> <signature2>")
        click.echo("Error: Subtract command requires exactly two signature files: <signature1> <signature2>", err=True)
        sys.exit(1)

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if len(signatures) != 2:
        logger.error("Failed to load exactly two signatures for subtraction.")
        click.echo("Error: Failed to load exactly two signatures for subtraction.", err=True)
        sys.exit(1)

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                                       min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    # Subtract the second signature from the first
    try:
        primary_sig, secondary_sig = signatures
        primary_sig.subtract(secondary_sig)
        logger.debug(f"Subtracted signature '{secondary_sig.name}' from '{primary_sig.name}'.")
    except Exception as e:
        logger.error(f"Failed to subtract signatures: {e}")
        click.echo(f"Error: Failed to subtract signatures: {e}", err=True)
        sys.exit(1)

    # Update the name if provided
    if name:
        primary_sig._name = name

    # Export the subtracted signature
    try:
        primary_sig.export(output_file)
        click.echo(f"Subtracted signature exported to {output_file}")
        logger.info(f"Subtracted signature exported to {output_file}")
    except FileExistsError:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to export subtracted signature: {e}")
        click.echo(f"Error: Failed to export subtracted signature: {e}", err=True)
        sys.exit(1)


@ops.command()
@common_options
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable debugging and detailed logging.'
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Overwrite the output file if it already exists.'
)
@click.pass_context
def union(ctx, sig_files, sigs_from_file, reset_abundance, trim_singletons,
          min_abund, max_abund, trim_below_median, output_file, name, debug, force):
    """
    Merge multiple signatures by taking the union of hashes.

    This command combines multiple signatures, retaining all unique hashes from each.
    If a hash appears in multiple signatures, its abundance in the resulting signature
    is the sum of its abundances across all input signatures.

    **Example:**

        snipe ops union sample1.sig.zip sample2.sig.zip -o union.sig.zip

    This command will:
      1. Load `sample1.sig.zip` and `sample2.sig.zip`.
      2. Apply any specified operations.
      3. Combine the signatures, summing abundances for overlapping hashes.
      4. Export the resulting union signature to `union.sig.zip`.
    """
    # Setup logging
    logger = logging.getLogger('ops.union')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # Collect all signature paths
    all_sig_paths = list(sig_files)
    if sigs_from_file:
        try:
            with open(sigs_from_file, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        if not os.path.isfile(line):
                            logger.error(f"Signature file does not exist: {line}")
                            click.echo(f"Error: Signature file does not exist: {line}", err=True)
                            sys.exit(1)
                        all_sig_paths.append(line)
        except Exception as e:
            logger.error(f"Failed to read signatures from file {sigs_from_file}: {e}")
            click.echo(f"Error: Failed to read signatures from file {sigs_from_file}: {e}", err=True)
            sys.exit(1)

    if not all_sig_paths:
        logger.error("No signature files provided. Use positional arguments or --sigs-from-file.")
        click.echo("Error: No signature files provided. Use positional arguments or --sigs-from-file.", err=True)
        sys.exit(1)

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if not signatures:
        logger.error("No signatures loaded. Exiting.")
        click.echo("Error: No signatures loaded. Exiting.", err=True)
        sys.exit(1)

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                                       min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    # Compute union
    try:
        # Initialize with the first signature
        union_sig = signatures[0]
        for sig in signatures[1:]:
            union_sig = union_sig + sig  # Using __add__ method
            logger.debug(f"Unioned with signature: {sig.name}")
        union_sig._name = name or "union_signature"  # Update the name if provided
        logger.debug(f"Union signature created with name: {union_sig.name}")
    except Exception as e:
        logger.error(f"Failed to compute union of signatures: {e}")
        click.echo(f"Error: Failed to compute union of signatures: {e}", err=True)
        sys.exit(1)

    # Export the union signature
    try:
        union_sig.export(output_file)
        click.echo(f"Union signature exported to {output_file}")
        logger.info(f"Union signature exported to {output_file}")
    except FileExistsError:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to export union signature: {e}")
        click.echo(f"Error: Failed to export union signature: {e}", err=True)
        sys.exit(1)


@ops.command()
@common_options
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable debugging and detailed logging.'
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Overwrite the output file if it already exists.'
)
@click.pass_context
def common(ctx, sig_files, sigs_from_file, reset_abundance, trim_singletons,
           min_abund, max_abund, trim_below_median, output_file, name, debug, force):
    """
    Extract hashes that are common to all input signatures.

    This command identifies hashes that are present in **every** input signature and
    creates a new signature containing only these common hashes. The abundance
    values from the first signature are retained.

    **Example:**

        snipe ops common sample1.sig.zip sample2.sig.zip sample3.sig.zip -o common_hashes.sig.zip

    This command will:
      1. Load `sample1.sig.zip`, `sample2.sig.zip`, and `sample3.sig.zip`.
      2. Apply any specified operations.
      3. Identify hashes common to all signatures.
      4. Retain abundance values from `sample1.sig.zip` for these common hashes.
      5. Export the resulting common hashes signature to `common_hashes.sig.zip`.
    """
    # Setup logging
    logger = logging.getLogger('ops.common')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # Collect all signature paths
    all_sig_paths = list(sig_files)
    if sigs_from_file:
        try:
            with open(sigs_from_file, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        if not os.path.isfile(line):
                            logger.error(f"Signature file does not exist: {line}")
                            click.echo(f"Error: Signature file does not exist: {line}", err=True)
                            sys.exit(1)
                        all_sig_paths.append(line)
        except Exception as e:
            logger.error(f"Failed to read signatures from file {sigs_from_file}: {e}")
            click.echo(f"Error: Failed to read signatures from file {sigs_from_file}: {e}", err=True)
            sys.exit(1)

    if not all_sig_paths:
        logger.error("No signature files provided. Use positional arguments or --sigs-from-file.")
        click.echo("Error: No signature files provided. Use positional arguments or --sigs-from-file.", err=True)
        sys.exit(1)

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if not signatures:
        logger.error("No signatures loaded. Exiting.")
        click.echo("Error: No signatures loaded. Exiting.", err=True)
        sys.exit(1)

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                                       min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    # Extract common hashes
    try:
        # Initialize with the first signature
        common_signature = signatures[0]
        for sig in signatures[1:]:
            common_signature = common_signature & sig  # Using __and__ method, retains abundance from the first signature
            logger.debug(f"Commoned with signature: {sig.name}")
        common_signature._name = name or "common_hashes_signature"  # Update the name if provided
        logger.debug(f"Common hashes signature created with name: {common_signature.name}")
    except Exception as e:
        logger.error(f"Failed to extract common hashes: {e}")
        click.echo(f"Error: Failed to extract common hashes: {e}", err=True)
        sys.exit(1)

    # Export the common hashes signature
    try:
        common_signature.export(output_file)
        click.echo(f"Common hashes signature exported to {output_file}")
        logger.info(f"Common hashes signature exported to {output_file}")
    except FileExistsError:
        logger.error(f"Output file '{output_file}' already exists. Use --force to overwrite.")
        click.echo(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to export common hashes signature: {e}")
        click.echo(f"Error: Failed to export common hashes signature: {e}", err=True)
        sys.exit(1)

def process_experiment(args):
    exp_name, sig_paths, operations, output_dir, force, debug = args
    # Create logger for this function
    logger = logging.getLogger(f'process_experiment.{exp_name}')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    result = {
        'exp_name': exp_name,
        'merged_signatures': [],       # To be populated after processing
        'skipped_signatures': [],      # To store skipped duplicates
        'output_file': None,
        'status': 'success',
        'error': None
    }

    # Load signatures
    try:
        signatures = load_signatures(sig_paths, logger, allow_duplicates=False)
        if not signatures:
            raise Exception(f"No valid signatures loaded for experiment '{exp_name}'.")
    except Exception as e:
        result['status'] = 'failure'
        result['error'] = str(e)
        return result

    # Duplicate Detection
    try:
        # Create a mapping from md5sum to list of signatures
        md5_to_signatures = defaultdict(list)
        for sig in signatures:
            md5_to_signatures[sig.md5sum].append(sig)
        
        # Identify duplicates
        unique_signatures = []
        skipped_signatures = []
        for md5, sig_list in md5_to_signatures.items():
            if len(sig_list) > 1:
                # Keep the first signature, skip the rest
                unique_signatures.append(sig_list[0])
                duplicates = sig_list[1:]
                skipped_signatures.extend([os.path.basename(sig.name) for sig in duplicates])
                logger.debug(f"Duplicate signatures detected for md5sum {md5}: {[sig.name for sig in duplicates]}")
            else:
                unique_signatures.append(sig_list[0])
        
        # Update the signatures list to only include unique signatures
        signatures = unique_signatures

        # Update the result with merged and skipped signatures
        result['merged_signatures'] = [os.path.basename(sig.name) for sig in signatures]
        result['skipped_signatures'] = skipped_signatures

    except Exception as e:
        result['status'] = 'failure'
        result['error'] = f"Duplicate detection failed: {e}"
        return result

    # Apply operations
    try:
        apply_operations(signatures, operations, logger)
    except Exception as e:
        result['status'] = 'failure'
        result['error'] = str(e)
        return result

    # Sum signatures
    try:
        merged_signature = SnipeSig.sum_signatures(
            signatures,
            name=exp_name,
            filename=None,
            enable_logging=debug
        )
    except Exception as e:
        result['status'] = 'failure'
        result['error'] = f"Failed to merge signatures for experiment '{exp_name}': {e}"
        return result

    # Define output file path
    output_file_path = os.path.join(output_dir, f"{exp_name}.zip")
    result['output_file'] = output_file_path

    # Check if output file exists
    if os.path.exists(output_file_path) and not force:
        result['status'] = 'failure'
        result['error'] = f"Output file '{output_file_path}' already exists. Use --force to overwrite."
        return result

    # Export the merged signature
    try:
        merged_signature.export(output_file_path)
    except Exception as e:
        result['status'] = 'failure'
        result['error'] = f"Failed to export merged signature for experiment '{exp_name}': {e}"
        return result

    return result

@ops.command()
@click.option('--table', '-t', type=click.Path(exists=True, dir_okay=False), required=True, help='Tabular file (CSV or TSV) with two columns: <signature_path> <experiment_name>')
@click.option('--output-dir', '-d', type=click.Path(file_okay=False), required=True, help='Directory to save merged signature files.')
@click.option('--reset-abundance', is_flag=True, default=False, help='Reset abundance for all input signatures to 1.')
@click.option('--trim-singletons', is_flag=True, default=False, help='Trim singletons from all input signatures.')
@click.option('--min-abund', type=int, help='Keep hashes with abundance >= this value.')
@click.option('--max-abund', type=int, help='Keep hashes with abundance <= this value.')
@click.option('--trim-below-median', is_flag=True, default=False, help='Trim hashes below the median abundance.')
@click.option('--debug', is_flag=True, default=False, help='Enable debugging and detailed logging.')
@click.option('--force', is_flag=True, default=False, help='Overwrite existing files in the output directory.')
@click.option('--cores', type=int, default=1, help='Number of cores to use for processing experiments in parallel.')
@click.pass_context
def guided_merge(ctx, table, output_dir, reset_abundance, trim_singletons,
                min_abund, max_abund, trim_below_median, debug, force, cores):
    """
    Guide signature merging by groups.

    This command reads a table file (CSV or TSV) where each line contains a signature file path and an experiment name.
    It groups signatures by experiment, applies specified operations, sums the signatures within each group,
    and saves the merged signatures as `{output_dir}/{experiment_name}.zip`.

    **Example:**

        snipe ops guided-merge --table mapping.tsv --output-dir merged_sigs --reset-abundance --force --cores 4

    **Example Table File (`mapping.tsv`):**

        /path/to/sig1.zip    exp1
        /path/to/sig2.zip    exp1
        /path/to/sig3.zip    exp2
        /path/to/sig4.zip    exp2
        /path/to/sig5.zip    exp3
    """
    # Setup logging
    logger = logging.getLogger('ops.guided_merge')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # Function to detect delimiter
    def detect_delimiter(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            sample = csvfile.read(1024)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters='\t,')
                delimiter = dialect.delimiter
                logger.debug(f"Detected delimiter: '{delimiter}'")
                return delimiter
            except csv.Error:
                logger.warning("Could not detect delimiter. Defaulting to tab.")
                return '\t'

    # Initialize counters and mapping
    total_mapped = 0
    total_valid = 0
    total_invalid = 0
    experiment_mapping = defaultdict(list)
    invalid_files = []

    # Detect delimiter
    delimiter = detect_delimiter(table)

    # Parse the table file
    try:
        with open(table, 'r', newline='') as tbl_file:
            reader = csv.reader(tbl_file, delimiter=delimiter)
            for line_num, row in enumerate(reader, start=1):
                if not row:
                    logger.debug(f"Line {line_num}: Empty line. Skipping.")
                    continue  # Skip empty lines
                if len(row) < 2:
                    logger.warning(f"Line {line_num}: Invalid format. Expected 2 columns, got {len(row)}. Skipping.")
                    total_invalid += 1
                    continue
                sig_path, exp_name = row[0].strip(), row[1].strip()
                if not sig_path or not exp_name:
                    logger.warning(f"Line {line_num}: Missing signature path or experiment name. Skipping.")
                    total_invalid += 1
                    continue
                total_mapped += 1
                if not os.path.isfile(sig_path):
                    logger.warning(f"Line {line_num}: Signature file does not exist: {sig_path}. Skipping.")
                    invalid_files.append(sig_path)
                    total_invalid += 1
                    continue
                experiment_mapping[exp_name].append(sig_path)
                total_valid += 1
    except Exception as e:
        logger.error(f"Failed to read table file {table}: {e}")
        click.echo(f"Error: Failed to read table file {table}: {e}", err=True)
        sys.exit(1)

    if total_valid == 0:
        logger.error("No valid signature files found in the table. Exiting.")
        click.echo("Error: No valid signature files found in the table. Exiting.", err=True)
        sys.exit(1)

    logger.debug(f"Total lines in table: {total_mapped + total_invalid}")
    logger.debug(f"Total valid signatures: {total_valid}")
    logger.debug(f"Total invalid signatures: {total_invalid}")
    logger.debug(f"Experiments found: {len(experiment_mapping)}")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            click.echo(f"Error: Failed to create output directory {output_dir}: {e}", err=True)
            sys.exit(1)
    else:
        if not os.path.isdir(output_dir):
            logger.error(f"Output path {output_dir} exists and is not a directory.")
            click.echo(f"Error: Output path {output_dir} exists and is not a directory.", err=True)
            sys.exit(1)

    # Define operation context
    operations = []
    if reset_abundance:
        operations.append(('reset_abundance', None))
    if trim_singletons:
        operations.append(('trim_singletons', None))
    if min_abund is not None:
        operations.append(('keep_min_abundance', min_abund))
    if max_abund is not None:
        operations.append(('keep_max_abundance', max_abund))
    if trim_below_median:
        operations.append(('trim_below_median', None))

    # Prepare arguments for multiprocessing
    experiments_args = []
    for exp_name, sig_paths in experiment_mapping.items():
        experiments_args.append((exp_name, sig_paths, operations, output_dir, force, debug))

    # Process experiments with multiprocessing
    results = []

    if cores > 1:
        with ProcessPoolExecutor(max_workers=cores) as executor:
            future_to_exp = {executor.submit(process_experiment, args): args[0] for args in experiments_args}
            for future in tqdm(as_completed(future_to_exp), total=len(future_to_exp), desc="Processing experiments"):
                exp_name = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.exception(f"Processing of experiment '{exp_name}' failed with an exception.")
                    results.append({
                        'exp_name': exp_name,
                        'merged_signatures': [os.path.basename(p) for p in experiment_mapping[exp_name]],
                        'skipped_signatures': [],
                        'output_file': None,
                        'status': 'failure',
                        'error': str(e)
                    })
    else:
        # Sequential processing
        for args in tqdm(experiments_args, desc="Processing experiments"):
            result = process_experiment(args)
            results.append(result)

    # Write the structured report
    report_file = os.path.join(output_dir, 'merge_report.txt')

    try:
        with open(report_file, 'w') as report:
            report.write("Merge Report\n")
            report.write("="*50 + "\n\n")
            for result in results:
                report.write(f"Experiment ID: {result['exp_name']}\n")
                report.write("-"*50 + "\n")
                report.write("Merged Signatures:\n")
                if result['merged_signatures']:
                    for sig in result['merged_signatures']:
                        report.write(f"    - {sig}\n")
                else:
                    report.write("    None\n")

                report.write("Skipped Signatures (Due to Duplication):\n")
                if result['skipped_signatures']:
                    for sig in result['skipped_signatures']:
                        report.write(f"    - {sig}\n")
                else:
                    report.write("    None\n")

                report.write(f"Output File: {result['output_file'] if result['output_file'] else 'N/A'}\n")
                report.write(f"Status: {result['status'].capitalize()}\n")
                if result['status'] == 'failure':
                    report.write(f"Error: {result['error']}\n")
                report.write("\n" + "-"*50 + "\n\n")
    except Exception as e:
        logger.error(f"Failed to write report file {report_file}: {e}")
        click.echo(f"Error: Failed to write report file {report_file}: {e}", err=True)
        sys.exit(1)

    # Summary Report
    total_experiments = len(results)
    successful_experiments = 0
    for r in results:
        if r['status'] == 'success':
            successful_experiments += 1
    failed_experiments = total_experiments - successful_experiments
    # total_skipped = sum(len(r.get('skipped_signatures', [])) for r in results)
    total_skipped = 0
    for r in results:
        total_skipped += len(r.get('skipped_signatures', []))

    click.echo("\nGuided Merge Summary:")
    click.echo(f"\t- Total experiments processed: {total_experiments}")
    click.echo(f"\t- Successful experiments: {successful_experiments}")
    click.echo(f"\t- Failed experiments: {failed_experiments}")
    click.echo(f"\t- Total signatures skipped due to duplication: {total_skipped}")
    click.echo(f"\t- Detailed report saved to {report_file}")

    click.echo(f"\nReport saved to {report_file}")