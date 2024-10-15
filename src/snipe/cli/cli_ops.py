# cli_ops.py

import os
import sys
import logging
from typing import List

import click

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
        if not allow_duplicates and path in loaded_paths:
            logger.warning(f"Duplicate signature file detected and skipped: {path}")
            click.echo(f"Warning: Duplicate signature file detected and skipped: {path}", err=True)
            continue
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
    Compute the intersection of multiple signatures, retaining only common hashes.

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
    Merge multiple signatures by taking the union of their hashes.

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
