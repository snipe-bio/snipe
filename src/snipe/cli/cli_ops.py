# cli_ops.py

import os
import sys
import csv
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
import click
from tqdm import tqdm
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
            elif param is None and kwargs.get(op_name):
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
                raise Exception(f"Unknown operation '{op}'.")
        except ValueError as ve:
            raise Exception(f"Value error during operation '{op}': {ve}") from ve
        except Exception as e:
            raise Exception(f"Failed to apply operation '{op}': {e}") from e


def load_signatures(sig_paths: List[str], logger: logging.Logger, allow_duplicates: bool = False) -> List[SnipeSig]:
    """
    Load SnipeSig signatures from the provided file paths.
    Sets each signature's name to the file basename if it's unset/empty.
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
            # If sig.name is empty, set it to the file's basename.
            if not sig.name:
                _tmp_sig_name = os.path.basename(path)
                _tmp_sig_name = os.path.splitext(_tmp_sig_name)[0]
                sig._name = _tmp_sig_name
            
            signatures.append(sig)
            loaded_paths.add(path)
            logger.debug(f"Loaded signature: {sig.name}")
        except Exception as e:
            raise Exception(f"Failed to load signature from {path}: {e}") from e
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
                            raise Exception(f"Signature file does not exist: {line}")
                        all_sig_paths.append(line)
        except Exception as e:
            raise Exception(f"Failed to read signatures from file {sigs_from_file}: {e}") from e

    if not all_sig_paths:
        logger.error("No signature files provided. Use positional arguments or --sigs-from-file.")
        click.echo("Error: No signature files provided. Use positional arguments or --sigs-from-file.", err=True)
        raise Exception("No signature files provided. Use positional arguments or --sigs-from-file.")

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures with duplicates allowed
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=True)

    if not signatures:
        logger.error("No signatures loaded. Exiting.")
        click.echo("Error: No signatures loaded. Exiting.", err=True)
        raise Exception("No signatures loaded. Exiting.")

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
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.")

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
        raise Exception(f"Failed to sum signatures: {e}") from e

    # Export the summed signature
    try:
        summed_signature.export(output_file)
        click.echo(f"Summed signature exported to {output_file}")
    except FileExistsError:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.") from None
    except Exception as e:
        raise Exception(f"Failed to export summed signature: {e}") from e


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
                            raise Exception(f"Signature file does not exist: {line}")
                        all_sig_paths.append(line)
        except Exception as e:
            raise Exception(f"Failed to read signatures from file {sigs_from_file}: {e}") from e

    if not all_sig_paths:
        raise Exception("No signature files provided. Use positional arguments or --sigs-from-file.")

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if not signatures:
        raise Exception("No signatures loaded. Exiting.")

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                                       min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.")

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
        raise Exception(f"Failed to compute intersection of signatures: {e}") from e

    # Export the common signature
    try:
        common_signature.export(output_file)
        click.echo(f"Common signature exported to {output_file}")
    except FileExistsError:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.") from None
    except Exception as e:
        raise Exception(f"Failed to export common signature: {e}") from e


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
                            raise Exception(f"Signature file does not exist: {line}")
                        all_sig_paths.append(line)
        except Exception as e:
            raise Exception(f"Failed to read signatures from file {sigs_from_file}: {e}") from e

    if len(all_sig_paths) != 2:
        raise Exception("Subtract command requires exactly two signature files: <signature1> <signature2>")

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if len(signatures) != 2:
        raise Exception("Failed to load exactly two signatures for subtraction.")

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.")

    # Subtract the second signature from the first
    try:
        primary_sig, secondary_sig = signatures
        primary_sig -= secondary_sig
        logger.debug(f"Subtracted signature '{secondary_sig.name}' from '{primary_sig.name}'.")
    except Exception as e:
        raise Exception(f"Failed to subtract signatures: {e}") from e

    # Update the name if provided
    if name:
        primary_sig._name = name

    # Export the subtracted signature
    try:
        primary_sig.export(output_file)
        click.echo(f"Subtracted signature exported to {output_file}")
        logger.info(f"Subtracted signature exported to {output_file}")
    except FileExistsError:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.") from None
    except Exception as e:
        raise Exception(f"Failed to export subtracted signature: {e}") from e


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
                            raise Exception(f"Signature file does not exist: {line}")
                        all_sig_paths.append(line)
        except Exception as e:
            raise Exception(f"Failed to read signatures from file {sigs_from_file}: {e}") from e

    if not all_sig_paths:
        raise Exception("No signature files provided. Use positional arguments or --sigs-from-file.")

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if not signatures:
        raise Exception("No signatures loaded. Exiting.")

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                    min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.")

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
        raise Exception(f"Failed to compute union of signatures: {e}") from e

    # Export the union signature
    try:
        union_sig.export(output_file)
        click.echo(f"Union signature exported to {output_file}")
        logger.info(f"Union signature exported to {output_file}")
    except FileExistsError:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.") from None
    except Exception as e:
        raise Exception(f"Failed to export union signature: {e}") from e


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
                            raise Exception(f"Signature file does not exist: {line}")
                        all_sig_paths.append(line)
        except Exception as e:
            raise Exception(f"Failed to read signatures from file {sigs_from_file}: {e}") from e

    if not all_sig_paths:
        raise Exception("No signature files provided. Use positional arguments or --sigs-from-file.")

    logger.debug(f"Total signature files to process: {len(all_sig_paths)}")

    # Load signatures without allowing duplicates
    signatures = load_signatures(all_sig_paths, logger, allow_duplicates=False)

    if not signatures:
        raise Exception("No signatures loaded. Exiting.")

    # Parse operation order
    operations = parse_operation_order(ctx, reset_abundance=reset_abundance, trim_singletons=trim_singletons,
                    min_abund=min_abund, max_abund=max_abund, trim_below_median=trim_below_median)

    logger.debug(f"Operations to apply in order: {operations}")

    # Apply operations
    apply_operations(signatures, operations, logger)

    # Check if output file exists
    if os.path.exists(output_file) and not force:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.")

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
        raise Exception(f"Failed to extract common hashes: {e}") from e

    # Export the common hashes signature
    try:
        common_signature.export(output_file)
        click.echo(f"Common hashes signature exported to {output_file}")
        logger.info(f"Common hashes signature exported to {output_file}")
    except FileExistsError:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --force to overwrite.") from None
    except Exception as e:
        raise Exception(f"Failed to export common hashes signature: {e}") from e

def process_experiment(args) -> Dict[str, Any]:
    """
    Process a single experiment:
      1. Load signatures
      2. Skip empty signatures
      3. Detect duplicates by md5sum, keep first
      4. Apply user-requested operations in order
      5. Sum/merge all unique signatures
      6. Export .zip
      7. Return structured result dict
    """
    exp_name, sig_paths, operations, output_dir, force, debug = args
    logger = logging.getLogger(f'process_experiment.{exp_name}')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    result = {
        'exp_name': exp_name,
        'merged_signatures': [],
        'skipped_signatures': [],
        'skipped_due_to_empty': [],
        'output_file': None,
        'status': 'success',
        'error': None
    }

    # 1. Load signatures
    try:
        signatures = load_signatures(sig_paths, logger, allow_duplicates=False)
        if not signatures:
            raise Exception(f"No valid signatures loaded for experiment '{exp_name}'.")
    except Exception as e:
        result['status'] = 'failure'
        result['error'] = str(e)
        return result

    # 2 & 3. Skip empty + deduplicate by md5
    md5_to_signatures = defaultdict(list)
    for sig in signatures:
        if len(sig.hashes):
            md5_to_signatures[sig.md5sum].append(sig)
        else:
            result['skipped_due_to_empty'].append(os.path.basename(sig.name))
            logger.debug(f"Skipping empty signature: {sig.name}")

    unique_signatures = []
    duplicates = []
    for md5, sig_list in md5_to_signatures.items():
        if len(sig_list) > 1:
            # Keep first, skip the rest
            unique_signatures.append(sig_list[0])
            dupes = sig_list[1:]
            duplicates.extend([os.path.basename(s.name) for s in dupes])
            logger.debug(f"Duplicate signatures for md5={md5}: {[s.name for s in dupes]}")
        else:
            unique_signatures.append(sig_list[0])

    result['merged_signatures'] = [os.path.basename(s.name) for s in unique_signatures]
    result['skipped_signatures'] = duplicates

    signatures = unique_signatures

    # If after dedup & skipping empty there are no signatures left => fail
    if not signatures:
        result['status'] = 'failure'
        result['error'] = f"No non-empty signatures left for experiment '{exp_name}' after deduplication."
        return result

    # 4. Apply operations
    try:
        apply_operations(signatures, operations, logger)
    except Exception as e:
        result['status'] = 'failure'
        result['error'] = str(e)
        return result

    # 5. Sum/merge
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

    # 6. Export
    output_file_path = os.path.join(output_dir, f"{exp_name}.zip")
    result['output_file'] = output_file_path

    if os.path.exists(output_file_path) and not force:
        result['status'] = 'failure'
        result['error'] = f"Output file '{output_file_path}' already exists. Use --force to overwrite."
        return result

    try:
        merged_signature.export(output_file_path)
    except Exception as e:
        result['status'] = 'failure'
        result['error'] = f"Failed to export merged signature for experiment '{exp_name}': {e}"
        return result

    return result

@ops.command()
@click.option('--table', '-t', type=click.Path(exists=True, dir_okay=False), required=True,
              help='Tabular file (CSV or TSV) with two columns: <signature_path> <experiment_name>')
@click.option('--output-dir', '-d', type=click.Path(file_okay=False), required=True,
              help='Directory to save merged signature files.')
@click.option('--reset-abundance', is_flag=True, default=False,
              help='Reset abundance for all input signatures to 1.')
@click.option('--trim-singletons', is_flag=True, default=False,
              help='Trim singletons from all input signatures.')
@click.option('--min-abund', type=int,
              help='Keep hashes with abundance >= this value.')
@click.option('--max-abund', type=int,
              help='Keep hashes with abundance <= this value.')
@click.option('--trim-below-median', is_flag=True, default=False,
              help='Trim hashes below the median abundance.')
@click.option('--debug', is_flag=True, default=False,
              help='Enable debugging and detailed logging.')
@click.option('--force', is_flag=True, default=False,
              help='Overwrite existing files in the output directory.')
@click.option('--cores', type=int, default=1,
              help='Number of cores to use for processing experiments in parallel.')
@click.pass_context
def guided_merge(ctx, table, output_dir, reset_abundance, trim_singletons,
                min_abund, max_abund, trim_below_median, debug, force, cores):
    """
    Guide signature merging by groups.

    This command reads a table file (CSV or TSV) where each line contains a signature file path
    and an experiment name. It groups signatures by experiment, applies specified operations
    in the *exact order* the user specifies on the command line, sums the signatures within 
    each group, and saves the merged signatures as `{output_dir}/{experiment_name}.zip`.
    """
    logger = logging.getLogger('ops.guided_merge')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)

    # 1. Detect table delimiter
    def detect_delimiter(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            sample = csvfile.read(1024)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters='\t,')
                delimiter_ = dialect.delimiter
                logger.debug(f"Detected delimiter: '{delimiter_}'")
                return delimiter_
            except csv.Error:
                logger.warning("Could not detect delimiter. Defaulting to tab.")
                return '\t'

    delimiter = detect_delimiter(table)

    # 2. Parse the table & gather signatures
    experiment_mapping = defaultdict(list)
    total_mapped = 0
    total_invalid = 0
    total_valid = 0
    invalid_files = []

    try:
        with open(table, 'r', newline='') as tbl_file:
            reader = csv.reader(tbl_file, delimiter=delimiter)
            for line_num, row in enumerate(reader, start=1):
                if not row:
                    logger.error(f"Line {line_num}: Empty line. Skipping.")
                    continue
                if len(row) < 2:
                    logger.error(f"Line {line_num}: Invalid format. Expected 2 columns, got {len(row)}. Skipping.")
                    total_invalid += 1
                    continue
                sig_path, exp_name = row[0].strip(), row[1].strip()
                if not sig_path or not exp_name:
                    logger.error(f"Line {line_num}: Missing signature path or experiment name. Skipping.")
                    total_invalid += 1
                    continue

                total_mapped += 1
                if not os.path.isfile(sig_path):
                    logger.error(f"Line {line_num}: Signature file does not exist: {sig_path}. Skipping.")
                    invalid_files.append(sig_path)
                    total_invalid += 1
                    continue

                experiment_mapping[exp_name].append(sig_path)
                total_valid += 1
    except Exception as e:
        raise Exception(f"Failed to read table file {table}: {e}") from e

    if total_valid == 0:
        raise Exception("No valid signature files found in the table. Exiting.")

    # 3. Ensure output directory
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            raise Exception(f"Failed to create output directory {output_dir}: {e}") from e
    else:
        if not os.path.isdir(output_dir):
            raise Exception(f"Output path {output_dir} exists and is not a directory.")

    # 4. Determine user-specified operation order
    #    (relying on parse_operation_order and the actual CLI arguments).
    operations = parse_operation_order(
        ctx,
        reset_abundance=reset_abundance,
        trim_singletons=trim_singletons,
        min_abund=min_abund,
        max_abund=max_abund,
        trim_below_median=trim_below_median
    )
    logger.debug(f"User-specified operation order: {operations}")

    # 5. Prepare arguments for each experiment
    experiments_args = []
    for exp_name, sig_paths in experiment_mapping.items():
        experiments_args.append((exp_name, sig_paths, operations, output_dir, force, debug))

    # 6. Process each experiment (in parallel or sequentially)
    results = []
    if cores > 1:
        with ProcessPoolExecutor(max_workers=cores) as executor:
            future_to_exp = {
                executor.submit(process_experiment, args): args[0]
                for args in experiments_args
            }
            for future in tqdm(as_completed(future_to_exp),
                               total=len(future_to_exp),
                               desc="Processing experiments",
                               file=sys.stderr):
                exp_name = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.exception(f"Processing of experiment '{exp_name}' failed.")
                    results.append({
                        'exp_name': exp_name,
                        'merged_signatures': [os.path.basename(p) for p in experiment_mapping[exp_name]],
                        'skipped_signatures': [],
                        'skipped_due_to_empty': [],
                        'output_file': None,
                        'status': 'failure',
                        'error': str(e)
                    })
    else:
        # Sequential processing
        for args in tqdm(experiments_args, desc="Processing experiments"):
            result = process_experiment(args)
            results.append(result)

    # 7. Build final JSON output:
    #    We categorize each experiment into one of:
    #      - failed_exp (status == 'failure')
    #      - multi_run_exps (status == 'success' and original # signatures > 1)
    #      - single_run_exps (status == 'success' and original # signatures == 1)
    final_report = {
        "failed_exp": [],
        "multi_run_exps": [],
        "single_run_exps": []
    }

    for r in results:
        # Gather complete info in a dict
        exp_info = {
            "experiment": r["exp_name"],
            "merged_signatures": r.get("merged_signatures", []),
            "skipped_signatures": r.get("skipped_signatures", []),
            "skipped_due_to_empty": r.get("skipped_due_to_empty", []),
            "output_file": r.get("output_file"),
            "status": r.get("status"),
            "error": r.get("error")
        }

        if exp_info["status"] == "failure":
            final_report["failed_exp"].append(exp_info)
        else:
            # Successful experiment => decide if multi-run or single-run
            original_count = len(experiment_mapping[r["exp_name"]])
            if original_count > 1:
                final_report["multi_run_exps"].append(exp_info)
            else:
                final_report["single_run_exps"].append(exp_info)

    # 8. Write the final JSON report
    json_report_file = os.path.join(output_dir, "merge_report.json")
    try:
        with open(json_report_file, "w") as fp:
            json.dump(final_report, fp, indent=2)
    except Exception as e:
        raise Exception(f"Failed to write JSON report '{json_report_file}': {e}") from e

    # 9. Optionally print a simple summary to the console
    total_exps = len(results)
    failed_count = len(final_report["failed_exp"])
    multi_run_count = len(final_report["multi_run_exps"])
    single_run_count = len(final_report["single_run_exps"])

    click.echo("\nGuided Merge Summary (JSON-based):")
    click.echo(f"\t- Total experiments processed: {total_exps}")
    click.echo(f"\t- Successful experiments: {multi_run_count + single_run_count}")
    click.echo(f"\t- Failed experiments: {failed_count}")
    click.echo(f"\t- Detailed JSON report saved to: {json_report_file}")
