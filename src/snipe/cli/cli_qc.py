import os
import sys
import time
import logging
from typing import Optional, Any, List, Dict, Set, Union

import click
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from snipe.api.enums import SigType
from snipe.api.snipe_sig import SnipeSig
from snipe.api.multisig_reference_QC import MultiSigReferenceQC


def validate_sig_input(ctx, param, value: tuple) -> str:
    supported_extensions = ['.zip', '.sig']
    for path in value:
        if not os.path.exists(path):
            raise click.BadParameter(f"File not found: {path}")
        if not any(path.lower().endswith(ext) for ext in supported_extensions):
            raise click.BadParameter(f"Unsupported file format: {path}, supported formats: {', '.join(supported_extensions)}")
    return value


def validate_tsv_file(ctx, param, value: str) -> str:
    if not value.lower().endswith('.tsv'):
        raise click.BadParameter('Output file must have a .tsv extension.')
    return value


@click.command()
@click.option('--ref', type=click.Path(exists=True), required=True, help='Reference genome signature file (required).')
@click.option('--sample', type=click.Path(exists=True), callback=validate_sig_input, multiple=True, default = None, help='Sample signature file. Can be provided multiple times.')
@click.option('--samples-from-file', type=click.Path(exists=True), help='File containing sample paths (one per line).')
@click.option('--amplicon', type=click.Path(exists=True), help='Amplicon signature file (optional).')
@click.option('--roi', is_flag=True, default=False, help='Calculate ROI for 1,2,5,9 folds.')
@click.option('--export-var', is_flag=True, default=False, help='Export signatures for variances')
@click.option('--ychr', type=click.Path(exists=True), help='Y chromosome signature file (overrides the reference ychr).')
@click.option('--debug', is_flag=True, default=False, help='Enable debugging and detailed logging.')
@click.option('-o', '--output', required=True, callback=validate_tsv_file, help='Output TSV file for QC results.')
@click.option('--var', 'vars', multiple=True, type=click.Path(exists=True), help='Extra signature file path to study variance of non-reference k-mers.')
def qc(ref: str, sample: List[str], samples_from_file: Optional[str],
       amplicon: Optional[str], roi: bool, export_var: bool, 
       ychr: Optional[str], debug: bool, output: str, vars: List[str]):
    """
        Perform quality control (QC) on multiple samples against a reference genome.

        This command calculates various QC metrics for each provided sample, optionally including advanced metrics and ROI (Return on investement) predictions. Results are aggregated and exported to a TSV file.

        ## Usage

        ```bash
        snipe qc [OPTIONS]
        ```

        ## Options

        - `--ref PATH` **[required]**  
        Reference genome signature file.

        - `--sample PATH`  
        Sample signature file. Can be provided multiple times.

        - `--samples-from-file PATH`  
        File containing sample paths (one per line).

        - `--amplicon PATH`  
        Amplicon signature file (optional).

        - `--roi`  
        Calculate ROI for 1x, 2x, 5x, and 9x coverage folds.

        - `--advanced`  
        Include advanced QC metrics.

        - `--ychr PATH`  
        Y chromosome signature file (overrides the reference ychr).

        - `--debug`  
        Enable debugging and detailed logging.

        - `-o`, `--output PATH` **[required]**  
        Output TSV file for QC results.

        - `--var PATH`  
        Variable signature file path. Can be used multiple times.

        ## Examples

        ### Performing QC on Multiple Samples

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig -o qc_results.tsv
        ```

        ### Performing QC with Samples Listed in a File

        ```bash
        snipe qc --ref reference.sig --samples-from-file samples.txt -o qc_results.tsv
        ```

        *Contents of `samples.txt`:*

        ```
        sample1.sig
        sample2.sig
        sample3.sig
        ```

        ### Performing QC with an Amplicon Signature

        ```bash
        snipe qc --ref reference.sig --amplicon amplicon.sig --sample sample1.sig -o qc_results.tsv
        ```

        ### Including Advanced QC Metrics and ROI Calculations

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --advanced --roi -o qc_results.tsv
        ```

        ### Using Multiple Variable Signatures

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --var var1.sig --var var2.sig -o qc_results.tsv
        ```

        ### Overriding the Y Chromosome Signature

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --ychr custom_y.sig -o qc_results.tsv
        ```

        ### Combining Multiple Options

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig --amplicon amplicon.sig --var var1.sig --var var2.sig --advanced --roi -o qc_results.tsv
        ```

        ## Detailed Use Cases

        ### Use Case 1: Basic QC on Single Sample

        **Objective:** Perform QC on a single sample against a reference genome without any advanced metrics or ROI.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig -o qc_basic.tsv
        ```

        **Explanation:**

        - `--ref reference.sig`: Specifies the reference genome signature file.
        - `--sample sample1.sig`: Specifies the sample signature file.
        - `-o qc_basic.tsv`: Specifies the output TSV file for QC results.

        **Expected Output:**

        A TSV file named `qc_basic.tsv` containing basic QC metrics for `sample1.sig`.

        ### Use Case 2: QC on Multiple Samples with ROI

        **Objective:** Perform QC on multiple samples and calculate Regions of Interest (ROI) for each.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig --roi -o qc_roi.tsv
        ```

        **Explanation:**

        - `--ref reference.sig`: Reference genome signature file.
        - `--sample sample1.sig` & `--sample sample2.sig`: Multiple sample signature files.
        - `--roi`: Enables ROI calculations.
        - `-o qc_roi.tsv`: Output file for QC results.

        **Expected Output:**

        A TSV file named `qc_roi.tsv` containing QC metrics along with ROI predictions for `sample1.sig` and `sample2.sig`.

        ### Use Case 3: Advanced QC with Amplicon and Variable Signatures

        **Objective:** Perform advanced QC on a sample using an amplicon signature and multiple variable signatures.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --amplicon amplicon.sig --sample sample1.sig --var var1.sig --var var2.sig --advanced -o qc_advanced.tsv
        ```

        **Explanation:**

        - `--ref reference.sig`: Reference genome signature file.
        - `--amplicon amplicon.sig`: Amplicon signature file.
        - `--sample sample1.sig`: Sample signature file.
        - `--var var1.sig` & `--var var2.sig`: Variable signature files.
        - `--export-var`: Export signatures for variances.
        - `-o qc_advanced.tsv`: Output file for QC results.

        **Expected Output:**

        A TSV file named `qc_advanced.tsv` containing comprehensive QC metrics, including advanced metrics and analyses based on the amplicon and variable signatures for `sample1.sig`.

        ### Use Case 4: Overwriting Existing Output File

        **Objective:** Perform QC and overwrite an existing output TSV file.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig -o qc_results.tsv
        ```

        **Explanation:**

        - If `qc_results.tsv` already exists, the command will **fail** to prevent accidental overwriting. To overwrite, use the `--force` flag (assuming you've implemented it; if not, you may need to adjust the `qc` command to include a `--force` option).

        **Adjusted Command with `--force` (if implemented):**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig -o qc_results.tsv --force
        ```

        **Expected Output:**

        The existing `qc_results.tsv` file will be overwritten with the new QC results for `sample1.sig`.

        ### Use Case 5: Using a Custom Y Chromosome Signature

        **Objective:** Override the default Y chromosome signature with a custom one during QC.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --ychr custom_y.sig -o qc_custom_y.tsv
        ```

        **Explanation:**

        - `--ychr custom_y.sig`: Specifies a custom Y chromosome signature file to override the default.

        **Expected Output:**

        A TSV file named `qc_custom_y.tsv` containing QC metrics for `sample1.sig` with analyses based on the custom Y chromosome signature.

        ### Use Case 6: Reading Sample Paths from a File

        **Objective:** Perform QC on multiple samples listed in a text file.

        **Command:**

        ```bash
        snipe qc --ref reference.sig --samples-from-file samples.txt -o qc_from_file.tsv
        ```

        **Explanation:**

        - `--samples-from-file samples.txt`: Specifies a file containing sample paths, one per line.

        **Contents of `samples.txt`:**

        ```
        sample1.sig
        sample2.sig
        sample3.sig
        ```

        **Expected Output:**

        A TSV file named `qc_from_file.tsv` containing QC metrics for `sample1.sig`, `sample2.sig`, and `sample3.sig`.

        ### Use Case 7: Combining Multiple Options for Comprehensive QC

        **Command:**

        ```bash
        snipe qc --ref reference.sig --sample sample1.sig --sample sample2.sig --amplicon amplicon.sig --var var1.sig --var var2.sig --advanced --roi -o qc_comprehensive.tsv
        ```

        **Explanation:**

        - `--ref reference.zip`: Reference genome signature file.
        - `--sample sample1.zip` & `--sample sample2.sig`: Multiple sample signature files.
        - `--amplicon amplicon.zip`: Amplicon signature file.
        - `--var var1.zip` & `--var var2.zip`: Variable signature files.
        - `--advanced`: Includes advanced QC metrics.
        - `--roi`: Enables ROI calculations.
        - `-o qc_comprehensive.tsv`: Output file for QC results.

        **Expected Output:**

        A TSV file named `qc_comprehensive.tsv` containing comprehensive QC metrics, including advanced analyses, ROI predictions, and data from amplicon and variable signatures for both `sample1.sig` and `sample2.sig`.
    """
    
    print(sample)
    
    start_time = time.time()

    # Configure logging
    logger = logging.getLogger('snipe_qc')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    logger.info("Starting QC process.")

    # Collect sample paths from --sample and --samples-from-file
    samples_set: Set[str] = set()
    if sample:
        for _sample in sample:
            logger.debug(f"Adding sample from command-line: {_sample}")
            samples_set.add(_sample)
            

    if samples_from_file:
        logger.debug(f"Reading samples from file: {samples_from_file}")
        try:
            with open(samples_from_file, 'r', encoding='utf-8') as f:
                file_samples = {line.strip() for line in f if line.strip()}
            samples_set.update(file_samples)
            logger.debug(f"Collected {len(file_samples)} samples from file.")
        except Exception as e:
            logger.error(f"Failed to read samples from file {samples_from_file}: {e}")
            sys.exit(1)

    # Deduplicate and validate sample paths
    valid_samples = []
    for sample_path in samples_set:
        if os.path.exists(sample_path):
            valid_samples.append(os.path.abspath(sample_path))
        else:
            logger.warning(f"Sample file does not exist and will be skipped: {sample_path}")

    if not valid_samples:
        logger.error("No valid samples provided for QC.")
        sys.exit(1)

    logger.info(f"Total valid samples to process: {len(valid_samples)}")

    # Load reference signature
    logger.info(f"Loading reference signature from: {ref}")
    try:
        reference_sig = SnipeSig(sourmash_sig=ref, sig_type=SigType.GENOME, enable_logging=debug)
        logger.debug(f"Loaded reference signature: {reference_sig.name}")
    except Exception as e:
        logger.error(f"Failed to load reference signature from {ref}: {e}")
        sys.exit(1)

    # Load amplicon signature if provided
    amplicon_sig = None
    if amplicon:
        logger.info(f"Loading amplicon signature from: {amplicon}")
        try:
            amplicon_sig = SnipeSig(sourmash_sig=amplicon, sig_type=SigType.AMPLICON, enable_logging=debug)
            logger.debug(f"Loaded amplicon signature: {amplicon_sig.name}")
        except Exception as e:
            logger.error(f"Failed to load amplicon signature from {amplicon}: {e}")
            sys.exit(1)
    
    # Load Y chromosome signature if provided
    ychr_sig = None
    if ychr:
        logger.info(f"Loading Y chromosome signature from: {ychr}")
        try:
            ychr_sig = SnipeSig(sourmash_sig=ychr, sig_type=SigType.GENOME, enable_logging=debug)
            logger.debug(f"Loaded Y chromosome signature: {ychr_sig.name}")
        except Exception as e:
            logger.error(f"Failed to load Y chromosome signature from {ychr}: {e}")
            sys.exit(1)

    # Prepare variable signatures if provided
    vars_paths = []
    vars_snipesigs = []
    if vars:
        logger.info(f"Loading {len(vars)} variable signature(s).")
        for path in vars:
            if not os.path.exists(path):
                logger.error(f"Variable signature file does not exist: {path}")
                sys.exit(1)
            vars_paths.append(os.path.abspath(path))
            try:
                var_sig = SnipeSig(sourmash_sig=path, sig_type=SigType.AMPLICON, enable_logging=debug)
                vars_snipesigs.append(var_sig)
                logger.debug(f"Loaded variable signature: {var_sig.name}")
            except Exception as e:
                logger.error(f"Failed to load variable signature from {path}: {e}")
                
        logger.debug(f"Variable signature paths: {vars_paths}")
        

    predict_extra_folds = [1, 2, 5, 9]
    
    
    qc_instance = MultiSigReferenceQC(
            reference_sig=reference_sig,
            amplicon_sig=amplicon_sig,
            ychr=ychr_sig if ychr_sig else None,
            varsigs=vars_snipesigs if vars_snipesigs else None,
            export_varsigs=export_var,
            enable_logging=debug
        )
    
    sample_to_stats = {}
    failed_samples = []
    for sample_path in tqdm(valid_samples):
        sample_sig = SnipeSig(sourmash_sig=sample_path, sig_type=SigType.SAMPLE, enable_logging=debug)
        try:
            sample_stats = qc_instance.process_sample(sample_sig=sample_sig,
                          predict_extra_folds = predict_extra_folds if roi else None,
                          advanced=True)
            sample_to_stats[sample_sig.name] = sample_stats
        except Exception as e:
            failed_samples.append(sample_sig.name)
            qc_instance.logger.error(f"Failed to process sample {sample_sig.name}: {e}")
            continue
    
    
    # Separate successful and failed results
    succeeded = list(sample_to_stats.keys())
    failed = len(failed_samples)

    # Handle complete failure
    if len(succeeded) == 0:
        logger.error("All samples failed during QC processing. Output TSV will not be generated.")
        sys.exit(1)
        
    # write total success and failure
    logger.info("Successfully processed samples: %d", len(succeeded))

    # Prepare the command-line invocation for comments
    command_invocation = ' '.join(sys.argv)

    # Create pandas DataFrame for succeeded samples
    df = pd.DataFrame(sample_to_stats.values())

    # Reorder columns to have 'sample' and 'file_path' first, if they exist
    cols = list(df.columns)
    reordered_cols = []
    for col in ['sample', 'file_path']:
        if col in cols:
            reordered_cols.append(col)
            cols.remove(col)
    reordered_cols += cols
    df = df[reordered_cols]

    # Export to TSV with comments
    try:
        with open(output, 'w', encoding='utf-8') as f:
            # Write comment with command invocation
            #! stop for now writing the comment
            # f.write(f"# Command: {command_invocation}\n")
            # Write the DataFrame to the file
            df.to_csv(f, sep='\t', index=False)
        logger.info(f"QC results successfully exported to {output}")
    except Exception as e:
        logger.error(f"Failed to export QC results to {output}: {e}")
        sys.exit(1)

    # Report failed samples if any
    if failed:
        failed_samples = [res['sample'] for res in failed]
        logger.warning(f"The following {len(failed_samples)} sample(s) failed during QC processing:")
        for sample in failed_samples:
            logger.warning(f"- {sample}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"QC process completed in {elapsed_time:.2f} seconds.")