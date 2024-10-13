import heapq
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from snipe.api import SnipeSig
from snipe.api.enums import SigType


class ReferenceQC:
    r"""
    Class for performing quality control of sequencing data against a reference genome.

    This class computes various metrics to assess the quality and characteristics of a sequencing sample, including coverage indices and abundance ratios, by comparing sample k-mer signatures with a reference genome and an optional amplicon signature.

    **Parameters**

    - `sample_sig` (`SnipeSig`): The sample k-mer signature (must be of type `SigType.SAMPLE`).
    - `reference_sig` (`SnipeSig`): The reference genome k-mer signature (must be of type `SigType.GENOME`).
    - `amplicon_sig` (`Optional[SnipeSig]`): The amplicon k-mer signature (must be of type `SigType.AMPLICON`), if applicable.
    - `enable_logging` (`bool`): Flag to enable detailed logging.

    **Attributes**

    - `sample_sig` (`SnipeSig`): The sample signature.
    - `reference_sig` (`SnipeSig`): The reference genome signature.
    - `amplicon_sig` (`Optional[SnipeSig]`): The amplicon signature.
    - `sample_stats` (`Dict[str, Any]`): Statistics of the sample signature.
    - `genome_stats` (`Dict[str, Any]`): Calculated genome-related statistics.
    - `amplicon_stats` (`Dict[str, Any]`): Calculated amplicon-related statistics (if `amplicon_sig` is provided).
    - `advanced_stats` (`Dict[str, Any]`): Calculated advanced statistics (optional).
    - `predicted_assay_type` (`str`): Predicted assay type based on metrics.

    **Calculated Metrics**

    The class calculates the following metrics:

    - **Total unique k-mers**
        - Description: Number of unique k-mers in the sample signature.
        - Calculation:
          $$
          \text{Total unique k-mers} = \left| \text{Sample k-mer set} \right|
          $$

    - **k-mer total abundance**
        - Description: Sum of abundances of all k-mers in the sample signature.
        - Calculation:
          $$
          \text{k-mer total abundance} = \sum_{k \in \text{Sample k-mer set}} \text{abundance}(k)
          $$

    - **k-mer mean abundance**
        - Description: Average abundance of k-mers in the sample signature.
        - Calculation:
          $$
          \text{k-mer mean abundance} = \frac{\text{k-mer total abundance}}{\text{Total unique k-mers}}
          $$

    - **k-mer median abundance**
        - Description: Median abundance of k-mers in the sample signature.
        - Calculation: Median of abundances in the sample k-mers.

    - **Number of singletons**
        - Description: Number of k-mers with an abundance of 1 in the sample signature.
        - Calculation:
          $$
          \text{Number of singletons} = \left| \{ k \in \text{Sample k-mer set} \mid \text{abundance}(k) = 1 \} \right|
          $$

    - **Genomic unique k-mers**
        - Description: Number of k-mers shared between the sample and the reference genome.
        - Calculation:
          $$
          \text{Genomic unique k-mers} = \left| \text{Sample k-mer set} \cap \text{Reference genome k-mer set} \right|
          $$

    - **Genome coverage index**
        - Description: Proportion of the reference genome's k-mers that are present in the sample.
        - Calculation:
          $$
          \text{Genome coverage index} = \frac{\text{Genomic unique k-mers}}{\left| \text{Reference genome k-mer set} \right|}
          $$

    - **Genomic k-mers total abundance**
        - Description: Sum of abundances for k-mers shared with the reference genome.
        - Calculation:
          $$
          \text{Genomic k-mers total abundance} = \sum_{k \in \text{Sample k-mer set} \cap \text{Reference genome k-mer set}} \text{abundance}(k)
          $$

    - **Genomic k-mers mean abundance**
        - Description: Average abundance of k-mers shared with the reference genome.
        - Calculation:
          $$
          \text{Genomic k-mers mean abundance} = \frac{\text{Genomic k-mers total abundance}}{\text{Genomic unique k-mers}}
          $$

    - **Mapping index**
        - Description: Proportion of the sample's total k-mer abundance that maps to the reference genome.
        - Calculation:
          $$
          \text{Mapping index} = \frac{\text{Genomic k-mers total abundance}}{\text{k-mer total abundance}}
          $$

    If `amplicon_sig` is provided, additional metrics are calculated:

    - **Amplicon unique k-mers**
        - Description: Number of k-mers shared between the sample and the amplicon.
        - Calculation:
          $$
          \text{Amplicon unique k-mers} = \left| \text{Sample k-mer set} \cap \text{Amplicon k-mer set} \right|
          $$

    - **Amplicon coverage index**
        - Description: Proportion of the amplicon's k-mers that are present in the sample.
        - Calculation:
          $$
          \text{Amplicon coverage index} = \frac{\text{Amplicon unique k-mers}}{\left| \text{Amplicon k-mer set} \right|}
          $$

    - **Amplicon k-mers total abundance**
        - Description: Sum of abundances for k-mers shared with the amplicon.
        - Calculation:
          $$
          \text{Amplicon k-mers total abundance} = \sum_{k \in \text{Sample k-mer set} \cap \text{Amplicon k-mer set}} \text{abundance}(k)
          $$

    - **Amplicon k-mers mean abundance**
        - Description: Average abundance of k-mers shared with the amplicon.
        - Calculation:
          $$
          \text{Amplicon k-mers mean abundance} = \frac{\text{Amplicon k-mers total abundance}}{\text{Amplicon unique k-mers}}
          $$

    - **Relative total abundance**
        - Description: Ratio of the amplicon k-mers total abundance to the genomic k-mers total abundance.
        - Calculation:
          $$
          \text{Relative total abundance} = \frac{\text{Amplicon k-mers total abundance}}{\text{Genomic k-mers total abundance}}
          $$

    - **Relative coverage**
        - Description: Ratio of the amplicon coverage index to the genome coverage index.
        - Calculation:
          $$
          \text{Relative coverage} = \frac{\text{Amplicon coverage index}}{\text{Genome coverage index}}
          $$

    - **Predicted Assay Type**
        - Description: Predicted assay type based on the `Relative total abundance`.
        - Calculation:
          - If \(\text{Relative total abundance} \leq 0.0809\), then **WGS** (Whole Genome Sequencing).
          - If \(\text{Relative total abundance} \geq 0.1188\), then **WXS** (Whole Exome Sequencing).
          - If between these values, assign based on the closest threshold.

    **Advanced Metrics** (optional, calculated if `include_advanced` is `True`):

    - **Median-trimmed unique k-mers**
        - Description: Number of unique k-mers in the sample after removing k-mers with abundance below the median.
        - Calculation:
          - Remove k-mers where \(\text{abundance}(k) < \text{Median abundance}\).
          - Count the remaining k-mers.

    - **Median-trimmed total abundance**
        - Description: Sum of abundances after median trimming.
        - Calculation:
          $$
          \text{Median-trimmed total abundance} = \sum_{k \in \text{Median-trimmed Sample k-mer set}} \text{abundance}(k)
          $$

    - **Median-trimmed mean abundance**
        - Description: Average abundance after median trimming.
        - Calculation:
          $$
          \text{Median-trimmed mean abundance} = \frac{\text{Median-trimmed total abundance}}{\text{Median-trimmed unique k-mers}}
          $$

    - **Median-trimmed median abundance**
        - Description: Median abundance after median trimming.
        - Calculation: Median of abundances in the median-trimmed sample.

    - **Median-trimmed Genomic unique k-mers**
        - Description: Number of genomic k-mers in the median-trimmed sample.
        - Calculation:
          $$
          \text{Median-trimmed Genomic unique k-mers} = \left| \text{Median-trimmed Sample k-mer set} \cap \text{Reference genome k-mer set} \right|
          $$

    - **Median-trimmed Genome coverage index**
        - Description: Genome coverage index after median trimming.
        - Calculation:
          $$
          \text{Median-trimmed Genome coverage index} = \frac{\text{Median-trimmed Genomic unique k-mers}}{\left| \text{Reference genome k-mer set} \right|}
          $$

    - **Median-trimmed Amplicon unique k-mers** (if `amplicon_sig` is provided)
        - Description: Number of amplicon k-mers in the median-trimmed sample.
        - Calculation:
          $$
          \text{Median-trimmed Amplicon unique k-mers} = \left| \text{Median-trimmed Sample k-mer set} \cap \text{Amplicon k-mer set} \right|
          $$

    - **Median-trimmed Amplicon coverage index**
        - Description: Amplicon coverage index after median trimming.
        - Calculation:
          $$
          \text{Median-trimmed Amplicon coverage index} = \frac{\text{Median-trimmed Amplicon unique k-mers}}{\left| \text{Amplicon k-mer set} \right|}
          $$

    - **Median-trimmed relative coverage**
        - Description: Relative coverage after median trimming.
        - Calculation:
          $$
          \text{Median-trimmed relative coverage} = \frac{\text{Median-trimmed Amplicon coverage index}}{\text{Median-trimmed Genome coverage index}}
          $$

    - **Median-trimmed relative mean abundance**
        - Description: Ratio of median-trimmed amplicon mean abundance to median-trimmed genomic mean abundance.
        - Calculation:
          $$
          \text{Median-trimmed relative mean abundance} = \frac{\text{Median-trimmed Amplicon mean abundance}}{\text{Median-trimmed Genomic mean abundance}}
          $$

    **Usage Example**

    ```python
    qc = ReferenceQC(
        sample_sig=sample_signature,
        reference_sig=reference_signature,
        amplicon_sig=amplicon_signature,
        enable_logging=True
    )

    stats = qc.get_aggregated_stats(include_advanced=True)
    ```
    ```
    """

    def __init__(self, *,
                 sample_sig: SnipeSig,
                 reference_sig: SnipeSig,
                 amplicon_sig: Optional[SnipeSig] = None,
                 enable_logging: bool = False,
                 chr_name_to_sig: Optional[Dict[str, SnipeSig]] = None,
                 flag_chr_specific: bool = False,
                 flag_ychr_in_reference: bool = False,
                 **kwargs):
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        if enable_logging:
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.hasHandlers():
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
            self.logger.debug("Logging is enabled for ReferenceQC.")
        else:
            self.logger.setLevel(logging.CRITICAL)
            
        # logging all passed parameters
        self.logger.debug("passed parameters:\n")
        for key, value in locals().items():
            self.logger.debug("\t%s: %s", key, value)
            

        # Verify signature types
        if sample_sig._type != SigType.SAMPLE:
            self.logger.error("Invalid signature type for sample_sig: %s | %s", sample_sig.sigtype, sample_sig._type)
            raise ValueError(f"sample_sig must be of type {SigType.SAMPLE}, got {sample_sig.sigtype}")

        if reference_sig.sigtype != SigType.GENOME:
            self.logger.error("Invalid signature type for reference_sig: %s", reference_sig.sigtype)
            raise ValueError(f"reference_sig must be of type {SigType.GENOME}, got {reference_sig.sigtype}")

        if amplicon_sig is not None and amplicon_sig.sigtype != SigType.AMPLICON:
            self.logger.error("Invalid signature type for amplicon_sig: %s", amplicon_sig.sigtype)
            raise ValueError(f"amplicon_sig must be of type {SigType.AMPLICON}, got {amplicon_sig.sigtype}")


        self.flag_ychr_in_reference = flag_ychr_in_reference
        self.logger.debug("Chromosome specific signatures provided.")
        self.flag_activate_sex_metrics = True
        self.chr_name_to_sig: Dict[str, SnipeSig] = chr_name_to_sig
        self.chr_to_specific_kmers: Dict[str, SnipeSig] = {}
        self.flag_chr_specific = flag_chr_specific
        if self.flag_chr_specific:
            self._create_chr_specific_sigs_if_needed()
            



        self.sample_sig = sample_sig
        self.reference_sig = reference_sig
        self.amplicon_sig = amplicon_sig
        self.enable_logging = enable_logging

        # Initialize attributes
        self.sample_stats: Dict[str, Any] = {}
        self.genome_stats: Dict[str, Any] = {}
        self.amplicon_stats: Dict[str, Any] = {}
        self.advanced_stats: Dict[str, Any] = {}
        self.predicted_assay_type: str = ""

        # Set grey zone thresholds
        self.relative_total_abundance_grey_zone = [0.08092723407173719, 0.11884490500267662]

        # Get sample statistics
        self.logger.debug("Getting sample statistics.")
        self.sample_stats_raw = self.sample_sig.get_sample_stats

        # Get reference genome statistics
        self.logger.debug("Getting reference genome statistics.")
        self.genome_sig_stats = self.reference_sig.get_sample_stats

        # If amplicon_sig is provided, get its stats
        if self.amplicon_sig is not None:
            self.logger.debug("Getting amplicon statistics.")
            self.amplicon_sig_stats = self.amplicon_sig.get_sample_stats

        # Compute metrics
        self.logger.debug("Calculating statistics.")
        self._calculate_stats()
    
    def _create_chr_specific_sigs_if_needed(self):
        # process the chromosome specific kmers to make sure no common kmers between them
        
        # first we generate chr-specific kmers
        for chr_name in self.chr_name_to_sig.keys():
            pass
            
        
        # if the genome has y chromosome, then we need to remove the y chromosome from the reference genome
        reference_genome: SnipeSig = self.reference_sig
        if self.flag_ychr_in_reference:
            reference_genome = reference_genome - self.chr_name_to_sig['y']
        
        for chr_name in self.chr_name_to_sig.keys():
            chr_name = chr_name.lower()
            
            if chr_name == 'y':
                continue
            self.logger.debug("Processing chromosome: %s", chr_name)
            if chr_name not in self.chr_name_to_sig:
                self.logger.error("Chromosome %s not found in the chromosome list.", chr_name)
                raise ValueError(f"Chromosome {chr_name} not found in the chromosome list.")
            
            # log original size and new size of the chr
            self.logger.debug("Original size of %s: %d", chr_name, len(self.chr_name_to_sig[chr_name]))
            self.chr_name_to_sig[chr_name] = self.chr_name_to_sig[chr_name] - reference_genome
            
            # subtract the genome from the chromosome

    def _calculate_stats(self):
        r"""
        Calculate the various metrics based on the sample, reference, and optional amplicon signatures.
        """
        # ============= SAMPLE STATS =============
        self.logger.debug("Processing sample statistics.")
        self.sample_stats = {
            "Total unique k-mers": self.sample_stats_raw["num_hashes"],
            "k-mer total abundance": self.sample_stats_raw["total_abundance"],
            "k-mer mean abundance": self.sample_stats_raw["mean_abundance"],
            "k-mer median abundance": self.sample_stats_raw["median_abundance"],
            "num_singletons": self.sample_stats_raw["num_singletons"],
            "ksize": self.sample_stats_raw["ksize"],
            "scale": self.sample_stats_raw["scale"],
            "name": self.sample_stats_raw["name"],
            "filename": self.sample_stats_raw["filename"],
        }

        # ============= GENOME STATS =============
        self.logger.debug("Calculating genome statistics.")
        # Compute intersection of sample and reference genome
        self.logger.debug("Type of sample_sig: %s | Type of reference_sig: %s", type(self.sample_sig), type(self.reference_sig))
        sample_genome = self.sample_sig & self.reference_sig
        # Get stats (call get_sample_stats only once)

        # Log hashes and abundances for both sample and reference
        self.logger.debug("Sample hashes: %s", self.sample_sig.hashes)
        self.logger.debug("Sample abundances: %s", self.sample_sig.abundances)
        self.logger.debug("Reference hashes: %s", self.reference_sig.hashes)
        self.logger.debug("Reference abundances: %s", self.reference_sig.abundances)

        sample_genome_stats = sample_genome.get_sample_stats

        self.genome_stats = {
            "Genomic unique k-mers": sample_genome_stats["num_hashes"],
            "Genomic k-mers total abundance": sample_genome_stats["total_abundance"],
            "Genomic k-mers mean abundance": sample_genome_stats["mean_abundance"],
            "Genomic k-mers median abundance": sample_genome_stats["median_abundance"],
            # Genome coverage index
            "Genome coverage index": (
                sample_genome_stats["num_hashes"] / self.genome_sig_stats["num_hashes"]
                if self.genome_sig_stats["num_hashes"] > 0 else 0
            ),
            # Mapping index
            "Mapping index": (
                sample_genome_stats["total_abundance"] / self.sample_stats["k-mer total abundance"]
                if self.sample_stats["k-mer total abundance"] > 0 else 0
            ),
        }

        # ============= AMPLICON STATS =============
        if self.amplicon_sig is not None:
            self.logger.debug("Calculating amplicon statistics.")
            # Compute intersection of sample and amplicon
            sample_amplicon = self.sample_sig & self.amplicon_sig
            # Get stats (call get_sample_stats only once)
            sample_amplicon_stats = sample_amplicon.get_sample_stats

            self.amplicon_stats = {
                "Amplicon unique k-mers": sample_amplicon_stats["num_hashes"],
                "Amplicon k-mers total abundance": sample_amplicon_stats["total_abundance"],
                "Amplicon k-mers mean abundance": sample_amplicon_stats["mean_abundance"],
                "Amplicon k-mers median abundance": sample_amplicon_stats["median_abundance"],
                # Amplicon coverage index
                "Amplicon coverage index": (
                    sample_amplicon_stats["num_hashes"] / self.amplicon_sig_stats["num_hashes"]
                    if self.amplicon_sig_stats["num_hashes"] > 0 else 0
                ),
            }

            # Relative metrics
            self.amplicon_stats["Relative total abundance"] = (
                self.amplicon_stats["Amplicon k-mers total abundance"] / self.genome_stats["Genomic k-mers total abundance"]
                if self.genome_stats["Genomic k-mers total abundance"] > 0 else 0
            )
            self.amplicon_stats["Relative coverage"] = (
                self.amplicon_stats["Amplicon coverage index"] / self.genome_stats["Genome coverage index"]
                if self.genome_stats["Genome coverage index"] > 0 else 0
            )

            # Predicted assay type
            relative_total_abundance = self.amplicon_stats["Relative total abundance"]
            if relative_total_abundance <= self.relative_total_abundance_grey_zone[0]:
                self.predicted_assay_type = "WGS"
            elif relative_total_abundance >= self.relative_total_abundance_grey_zone[1]:
                self.predicted_assay_type = "WXS"
            else:
                # Assign based on the closest threshold
                distance_to_wgs = abs(relative_total_abundance - self.relative_total_abundance_grey_zone[0])
                distance_to_wxs = abs(relative_total_abundance - self.relative_total_abundance_grey_zone[1])
                self.predicted_assay_type = "WGS" if distance_to_wgs < distance_to_wxs else "WXS"
            self.logger.debug("Predicted assay type: %s", self.predicted_assay_type)

    def get_aggregated_stats(self, include_advanced: bool = False) -> Dict[str, Any]:
        r"""
        Retrieve aggregated statistics from the quality control analysis.

        **Parameters**

        - `include_advanced (bool)`:  
          If set to `True`, includes advanced metrics in the aggregated statistics.

        **Returns**

        - `Dict[str, Any]`:  
          A dictionary containing the aggregated statistics, which may include:
          - Sample statistics
          - Genome statistics
          - Amplicon statistics (if provided)
          - Predicted assay type
          - Advanced statistics (if `include_advanced` is `True`)
        """
        aggregated_stats: Dict[str, Any] = {}
        # Include sample_stats
        aggregated_stats.update(self.sample_stats)
        # Include genome_stats
        aggregated_stats.update(self.genome_stats)
        # Include amplicon_stats if available
        if self.amplicon_sig is not None:
            aggregated_stats.update(self.amplicon_stats)
        # Include predicted assay type if amplicon_sig is provided
        if self.amplicon_sig is not None:
            aggregated_stats["Predicted Assay Type"] = self.predicted_assay_type
        # Include advanced_stats if requested
        if include_advanced:
            self._calculate_advanced_stats()
            aggregated_stats.update(self.advanced_stats)
        return aggregated_stats

    def _calculate_advanced_stats(self):
        r"""
        Calculate advanced statistics, such as median-trimmed metrics.
        """
        self.logger.debug("Calculating advanced statistics.")

        # Copy sample signature to avoid modifying the original
        median_trimmed_sample_sig = self.sample_sig.copy()
        # Trim below median
        median_trimmed_sample_sig.trim_below_median()
        # Get stats
        median_trimmed_sample_stats = median_trimmed_sample_sig.get_sample_stats
        self.advanced_stats.update({
            "Median-trimmed unique k-mers": median_trimmed_sample_stats["num_hashes"],
            "Median-trimmed total abundance": median_trimmed_sample_stats["total_abundance"],
            "Median-trimmed mean abundance": median_trimmed_sample_stats["mean_abundance"],
            "Median-trimmed median abundance": median_trimmed_sample_stats["median_abundance"],
        })

        # Genome stats for median-trimmed sample
        median_trimmed_sample_genome = median_trimmed_sample_sig & self.reference_sig
        median_trimmed_sample_genome_stats = median_trimmed_sample_genome.get_sample_stats
        self.advanced_stats.update({
            "Median-trimmed Genomic unique k-mers": median_trimmed_sample_genome_stats["num_hashes"],
            "Median-trimmed Genomic total abundance": median_trimmed_sample_genome_stats["total_abundance"],
            "Median-trimmed Genomic mean abundance": median_trimmed_sample_genome_stats["mean_abundance"],
            "Median-trimmed Genomic median abundance": median_trimmed_sample_genome_stats["median_abundance"],
            "Median-trimmed Genome coverage index": (
                median_trimmed_sample_genome_stats["num_hashes"] / self.genome_sig_stats["num_hashes"]
                if self.genome_sig_stats["num_hashes"] > 0 else 0
            ),
        })

        if self.amplicon_sig is not None:
            # Amplicon stats for median-trimmed sample
            median_trimmed_sample_amplicon = median_trimmed_sample_sig & self.amplicon_sig
            median_trimmed_sample_amplicon_stats = median_trimmed_sample_amplicon.get_sample_stats
            self.advanced_stats.update({
                "Median-trimmed Amplicon unique k-mers": median_trimmed_sample_amplicon_stats["num_hashes"],
                "Median-trimmed Amplicon total abundance": median_trimmed_sample_amplicon_stats["total_abundance"],
                "Median-trimmed Amplicon mean abundance": median_trimmed_sample_amplicon_stats["mean_abundance"],
                "Median-trimmed Amplicon median abundance": median_trimmed_sample_amplicon_stats["median_abundance"],
                "Median-trimmed Amplicon coverage index": (
                    median_trimmed_sample_amplicon_stats["num_hashes"] / self.amplicon_sig_stats["num_hashes"]
                    if self.amplicon_sig_stats["num_hashes"] > 0 else 0
                ),
            })
            # Additional advanced relative metrics
            self.amplicon_stats["Median-trimmed relative coverage"] = (
                self.advanced_stats["Median-trimmed Amplicon coverage index"] / self.advanced_stats["Median-trimmed Genome coverage index"]
                if self.advanced_stats["Median-trimmed Genome coverage index"] > 0 else 0
            )
            self.amplicon_stats["Median-trimmed relative mean abundance"] = (
                self.advanced_stats["Median-trimmed Amplicon mean abundance"] / self.advanced_stats["Median-trimmed Genomic mean abundance"]
                if self.advanced_stats["Median-trimmed Genomic mean abundance"] > 0 else 0
            )
            # Update amplicon_stats with advanced metrics
            self.amplicon_stats.update({
                "Median-trimmed relative coverage": self.amplicon_stats["Median-trimmed relative coverage"],
                "Median-trimmed relative mean abundance": self.amplicon_stats["Median-trimmed relative mean abundance"],
            })

    def split_sig_randomly(self, n: int) -> List[SnipeSig]:
        """
        Split the sample signature into `n` random parts based on abundances.

        Parameters:
            n (int): Number of parts to split into.

        Returns:
            List[SnipeSig]: List of `SnipeSig` instances representing the split parts.
        """
        self.logger.debug("Splitting sample signature into %d random parts.", n)
        # Get k-mers and abundances
        hash_to_abund = dict(zip(self.sample_sig.hashes, self.sample_sig.abundances))
        random_split_sigs = self.distribute_kmers_random(hash_to_abund, n)
        split_sigs = [
            SnipeSig.create_from_hashes_abundances(
                hashes=np.array(list(kmer_dict.keys()), dtype=np.uint64),
                abundances=np.array(list(kmer_dict.values()), dtype=np.uint32),
                ksize=self.sample_sig.ksize,
                scale=self.sample_sig.scale,
                name=f"{self.sample_sig.name}_part_{i+1}",
                filename=self.sample_sig.filename,
                enable_logging=self.enable_logging
            )
            for i, kmer_dict in enumerate(random_split_sigs)
        ]
        return split_sigs

    @staticmethod
    def distribute_kmers_random(original_dict: Dict[int, int], n: int) -> List[Dict[int, int]]:
        """
        Distribute the k-mers randomly into `n` parts based on their abundances.

        Parameters:
            original_dict (Dict[int, int]): Dictionary mapping k-mer hashes to their abundances.
            n (int): Number of parts to split into.

        Returns:
            List[Dict[int, int]]: List of dictionaries, each mapping k-mer hashes to their abundances.
        """
        # Initialize the resulting dictionaries
        distributed_dicts = [{} for _ in range(n)]

        # For each k-mer and its abundance
        for kmer_hash, abundance in original_dict.items():
            if abundance == 0:
                continue  # Skip zero abundances
            # Generate multinomial split of abundance
            counts = np.random.multinomial(abundance, [1.0 / n] * n)
            # Update each dictionary
            for i in range(n):
                if counts[i] > 0:
                    distributed_dicts[i][kmer_hash] = counts[i]

        return distributed_dicts

    def calculate_coverage_vs_depth(self, n: int = 30) -> List[Dict[str, Any]]:
        """
        Calculate cumulative coverage index vs cumulative sequencing depth.

        Parameters:
            n (int): Number of parts to split the signature into.

        Returns:
            List[Dict[str, Any]]: List of stats for each cumulative part.
        """
        self.logger.debug("Calculating coverage vs depth with %d parts.", n)
        # Determine the ROI reference signature
        if isinstance(self.amplicon_sig, SnipeSig):
            roi_reference_sig = self.amplicon_sig
            self.logger.debug("Using amplicon signature as ROI reference.")
        else:
            roi_reference_sig = self.reference_sig
            self.logger.debug("Using reference genome signature as ROI reference.")

        # Split the sample signature into n random parts
        split_sigs = self.split_sig_randomly(n)

        coverage_depth_data = []

        cumulative_snipe_sig = split_sigs[0].copy()
        cumulative_total_abundance = cumulative_snipe_sig.total_abundance

        # Compute initial coverage index
        cumulative_qc = ReferenceQC(
            sample_sig=cumulative_snipe_sig,
            reference_sig=roi_reference_sig,
            enable_logging=self.enable_logging
        )
        cumulative_stats = cumulative_qc.get_aggregated_stats()
        cumulative_coverage_index = cumulative_stats["Genome coverage index"]

        coverage_depth_data.append({
            "cumulative_parts": 1,
            "cumulative_total_abundance": cumulative_total_abundance,
            "cumulative_coverage_index": cumulative_coverage_index,
        })

        # Iterate over the rest of the parts
        for i in range(1, n):
            current_part = split_sigs[i]

            # Add current part to cumulative signature
            cumulative_snipe_sig += current_part
            cumulative_total_abundance += current_part.total_abundance

            # Compute new coverage index
            cumulative_qc = ReferenceQC(
                sample_sig=cumulative_snipe_sig,
                reference_sig=roi_reference_sig,
                enable_logging=self.enable_logging
            )
            cumulative_stats = cumulative_qc.get_aggregated_stats()
            cumulative_coverage_index = cumulative_stats["Genome coverage index"]

            coverage_depth_data.append({
                "cumulative_parts": i + 1,
                "cumulative_total_abundance": cumulative_total_abundance,
                "cumulative_coverage_index": cumulative_coverage_index,
            })

        self.logger.debug("Coverage vs depth calculation completed.")
        return coverage_depth_data

    def predict_coverage(self, extra_fold: float, n: int = 30) -> float:
        r"""
        Predict the coverage index if additional sequencing is performed.

        This method estimates the potential increase in the genome coverage index when the sequencing depth
        is increased by a specified fold (extra sequencing). It does so by:

        1. **Cumulative Coverage Calculation**:
        - Splitting the sample signature into `n` random parts to simulate incremental sequencing data.
        - Calculating the cumulative coverage index and cumulative sequencing depth at each incremental step.

        2. **Saturation Curve Fitting**:
        - Modeling the relationship between cumulative coverage and cumulative sequencing depth using
            a hyperbolic saturation function.
        - The saturation model reflects how coverage approaches a maximum limit as sequencing depth increases.

        3. **Coverage Prediction**:
        - Using the fitted model to predict the coverage index at an increased sequencing depth (current depth
            multiplied by `1 + extra_fold`).

        **Mathematical Explanation**:

        - **Saturation Model**:
        The coverage index \( C \) as a function of sequencing depth \( D \) is modeled using the function:

        $$
        C(D) = \\frac{a \cdot D}{b + D}
        $$

        Where:
        - \( a \) and \( b \) are parameters estimated from the data.
        - \( D \) is the cumulative sequencing depth (total abundance).
        - \( C(D) \) is the cumulative coverage index at depth \( D \).

        - **Parameter Estimation**:
        The parameters \( a \) and \( b \) are determined by fitting the model to the observed cumulative
        coverage and depth data using non-linear least squares optimization.

        - **Coverage Prediction**:
        The predicted coverage index \( C_{\text{pred}} \) at an increased sequencing depth \( D_{\text{pred}} \)
        is calculated as:

        $$
        D_{\text{pred}} = D_{\text{current}} \times (1 + \text{extra\_fold})
        $$

        $$
        C_{\text{pred}} = \\frac{a \cdot D_{\text{pred}}}{b + D_{\text{pred}}}
        $$

        **Parameters**:
            
        - `extra_fold` (*float*):  
        The fold increase in sequencing depth to simulate. For example, extra_fold = 1.0 represents doubling
        the current sequencing depth.
        
        - `n` (*int, optional*):  
        The number of parts to split the sample signature into for modeling the saturation curve.
        Default is 30.

        **Returns**:
            - `float`:  
            The predicted genome coverage index at the increased sequencing depth.

        **Raises**:
            - `RuntimeError`:  
            If the saturation model fails to converge during curve fitting.

        **Usage Example**:

        ```python
        # Create a ReferenceQC instance with sample and reference signatures
        qc = ReferenceQC(sample_sig=sample_sig, reference_sig=reference_sig)

        # Predict coverage index after increasing sequencing depth by 50%
        predicted_coverage = qc.predict_coverage(extra_fold=0.5)

        print(f"Predicted coverage index at 1.5x sequencing depth: {predicted_coverage:.4f}")
        ```

        **Implementation Details**:

        - **Splitting the Sample Signature**:
            - The sample signature is split into `n` random parts using a multinomial distribution based on k-mer abundances.
            - Each part represents an incremental addition of sequencing data.

        - **Cumulative Calculations**:
            - At each incremental step, the cumulative signature is updated, and the cumulative coverage index and sequencing depth are calculated.

        - **Curve Fitting**:
            - The `scipy.optimize.curve_fit` function is used to fit the saturation model to the cumulative data.
            - Initial parameter guesses are based on the observed data to aid convergence.
        """
        if extra_fold < 1:
            raise ValueError("extra_fold must be >= 1.0.")

        if n < 1 or not isinstance(n, int):
            raise ValueError("n must be a positive integer.")

        self.logger.debug("Predicting coverage with extra fold: %f", extra_fold)
        coverage_depth_data = self.calculate_coverage_vs_depth(n=n)

        # Extract cumulative total abundance and coverage index
        x_data = np.array([d["cumulative_total_abundance"] for d in coverage_depth_data])
        y_data = np.array([d["cumulative_coverage_index"] for d in coverage_depth_data])

        # Saturation model function
        def saturation_model(x, a, b):
            return a * x / (b + x)

        # Initial parameter guesses
        initial_guess = [y_data[-1], x_data[int(len(x_data) / 2)]]

        # Fit the model to the data
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                params, covariance = curve_fit(
                    saturation_model,
                    x_data,
                    y_data,
                    p0=initial_guess,
                    bounds=(0, np.inf),
                    maxfev=10000
                )
        except (RuntimeError, OptimizeWarning) as exc:
            self.logger.error("Curve fitting failed.")
            raise RuntimeError("Saturation model fitting failed. Cannot predict coverage.") from exc

        # Check if covariance contains inf or nan
        if np.isinf(covariance).any() or np.isnan(covariance).any():
            self.logger.error("Covariance of parameters could not be estimated.")
            raise RuntimeError("Saturation model fitting failed. Cannot predict coverage.")

        a, b = params

        # Predict coverage at increased sequencing depth
        total_abundance = x_data[-1]
        predicted_total_abundance = total_abundance * (1 + extra_fold)
        predicted_coverage = saturation_model(predicted_total_abundance, a, b)

        # Ensure the predicted coverage does not exceed maximum possible coverage
        max_coverage = 1.0  # Coverage index cannot exceed 1
        predicted_coverage = min(predicted_coverage, max_coverage)

        self.logger.debug("Predicted coverage at %.2f-fold increase: %f", extra_fold, predicted_coverage)
        return predicted_coverage

    def calculate_sex_metrics(self) -> Dict[str, Any]:
        """_summary_
        
        1) Load individual chromosome k-mer signatures
        2) Create a new version so each sig has its own specific hashes (no common hashes)
        3) Calculate the sample mean abundance for each chromosome
        4) Calculate the CV (Coefficient of Variation) for the whole sample by using point (3)
        5) Calculate the x-ploidy score = 

        """
        
