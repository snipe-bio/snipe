import heapq
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from snipe.api.snipe_sig import SnipeSig
from snipe.api.enums import SigType
import os
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from typing import Optional
import sourmash

# pylint disable C0301


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
    """

    def __init__(self, *,
                 sample_sig: SnipeSig,
                 reference_sig: SnipeSig,
                 amplicon_sig: Optional[SnipeSig] = None,
                 enable_logging: bool = False,
                 **kwargs):
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize split cache
        self._split_cache: Dict[int, List[SnipeSig]] = {}
        self.logger.debug("Initialized split cache.")


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
            
            
        # Validate that all signatures have the same ksize and scale
        self.logger.debug("Validating ksize and scale across signatures.")
        if sample_sig.ksize != reference_sig.ksize:
            self.logger.error("K-mer sizes do not match: sample_sig.ksize=%d vs reference_sig.ksize=%d",
                              sample_sig.ksize, reference_sig.ksize)
            raise ValueError(f"sample_sig kszie ({sample_sig.ksize}) does not match reference_sig ksize ({reference_sig.ksize}).")
        if sample_sig.scale != reference_sig.scale:
            self.logger.error("Scale values do not match: sample_sig.scale=%d vs reference_sig.scale=%d",
                              sample_sig.scale, reference_sig.scale)
            raise ValueError(f"sample_sig scale ({sample_sig.scale}) does not match reference_sig scale ({reference_sig.scale}).")
        
        if amplicon_sig is not None:
            if amplicon_sig.ksize != sample_sig.ksize:
                self.logger.error("K-mer sizes do not match: amplicon_sig.ksize=%d vs sample_sig.ksize=%d",
                                  amplicon_sig.ksize, sample_sig.ksize)
                raise ValueError(f"amplicon_sig ksize ({amplicon_sig.ksize}) does not match sample_sig ksize ({sample_sig.ksize}).")
            if amplicon_sig.scale != sample_sig.scale:
                self.logger.error("Scale values do not match: amplicon_sig.scale=%d vs sample_sig.scale=%d",
                                  amplicon_sig.scale, sample_sig.scale)
                raise ValueError(f"amplicon_sig scale ({amplicon_sig.scale}) does not match sample_sig scale ({sample_sig.scale}).")

        self.logger.debug("All signatures have matching ksize and scale.")
            

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


        self.logger.debug("Chromosome specific signatures provided.")
        self.flag_activate_sex_metrics = True


        self.sample_sig = sample_sig
        self.reference_sig = reference_sig
        self.amplicon_sig = amplicon_sig
        self.enable_logging = enable_logging

        # Initialize attributes
        self.sample_stats: Dict[str, Any] = {}
        self.genome_stats: Dict[str, Any] = {}
        self.amplicon_stats: Dict[str, Any] = {}
        self.advanced_stats: Dict[str, Any] = {}
        self.chrs_stats: Dict[str, Dict[str, Any]] = {}
        self.sex_stats: Dict[str, Any] = {}
        self.predicted_error_contamination_index: Dict[str, Any] = {}
        self.vars_nonref_stats: Dict[str, Any] = {}
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
        self.logger.debug("Type of sample_sig: %s | Type of reference_sig: %s", self.sample_sig.sigtype, self.reference_sig.sigtype)
        sample_genome = self.sample_sig & self.reference_sig
        # Get stats (call get_sample_stats only once)

        # Log hashes and abundances for both sample and reference
        # self.logger.debug("Sample hashes: %s", self.sample_sig.hashes)
        # self.logger.debug("Sample abundances: %s", self.sample_sig.abundances)
        # self.logger.debug("Reference hashes: %s", self.reference_sig.hashes)
        # self.logger.debug("Reference abundances: %s", self.reference_sig.abundances)

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

            # ============= RELATIVE STATS =============
            self.amplicon_stats["Relative total abundance"] = (
                self.amplicon_stats["Amplicon k-mers total abundance"] / self.genome_stats["Genomic k-mers total abundance"]
                if self.genome_stats["Genomic k-mers total abundance"] > 0 else 0
            )
            self.amplicon_stats["Relative coverage"] = (
                self.amplicon_stats["Amplicon coverage index"] / self.genome_stats["Genome coverage index"]
                if self.genome_stats["Genome coverage index"] > 0 else 0
            )

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
        
        self.logger.debug("Calculuating error and contamination indices.")
        try:
            sample_nonref = self.sample_sig - self.reference_sig
            sample_nonref_singletons = sample_nonref.count_singletons()
            sample_nonref_non_singletons = sample_nonref.total_abundance - sample_nonref_singletons
            sample_total_abundance = self.sample_sig.total_abundance
            
            predicted_error_index = sample_nonref_singletons / sample_total_abundance
            predicted_contamination_index = sample_nonref_non_singletons / sample_total_abundance

            # predict error and contamination index
            self.predicted_error_contamination_index["Predicted contamination index"] = predicted_contamination_index
            self.predicted_error_contamination_index["Sequencing errors index"] = predicted_error_index
        # except zero division error
        except ZeroDivisionError:
            self.logger.error("Please check the sample signature, it seems to be empty.")
        

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
            self.logger.debug("While aggregating stats; amplicon signature provided.")
            aggregated_stats.update(self.amplicon_stats)
            aggregated_stats["Predicted Assay Type"] = self.predicted_assay_type
                    
        if self.chrs_stats:
            aggregated_stats.update(self.chrs_stats)
        
        if self.sex_stats:
            aggregated_stats.update(self.sex_stats)
            
        if self.vars_nonref_stats:
            aggregated_stats.update(self.vars_nonref_stats)

        # Include advanced_stats if requested
        if include_advanced:
            self._calculate_advanced_stats()
            aggregated_stats.update(self.advanced_stats)
            
        if self.predicted_error_contamination_index:
            aggregated_stats.update(self.predicted_error_contamination_index)
        
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
            self.logger.debug("Calculating advanced amplicon statistics.")
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
            self.logger.debug("Calculating advanced relative metrics.")
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

            self.advanced_stats.update(self.amplicon_stats)

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
            self.logger.debug("Calculating advanced amplicon statistics.")
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
            self.logger.debug("Calculating advanced relative metrics.")
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
            
            self.advanced_stats.update(self.amplicon_stats)

    def split_sig_randomly(self, n: int) -> List[SnipeSig]:
        r"""
        Split the sample signature into `n` random parts based on abundances.

        This method distributes the k-mers of the sample signature into `n` parts using a multinomial distribution
        based on their abundances. Each k-mer's abundance is split across the `n` parts proportionally.

        **Mathematical Explanation**:

        For each k-mer with hash \( h \) and abundance \( a_h \), its abundance is distributed into \( n \) parts
        according to a multinomial distribution. Specifically, the abundance in each part \( i \) is given by:

        $$
        a_{h,i} \sim \text{Multinomial}(a_h, \frac{1}{n}, \frac{1}{n}, \dots, \frac{1}{n})
        $$

        Where:
        - \( a_{h,i} \) is the abundance of k-mer \( h \) in part \( i \).
        - Each \( a_{h,i} \) is a non-negative integer such that \( \sum_{i=1}^{n} a_{h,i} = a_h \).

        **Parameters**:

        - `n` (`int`): Number of parts to split into.

        **Returns**:

        - `List[SnipeSig]`:  
          List of `SnipeSig` instances representing the split parts.

        **Usage Example**:

        ```python
        split_sigs = qc.split_sig_randomly(n=3)
        for idx, sig in enumerate(split_sigs, 1):
            print(f"Signature part {idx}: {sig}")
        ```
        """
        self.logger.debug("Attempting to split sample signature into %d random parts.", n)

        # Check if the split for this n is already cached
        if n in self._split_cache:
            self.logger.debug("Using cached split signatures for n=%d.", n)
            # Return deep copies to prevent external modifications
            return [sig.copy() for sig in self._split_cache[n]]

        self.logger.debug("No cached splits found for n=%d. Proceeding to split.", n)
        # Get k-mers and abundances
        _sample_genome = self.sample_sig & self.reference_sig
        hash_to_abund = dict(zip(_sample_genome.hashes, _sample_genome.abundances))
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

        # Cache the split signatures
        self._split_cache[n] = split_sigs
        self.logger.debug("Cached split signatures for n=%d.", n)

        return split_sigs
    
    @staticmethod
    def distribute_kmers_random(original_dict: Dict[int, int], n: int) -> List[Dict[int, int]]:
        r"""
        Distribute the k-mers randomly into `n` parts based on their abundances.

        This helper method performs the actual distribution of k-mers using a multinomial distribution.

        **Mathematical Explanation**:

        Given a k-mer with hash \( h \) and abundance \( a_h \), the distribution of its abundance across \( n \)
        parts is modeled as:

        $$
        a_{h,1}, a_{h,2}, \dots, a_{h,n} \sim \text{Multinomial}(a_h, p_1, p_2, \dots, p_n)
        $$

        Where \( p_i = \frac{1}{n} \) for all \( i \).

        **Parameters**:

        - `original_dict` (`Dict[int, int]`):  
          Dictionary mapping k-mer hashes to their abundances.
        - `n` (`int`): Number of parts to split into.

        **Returns**:

        - `List[Dict[int, int]]`:  
          List of dictionaries, each mapping k-mer hashes to their abundances in that part.

        **Usage Example**:

        ```python
        distributed = ReferenceQC.distribute_kmers_random(hash_to_abund, n=3)
        ```
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
        r"""
        Calculate cumulative coverage index vs cumulative sequencing depth.

        This method simulates incremental sequencing by splitting the sample signature into `n` parts and
        calculating the cumulative coverage index at each step. It helps in understanding how coverage
        improves with increased sequencing depth.

        **Mathematical Explanation**:

        For each cumulative part \( i \) (where \( 1 \leq i \leq n \)):

        - **Cumulative Sequencing Depth** (\( D_i \)):
          $$
          D_i = \sum_{j=1}^{i} a_j
          $$
          Where \( a_j \) is the total abundance of the \( j^{th} \) part.

        - **Cumulative Coverage Index** (\( C_i \)):
          $$
          C_i = \frac{\text{Number of genomic unique k-mers in first } i \text{ parts}}{\left| \text{Reference genome k-mer set} \right|}
          $$

        **Parameters**:

        - `n` (`int`): Number of parts to split the signature into.

        **Returns**:

        - `List[Dict[str, Any]]`:  
          List of dictionaries containing:
            - `"cumulative_parts"` (`int`): Number of parts included.
            - `"cumulative_total_abundance"` (`int`): Total sequencing depth up to this part.
            - `"cumulative_coverage_index"` (`float`): Coverage index up to this part.

        **Usage Example**:

        ```python
        coverage_depth_data = qc.calculate_coverage_vs_depth(n=10)
        for data in coverage_depth_data:
            print(data)
        ```
        """
        self.logger.debug("Calculating coverage vs depth with %d parts.", n)
        # Determine the ROI reference signature
        if isinstance(self.amplicon_sig, SnipeSig):
            roi_reference_sig = self.amplicon_sig
            self.logger.debug("Using amplicon signature as ROI reference.")
        else:
            roi_reference_sig = self.reference_sig
            self.logger.debug("Using reference genome signature as ROI reference.")

        # Split the sample signature into n random parts (cached if available)
        split_sigs = self.split_sig_randomly(n)

        coverage_depth_data = []

        if not split_sigs:
            self.logger.error("No split signatures available. Cannot calculate coverage vs depth.")
            return coverage_depth_data

        cumulative_snipe_sig = split_sigs[0].copy()
        cumulative_total_abundance = cumulative_snipe_sig.total_abundance

        # Force conversion to GENOME
        roi_reference_sig.sigtype = SigType.GENOME

        # Compute initial coverage index
        cumulative_qc = ReferenceQC(
            sample_sig=cumulative_snipe_sig,
            reference_sig=roi_reference_sig,
            enable_logging=self.enable_logging
        )
        cumulative_stats = cumulative_qc.get_aggregated_stats()
        cumulative_coverage_index = cumulative_stats.get("Genome coverage index", 0.0)

        coverage_depth_data.append({
            "cumulative_parts": 1,
            "cumulative_total_abundance": cumulative_total_abundance,
            "cumulative_coverage_index": cumulative_coverage_index,
        })

        self.logger.debug("Added initial coverage depth data for part 1.")

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
            cumulative_coverage_index = cumulative_stats.get("Genome coverage index", 0.0)

            coverage_depth_data.append({
                "cumulative_parts": i + 1,
                "cumulative_total_abundance": cumulative_total_abundance,
                "cumulative_coverage_index": cumulative_coverage_index,
            })

            self.logger.debug("Added coverage depth data for part %d.", i + 1)

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
        C(D) = \frac{a \cdot D}{b + D}
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
        C_{\text{pred}} = \frac{a \cdot D_{\text{pred}}}{b + D_{\text{pred}}}
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
        return float(predicted_coverage)

    def calculate_chromosome_metrics(self, chr_to_sig: Dict[str, SnipeSig]) -> Dict[str, Any]:
        r"""
        Calculate the coefficient of variation (CV) of mean abundances across autosomal chromosomes.

        This method computes the CV to assess the variability of mean abundances among autosomal chromosomes,
        excluding any sex chromosomes.

        **Mathematical Explanation**:

        The Coefficient of Variation (CV) is defined as:

        $$
        \text{CV} = \frac{\sigma}{\mu}
        $$

        Where:
        - \( \sigma \) is the standard deviation of the mean abundances across autosomal chromosomes.
        - \( \mu \) is the mean of the mean abundances across autosomal chromosomes.

        **Parameters**:

        - `chr_to_sig` (`Dict[str, SnipeSig]`):  
          A dictionary mapping chromosome names (e.g., `'autosomal-1'`, `'autosomal-2'`, `'sex-x'`, `'sex-y'`) to their corresponding
          `SnipeSig` instances. Each `SnipeSig` should represent the k-mer signature of a specific chromosome.

        **Returns**:

        - `Dict[str, Any]`:  
          A dictionary containing the computed metrics:
              - `"Autosomal_CV"` (`float`):  
                The coefficient of variation of mean abundances across autosomal chromosomes.

        **Raises**:

        - `ValueError`:  
          If `chr_to_sig` is empty or if there is an inconsistency in the signatures' parameters.

        **Usage Example**:

        ```python
        # Assume `chr_signatures` is a dictionary of chromosome-specific SnipeSig instances
        chr_signatures = {
            "1": sig_chr1,
            "2": sig_chr2,
            "X": sig_chrX,
            "Y": sig_chrY
        }

        # Calculate chromosome metrics
        metrics = qc.calculate_chromosome_metrics(chr_to_sig=chr_signatures)

        print(metrics)
        # Output:
        # {'Autosomal_CV': 0.15}
        ```

        **Notes**:

        - **Exclusion of Sex Chromosomes**:  
          Chromosomes with names containing the substring `"sex"` (e.g., `'sex-y'`, `'sex-x'`) are excluded from the CV calculation to focus solely on autosomal chromosomes.

        - **Mean Abundance Calculation**:  
          The mean abundance for each chromosome is calculated by intersecting the sample signature with the chromosome-specific signature and averaging the abundances of the shared k-mers.
        """
        
        # Implementation of the method
        # let's make sure all chromosome sigs are unique
        self.logger.debug("Computing specific chromosome hashes for %s.", ','.join(chr_to_sig.keys()))
        self.logger.debug(f"\t-All hashes for chromosomes before getting unique sigs {len(SnipeSig.sum_signatures(list(chr_to_sig.values())))}")
        specific_chr_to_sig = SnipeSig.get_unique_signatures(chr_to_sig)
        self.logger.debug(f"\t-All hashes for chromosomes after getting unique sigs {len(SnipeSig.sum_signatures(list(specific_chr_to_sig.values())))}")
        
        # calculate mean abundance for each chromosome and loaded sample sig
        chr_to_mean_abundance = {}
        self.logger.debug("Calculating mean abundance for each chromosome.")
        for chr_name, chr_sig in specific_chr_to_sig.items():
            self.logger.debug("Intersecting %s (%d) with %s (%d)", self.sample_sig.name, len(self.sample_sig), chr_name, len(chr_sig))
            chr_sample_sig = self.sample_sig & chr_sig
            chr_stats = chr_sample_sig.get_sample_stats
            chr_to_mean_abundance[chr_name] = chr_stats["mean_abundance"]
            self.logger.debug("\t-Mean abundance for %s: %f", chr_name, chr_stats["mean_abundance"])
        
        # chromosomes are numberd from 1 to ..., sort them by numer (might be string for sex chromosomes) then prefix them with chr-
        def sort_chromosomes(chr_name):
            try:
                # Try to convert to integer for numeric chromosomes
                return (0, int(chr_name))
            except ValueError:
                # Non-numeric chromosomes (like 'x', 'y', 'z', etc.)
                return (1, chr_name)

        # Create a new dictionary with sorted chromosome names and prefixed with 'chr-'
        sorted_chr_to_mean_abundance = {
            f"chr-{chr_name}": chr_to_mean_abundance[chr_name]
            for chr_name in sorted(chr_to_mean_abundance, key=sort_chromosomes)
        }
        
        self.chrs_stats.update(sorted_chr_to_mean_abundance)

        # chr_to_mean_abundance but without any chr with partial name sex
        autosomal_chr_to_mean_abundance = {}
        for chr_name, mean_abundance in chr_to_mean_abundance.items():
            if "sex" in chr_name.lower():
                continue
            autosomal_chr_to_mean_abundance[chr_name] = mean_abundance
        
        
        # calculate the CV for the whole sample
        if autosomal_chr_to_mean_abundance:
            mean_abundances = np.array(list(autosomal_chr_to_mean_abundance.values()), dtype=float)
            cv = np.std(mean_abundances) / np.mean(mean_abundances) if np.mean(mean_abundances) != 0 else 0.0
            self.chrs_stats.update({"Autosomal_CV": cv})
            self.logger.debug("Calculated Autosomal CV: %f", cv)
        else:
            self.logger.warning("No autosomal chromosomes were processed. 'Autosomal_CV' set to None.")
            self.chrs_stats.update({"Autosomal_CV": None})
        
        # optional return, not required
        return self.chrs_stats
    
    
    def calculate_sex_chrs_metrics(self, genome_and_chr_to_sig: Dict[str, SnipeSig]) -> Dict[str, Any]:
        r"""
        Calculate sex chromosome-related metrics based on genome and chromosome-specific signatures.

        This method processes a collection of genome and chromosome-specific `SnipeSig` instances to compute
        metrics such as the X-Ploidy score and Y-Coverage. It ensures that each chromosome signature contains
        only unique hashes that do not overlap with hashes from other chromosomes or the autosomal genome.
        The method excludes sex chromosomes (e.g., Y chromosome) from the autosomal genome signature to
        accurately assess sex chromosome metrics.

        **Mathematical Explanation**:

        - **X-Ploidy Score**:
        
          The X-Ploidy score is calculated using the formula:

          $$
          \text{X-Ploidy} = \left(\frac{\mu_X}{\mu_{\text{autosomal}}}\right) \times \left(\frac{N_{\text{autosomal}}}{N_X}\right)
          $$

          Where:
          - \( \mu_X \) is the mean abundance of X chromosome-specific k-mers in the sample.
          - \( \mu_{\text{autosomal}} \) is the mean abundance of autosomal k-mers in the sample.
          - \( N_{\text{autosomal}} \) is the number of autosomal k-mers in the reference genome.
          - \( N_X \) is the number of X chromosome-specific k-mers in the reference genome.

        - **Y-Coverage**:

          The Y-Coverage is calculated using the formula:

          $$
          \text{Y-Coverage} = \frac{\left(\frac{N_Y^{\text{sample}}}{N_Y}\right)}{\left(\frac{N_{\text{autosomal}}^{\text{sample}}}{N_{\text{autosomal}}}\right)}
          $$

          Where:
          - \( N_Y^{\text{sample}} \) is the number of Y chromosome-specific k-mers in the sample.
          - \( N_Y \) is the number of Y chromosome-specific k-mers in the reference genome.
          - \( N_{\text{autosomal}}^{\text{sample}} \) is the number of autosomal k-mers in the sample.
          - \( N_{\text{autosomal}} \) is the number of autosomal k-mers in the reference genome.

        **Parameters**:

            - `genome_and_chr_to_sig` (`Dict[str, SnipeSig]`):  
              A dictionary mapping signature names to their corresponding `SnipeSig` instances. This should include
              the autosomal genome signature (with a name ending in `'-snipegenome'`) and chromosome-specific
              signatures (e.g., `'sex-x'`, `'sex-y'`, `'autosome-1'`, `'autosome-2'`, etc.).

        **Returns**:

            - `Dict[str, Any]`:  
              A dictionary containing the calculated sex-related metrics:
                  - `"X-Ploidy score"` (`float`):  
                    The ploidy score of the X chromosome, reflecting the ratio of X chromosome k-mer abundance
                    to autosomal k-mer abundance, adjusted by genome and X chromosome sizes.
                  - `"Y-Coverage"` (`float`, optional):  
                    The coverage of Y chromosome-specific k-mers in the sample relative to autosomal coverage.
                    This key is present only if a Y chromosome signature is provided.

        **Raises**:

            - `ValueError`:  
              - If the `'sex-x'` chromosome signature is not found in `genome_and_chr_to_sig`.
              - If the autosomal genome signature is not found or improperly labeled.

        **Usage Example**:

        ```python
        # Assume `genome_and_chr_signatures` is a dictionary of genome and chromosome-specific SnipeSig instances
        genome_and_chr_signatures = {
            "autosomal-snipegenome": sig_autosomal_genome,
            "1": sig_chr1,
            "2": sig_chr2,
            "sex-x": sig_sex_x,
            "sex-y": sig_sex_y
        }

        # Calculate sex chromosome metrics
        metrics = qc.calculate_sex_chrs_metrics(genome_and_chr_to_sig=genome_and_chr_signatures)

        print(metrics)
        # Output Example:
        # {
        #     "X-Ploidy score": 2.6667,
        #     "Y-Coverage": 0.0
        # }
        ```

        **Notes**:

            - **Signature Naming Convention**:  
              The autosomal genome signature must have a name ending with `'-snipegenome'`. Chromosome-specific
              signatures should be named accordingly (e.g., `'sex-x'`, `'sex-y'`, `'autosomal-1'`, `'autosomal-2'`, etc.).

            - **Exclusion of Sex Chromosomes from Autosomal Genome**:  
              The Y chromosome signature (`'sex-y'`) is subtracted from the autosomal genome signature to ensure
              that Y chromosome k-mers are not counted towards autosomal metrics.

            - **Robustness**:  
              The method includes comprehensive logging for debugging purposes, tracking each major step and
              any exclusions made during processing.
        """
        
        # Ensure that the chromosome X signature exists
        if 'sex-x' not in genome_and_chr_to_sig:
            self.logger.warning("Chromosome X ('sex-x') not found in the provided signatures. X-Ploidy score will be set to zero.")
            # set sex-x to an empty signature
            genome_and_chr_to_sig['sex-x'] = SnipeSig.create_from_hashes_abundances(
                hashes=np.array([], dtype=np.uint64),
                abundances=np.array([], dtype=np.uint32),
                ksize=genome_and_chr_to_sig[list(genome_and_chr_to_sig.keys())[0]].ksize,
                scale=genome_and_chr_to_sig[list(genome_and_chr_to_sig.keys())[0]].scale,
            )
        
        # Separate the autosomal genome signature from chromosome-specific signatures
        chr_to_sig: Dict[str, SnipeSig] = {}
        autosomals_genome_sig: Optional[SnipeSig] = None
        self.logger.debug("Separating autosomal genome signature from chromosome-specific signatures.")
        
        for name, sig in genome_and_chr_to_sig.items():
            if name.endswith('-snipegenome'):
                self.logger.debug("\t- Identified autosomal genome signature: '%s'.", name)
                autosomals_genome_sig = sig
            else:
                chr_to_sig[name] = sig
        
        if autosomals_genome_sig is None:
            self.logger.error("Autosomal genome signature (ending with '-snipegenome') not found.")
            raise ValueError("Autosomal genome signature (ending with '-snipegenome') not found.")
        
        # Ensure all chromosome signatures have unique hashes
        specific_chr_to_sig = SnipeSig.get_unique_signatures(chr_to_sig)
        
        # Exclude Y chromosome from the autosomal genome signature if present
        if 'sex-y' in chr_to_sig:
            self.logger.debug("Y chromosome ('sex-y') detected. Removing its hashes from the autosomal genome signature.")
            self.logger.debug("\t- Original autosomal genome size: %d hashes.", len(autosomals_genome_sig))
            autosomals_genome_sig = autosomals_genome_sig - chr_to_sig['sex-y']
            self.logger.debug("\t- Updated autosomal genome size after removing Y chromosome: %d hashes.", len(autosomals_genome_sig))
        
        # Remove X chromosome hashes from the autosomal genome signature
        self.logger.debug("Removing X chromosome ('sex-x') hashes from the autosomal genome signature.")
        autosomals_genome_sig = autosomals_genome_sig - chr_to_sig['sex-x']
        self.logger.debug("\t- Updated autosomal genome size after removing X chromosome: %d hashes.", len(autosomals_genome_sig))
        
        # Derive the X chromosome-specific signature by subtracting autosomal genome hashes
        specific_xchr_sig = specific_chr_to_sig["sex-x"] - autosomals_genome_sig
        self.logger.debug("\t-Derived X chromosome-specific signature size: %d hashes.", len(specific_xchr_sig))
        
        # Intersect the sample signature with chromosome-specific signatures
        sample_specific_xchr_sig = self.sample_sig & specific_xchr_sig
        if len(sample_specific_xchr_sig) == 0:
            self.logger.warning("No X chromosome-specific k-mers found in the sample signature.")
        self.logger.debug("\t-Intersected sample signature with X chromosome-specific k-mers = %d hashes.", len(sample_specific_xchr_sig))
        sample_autosomal_sig = self.sample_sig & autosomals_genome_sig
        self.logger.debug("\t-Intersected sample signature with autosomal genome k-mers = %d hashes.", len(sample_autosomal_sig))
        
        # Retrieve mean abundances
        xchr_mean_abundance = sample_specific_xchr_sig.get_sample_stats.get("mean_abundance", 0.0)
        autosomal_mean_abundance = sample_autosomal_sig.get_sample_stats.get("mean_abundance", 0.0)
        
        # Calculate X-Ploidy score
        if autosomal_mean_abundance == 0:
            self.logger.warning("Autosomal mean abundance is zero. Setting X-Ploidy score to zero to avoid division by zero.")
            xploidy_score = 0.0
        else:
            xploidy_score = (xchr_mean_abundance / autosomal_mean_abundance) * \
                            (len(autosomals_genome_sig) / len(specific_xchr_sig) if len(specific_xchr_sig) > 0 else 0.0)
        
        self.logger.debug("Calculated X-Ploidy score: %.4f", xploidy_score)
        self.sex_stats.update({"X-Ploidy score": xploidy_score})
        
        # Calculate Y-Coverage if Y chromosome is present
        if 'sex-y' in specific_chr_to_sig:
            self.logger.debug("Calculating Y-Coverage based on Y chromosome-specific k-mers.")
            
            # Derive Y chromosome-specific k-mers by excluding autosomal and X chromosome k-mers
            ychr_specific_kmers = chr_to_sig["sex-y"] - autosomals_genome_sig - specific_xchr_sig
            self.logger.debug("\t-Derived Y chromosome-specific signature size: %d hashes.", len(ychr_specific_kmers))
            
            # Intersect Y chromosome-specific k-mers with the sample signature
            ychr_in_sample = self.sample_sig & ychr_specific_kmers
            self.logger.debug("\t-Intersected sample signature with Y chromosome-specific k-mers = %d hashes.", len(ychr_in_sample))
            if len(ychr_in_sample) == 0:
                self.logger.warning("No Y chromosome-specific k-mers found in the sample signature.")
            
            # Derive autosomal-specific k-mers by excluding X and Y chromosome k-mers from the reference signature
            autosomals_specific_kmers = self.reference_sig - specific_chr_to_sig["sex-x"] - specific_chr_to_sig['sex-y']
            
            # Calculate Y-Coverage metric
            if len(ychr_specific_kmers) == 0 or len(autosomals_specific_kmers) == 0:
                self.logger.warning("Insufficient k-mers for Y-Coverage calculation. Setting Y-Coverage to zero.")
                ycoverage = 0.0
            else:
                ycoverage = (len(ychr_in_sample) / len(ychr_specific_kmers)) / \
                        (len(sample_autosomal_sig) / len(autosomals_specific_kmers))
            
            self.logger.debug("Calculated Y-Coverage: %.4f", ycoverage)
            self.sex_stats.update({"Y-Coverage": ycoverage})

        return self.sex_stats
        
        
        
    def nonref_consume_from_vars(self, *, vars: Dict[str, SnipeSig], vars_order: List[str], **kwargs) -> Dict[str, float]:
        r"""
        Consume and analyze non-reference k-mers from provided variable signatures.

        This method processes non-reference k-mers in the sample signature by intersecting them with a set of
        variable-specific `SnipeSig` instances. It calculates coverage and total abundance metrics for each
        variable in a specified order, ensuring that each non-reference k-mer is accounted for without overlap
        between variables. The method updates internal statistics that reflect the distribution of non-reference
        k-mers across the provided variables.

        **Process Overview**:

        1. **Validation**:
        - Verifies that all variable names specified in `vars_order` are present in the `vars` dictionary.
        - Raises a `ValueError` if any variable in `vars_order` is missing from `vars`.

        2. **Non-Reference K-mer Extraction**:
        - Computes the set of non-reference non-singleton k-mers by subtracting the reference signature from the sample signature.
        - If no non-reference k-mers are found, the method logs a warning and returns an empty dictionary.

        3. **Variable-wise Consumption**:
        - Iterates over each variable name in `vars_order`.
        - For each variable:
            - Intersects the remaining non-reference k-mers with the variable-specific signature.
            - Calculates the total abundance and coverage index for the intersected k-mers.
            - Updates the `vars_nonref_stats` dictionary with the computed metrics.
            - Removes the consumed k-mers from the remaining non-reference set to prevent overlap.

        4. **Final State Logging**:
        - Logs the final size and total abundance of the remaining non-reference k-mers after consumption.

        **Parameters**:

            - `vars` (`Dict[str, SnipeSig]`):  
            A dictionary mapping variable names to their corresponding `SnipeSig` instances. Each `SnipeSig` 
            represents a set of k-mers associated with a specific non-reference category or variable.

            - `vars_order` (`List[str]`):  
            A list specifying the order in which variables should be processed. The order determines the priority 
            of consumption, ensuring that earlier variables in the list have their k-mers accounted for before 
            later ones.

            - `**kwargs`:  
            Additional keyword arguments. Reserved for future extensions and should not be used in the current context.

        **Returns**:

            - `Dict[str, float]`:  
            A dictionary containing statistics for each variable name in `vars_order`, 
                - `"non-genomic total k-mer abundance"` (`float`):  
                    The sum of abundances of non-reference k-mers associated with the variable.
                - `"non-genomic coverage index"` (`float`):  
                    The ratio of unique non-reference k-mers associated with the variable to the total number 
                    of non-reference k-mers in the sample before consumption.

            Example Output:
            ```python
            {
                "variable_A non-genomic total k-mer abundance": 1500.0,
                "variable_A non-genomic coverage index": 0.20
                "variable_B non-genomic total k-mer abundance": 3500.0,
                "variable_B non-genomic coverage index": 0.70
                "non-var non-genomic total k-mer abundance": 0.10,
                "non-var non-genomic coverage index": 218
            }
            ```

        **Raises**:

            - `ValueError`:  
            - If any variable specified in `vars_order` is not present in the `vars` dictionary.
            - This ensures that all variables intended for consumption are available for processing.

        **Usage Example**:

        ```python
        # Assume `variables_signatures` is a dictionary of variable-specific SnipeSig instances
        variables_signatures = {
            "GTDB": sig_GTDB,
            "VIRALDB": sig_VIRALDB,
            "contaminant_X": sig_contaminant_x
        }

        # Define the order in which variables should be processed
        processing_order = ["GTDB", "VIRALDB", "contaminant_X"]

        # Consume non-reference k-mers and retrieve statistics
        nonref_stats = qc.nonref_consume_from_vars(vars=variables_signatures, vars_order=processing_order)

        print(nonref_stats)
        # Output Example:
        # {
        #     "GTDB non-genomic total k-mer abundance": 1500.0,
        #     "GTDB non-genomic coverage index": 0.2,
        #     "VIRALDB non-genomic total k-mer abundance": 3500.0,
        #     "VIRALDB non-genomic coverage index": 0.70,
        #     "contaminant_X non-genomic total k-mer abundance": 0.0,
        #     "contaminant_X non-genomic coverage index": 0.0,
        #     "non-var non-genomic total k-mer abundance": 100.0,
        #     "non-var non-genomic coverage index": 0.1
        # }
        ```

        **Notes**:

            - **Variable Processing Order**:  
            The `vars_order` list determines the sequence in which variables are processed. This order is crucial
            when there is potential overlap in k-mers between variables, as earlier variables in the list have 
            higher priority in consuming shared k-mers.

            - **Non-Reference K-mers Definition**:  
            Non-reference k-mers are defined as those present in the sample signature but absent in the reference 
            signature. This method focuses on characterizing these unique k-mers relative to provided variables.
        """
        
        # check the all vars in vars_order are in vars
        if not all([var in vars for var in vars_order]):
            # report dict keys, and the vars order
            self.logger.debug("Provided vars_order: %s, and vars keys: %s", vars_order, list(vars.keys()))
            self.logger.error("All variables in vars_order must be present in vars.")
            raise ValueError("All variables in vars_order must be present in vars.")
        
        self.logger.debug("Consuming non-reference k-mers from provided variables.")
        self.logger.debug("\t-Current size of the sample signature: %d hashes.", len(self.sample_sig))
        
        sample_nonref = self.sample_sig - self.reference_sig

        sample_nonref.trim_singletons()
        
        sample_nonref_unique_hashes = len(sample_nonref)
        
        self.logger.debug("\t-Size of non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
        if len(sample_nonref) == 0:
            self.logger.warning("No non-reference k-mers found in the sample signature.")
            return {}
        
        # intersect and report coverage and depth, then subtract from sample_nonref so sum will be 100%
        for var_name in vars_order:
            sample_nonref_var: SnipeSig = sample_nonref & vars[var_name]
            sample_nonref_var_total_abundance = sample_nonref_var.total_abundance
            sample_nonref_var_unique_hashes = len(sample_nonref_var)
            sample_nonref_var_coverage_index = sample_nonref_var_unique_hashes / sample_nonref_unique_hashes
            self.vars_nonref_stats.update({
                f"{var_name} non-genomic total k-mer abundance": sample_nonref_var_total_abundance,
                f"{var_name} non-genomic coverage index": sample_nonref_var_coverage_index
            })
            
            self.logger.debug("\t-Consuming non-reference k-mers from variable '%s'.", var_name)
            sample_nonref -= sample_nonref_var
            self.logger.debug("\t-Size of remaining non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
            
        self.vars_nonref_stats["non-var non-genomic total k-mer abundance"] = sample_nonref.total_abundance
        self.vars_nonref_stats["non-var non-genomic coverage index"] = len(sample_nonref) / sample_nonref_unique_hashes if sample_nonref_unique_hashes > 0 else 0.0
        
        self.logger.debug(
            "After consuming all vars from the non reference k-mers, the size of the sample signature is: %d hashes, "
            "with total abundance of %s.", 
            len(sample_nonref), sample_nonref.total_abundance
        )
        
        return self.vars_nonref_stats
    
    def load_genome_sig_to_dict(self, *, zip_file_path: str, **kwargs) -> Dict[str, 'SnipeSig']:
        """
        Load a genome signature into a dictionary of SnipeSig instances.
        
        Parameters:
            zip_file_path (str): Path to the zip file containing the genome signatures.
            **kwargs: Additional keyword arguments to pass to the SnipeSig constructor.
            
        Returns:
            Dict[str, SnipeSig]: A dictionary mapping genome names to SnipeSig instances.
        """
        
        genome_chr_name_to_sig = {}
        
        sourmash_sigs: List[sourmash.signature.SourmashSignature] = sourmash.load_file_as_signatures(zip_file_path)
        sex_count = 0
        autosome_count = 0
        genome_count = 0
        for sig in sourmash_sigs:
            name = sig.name
            if name.endswith("-snipegenome"):
                self.logger.debug(f"Loading genome signature: {name}")
                restored_name = name.replace("-snipegenome", "")
                genome_chr_name_to_sig[restored_name] = SnipeSig(sourmash_sig=sig, sig_type=SigType.GENOME)
                genome_count += 1
            elif "sex" in name:
                sex_count += 1
                genome_chr_name_to_sig[name.replace('sex-','')] = SnipeSig(sourmash_sig=sig, sig_type=SigType.GENOME)
            elif "autosome" in name:
                autosome_count += 1
                genome_chr_name_to_sig[name.replace('autosome-','')] = SnipeSig(sourmash_sig=sig, sig_type=SigType.GENOME)
            else:
                logging.warning(f"Unknown genome signature name: {name}, are you sure you generated this with `snipe sketch --ref`?")
                
        self.logger.debug("Loaded %d genome signatures and %d sex chrs and %d autosome chrs", genome_count, sex_count, autosome_count)
                
        if genome_count != 1:
            logging.error(f"Expected 1 genome signature, found {genome_count}")
        
            
        return genome_chr_name_to_sig
