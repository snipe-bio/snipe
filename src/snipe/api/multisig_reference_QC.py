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
from snipe.api.reference_QC import ReferenceQC
import concurrent

# pylint disable C0301

class MultiSigReferenceQC:
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
                 reference_sig: SnipeSig,
                 amplicon_sig: Optional[SnipeSig] = None,
                 ychr: Optional[SnipeSig] = None,
                 varsigs: Optional[List[SnipeSig]] = None,
                 enable_logging: bool = False,
                 export_varsigs: bool = False,
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
            
            
        if reference_sig.sigtype != SigType.GENOME:
            self.logger.error("Invalid signature type for reference_sig: %s", reference_sig.sigtype)
            raise ValueError(f"reference_sig must be of type {SigType.GENOME}, got {reference_sig.sigtype}")

        if amplicon_sig is not None and amplicon_sig.sigtype != SigType.AMPLICON:
            self.logger.error("Invalid signature type for amplicon_sig: %s", amplicon_sig.sigtype)
            raise ValueError(f"amplicon_sig must be of type {SigType.AMPLICON}, got {amplicon_sig.sigtype}")
        
        if ychr and not isinstance(ychr, SnipeSig):
            self.logger.error("Invalid signature type for ychr: %s", ychr.sigtype)
            raise ValueError(f"ychr must be of type {SigType.SAMPLE}, got {ychr.sigtype}")
        
        self.specific_chr_to_sig: Optional[Dict[str, SnipeSig]] = reference_sig.chr_to_sig
        
        if ychr is not None and self.specific_chr_to_sig is not None:
            self.logger.debug("Y chromosome signature provided and passed to the specific_kmers function.")
            self.specific_chr_to_sig['sex-y'] = ychr
        
        if self.specific_chr_to_sig is not None:
            self.logger.debug("Computing specific chromosome hashes for %s.", ','.join(self.specific_chr_to_sig.keys()))
            self.logger.debug(f"\t-All hashes for chromosomes before getting unique sigs {len(SnipeSig.sum_signatures(list(self.specific_chr_to_sig.values())))}")
            self.specific_chr_to_sig = SnipeSig.get_unique_signatures({sig_name: sig for sig_name, sig in self.specific_chr_to_sig.items() if not sig_name.endswith("-snipegenome")})
            self.logger.debug(f"\t-All hashes for chromosomes after getting unique sigs {len(SnipeSig.sum_signatures(list(self.specific_chr_to_sig.values())))}")
        
        # now remove the mitochondrial if present
        # if "mitochondrial-M" in self.specific_chr_to_sig:
        #     self.specific_chr_to_sig.pop("mitochondrial-M")
        #     self.logger.debug("Removed mitochondrial-M from specific_chr_to_sig.")
        #     self.logger.debug(f"\t-All hashes for chromosomes after removing mitochondrial-M {len(SnipeSig.sum_signatures(list(self.specific_chr_to_sig.values())))}")
        
        self.variance_sigs: Optional[List[SnipeSig]] = None
        if varsigs is not None:
            self.logger.debug("Variance signatures provided.")
            # make sure they are same ksize and scale as reference_sig
            for sig in varsigs:
                if sig.ksize != reference_sig.ksize:
                    self.logger.error("K-mer sizes do not match: varsigs.ksize=%d vs reference_sig.ksize=%d",
                                      sig.ksize, reference_sig.ksize)
                    raise ValueError(f"varsigs ksize ({sig.ksize}) does not match reference_sig ksize ({reference_sig.ksize}).")
                if sig.scale != reference_sig.scale:
                    self.logger.error("Scale values do not match: varsigs.scale=%d vs reference_sig.scale=%d",
                                      sig.scale, reference_sig.scale)
                    raise ValueError(f"varsigs scale ({sig.scale}) does not match reference_sig scale ({reference_sig.scale}).")
            self.variance_sigs = varsigs

        self.logger.debug("Chromosome specific signatures provided.")
        self.flag_activate_sex_metrics = True


        self.reference_sig = reference_sig
        self.amplicon_sig = amplicon_sig
        self.enable_logging = enable_logging
        self.export_varsigs = export_varsigs
        self.sample_to_stats = {}


        # Set grey zone thresholds
        self.relative_total_abundance_grey_zone = [0.08092723407173719, 0.11884490500267662]


    def process_sample(self, sample_sig: SnipeSig, predict_extra_folds: Optional[List[int]] = None, advanced: Optional[bool] = False) -> Dict[str, Any]:
        
        # ============= Attributes =============
        
        # Initialize attributes
        sample_stats: Dict[str, Any] = {}
        genome_stats: Dict[str, Any] = {}
        amplicon_stats: Dict[str, Any] = {}
        advanced_stats: Dict[str, Any] = {}
        chrs_stats: Dict[str, Dict[str, Any]] = {}
        sex_stats: Dict[str, Any] = {}
        predicted_error_contamination_index: Dict[str, Any] = {}
        vars_nonref_stats: Dict[str, Any] = {}
        chr_to_mean_abundance: Dict[str, np.float64] = {}
        predicted_assay_type: str = "WGS"
        roi_stats: Dict[str, Any] = {}
        
        
        # ============= SAMPLE Verification =============
        
        
        self.logger.debug("Validating ksize and scale across signatures.")
        if sample_sig.ksize != self.reference_sig.ksize:
            self.logger.error("K-mer sizes do not match: sample_sig.ksize=%d vs reference_sig.ksize=%d",
                              sample_sig.ksize, self.reference_sig.ksize)
            raise ValueError(f"sample_sig kszie ({sample_sig.ksize}) does not match reference_sig ksize ({self.reference_sig.ksize}).")
        if sample_sig.scale != self.reference_sig.scale:
            self.logger.error("Scale values do not match: sample_sig.scale=%d vs reference_sig.scale=%d",
                              sample_sig.scale, self.reference_sig.scale)
            raise ValueError(f"sample_sig scale ({sample_sig.scale}) does not match reference_sig scale ({self.reference_sig.scale}).")
        
        if self.amplicon_sig is not None:
            if self.amplicon_sig.ksize != sample_sig.ksize:
                self.logger.error("K-mer sizes do not match: amplicon_sig.ksize=%d vs sample_sig.ksize=%d",
                                  self.amplicon_sig.ksize, sample_sig.ksize)
                raise ValueError(f"amplicon_sig ksize ({self.amplicon_sig.ksize}) does not match sample_sig ksize ({sample_sig.ksize}).")
            if self.amplicon_sig.scale != sample_sig.scale:
                self.logger.error("Scale values do not match: amplicon_sig.scale=%d vs sample_sig.scale=%d",
                                  self.amplicon_sig.scale, sample_sig.scale)
                raise ValueError(f"amplicon_sig scale ({self.amplicon_sig.scale}) does not match sample_sig scale ({sample_sig.scale}).")

        self.logger.debug("All signatures have matching ksize and scale.")
            
        # Verify signature types
        if sample_sig._type != SigType.SAMPLE:
            self.logger.error("Invalid signature type for sample_sig: %s | %s", sample_sig.sigtype, sample_sig._type)
            raise ValueError(f"sample_sig must be of type {SigType.SAMPLE}, got {sample_sig.sigtype}")

        
        # ============= SAMPLE STATS =============
        
        self.logger.debug("Processing sample statistics.")
        sample_stats_raw = sample_sig.get_sample_stats
        sample_stats.update({
            "name": sample_stats_raw["name"],
            "ksize": sample_stats_raw["ksize"],
            "scale": sample_stats_raw["scale"],
            "filename": sample_stats_raw["filename"],
            "Total unique k-mers": sample_stats_raw["num_hashes"],
            "k-mer total abundance": sample_stats_raw["total_abundance"],
            "k-mer mean abundance": sample_stats_raw["mean_abundance"],
            "k-mer median abundance": sample_stats_raw["median_abundance"],
            "num_singletons": sample_stats_raw["num_singletons"],
        })

        # ============= GENOME STATS =============
        
        self.logger.debug("Calculating genome statistics.")
        # Compute intersection of sample and reference genome
        self.logger.debug("Type of sample_sig: %s | Type of reference_sig: %s", sample_sig.sigtype, self.reference_sig.sigtype)
        sample_genome = sample_sig & self.reference_sig
        # Get stats (call get_sample_stats only once)

        # Log hashes and abundances for both sample and reference
        # self.logger.debug("Sample hashes: %s", self.sample_sig.hashes)
        # self.logger.debug("Sample abundances: %s", self.sample_sig.abundances)
        # self.logger.debug("Reference hashes: %s", self.reference_sig.hashes)
        # self.logger.debug("Reference abundances: %s", self.reference_sig.abundances)

        sample_genome_stats = sample_genome.get_sample_stats

        genome_stats.update({
            "Genomic unique k-mers": sample_genome_stats["num_hashes"],
            "Genomic k-mers total abundance": sample_genome_stats["total_abundance"],
            "Genomic k-mers mean abundance": sample_genome_stats["mean_abundance"],
            "Genomic k-mers median abundance": sample_genome_stats["median_abundance"],
            # Genome coverage index
            "Genome coverage index": (
                sample_genome_stats["num_hashes"] / len(self.reference_sig)
                if len(self.reference_sig) > 0 else 0
            ),
            "Mapping index": (
                sample_genome_stats["total_abundance"] / sample_stats["k-mer total abundance"]
                if sample_stats["k-mer total abundance"] > 0 else 0
            ),
        })

        # ============= AMPLICON STATS =============
        if self.amplicon_sig is not None:
            self.logger.debug("Calculating amplicon statistics.")
            sample_amplicon = sample_sig & self.amplicon_sig
            sample_amplicon_stats = sample_amplicon.get_sample_stats

            amplicon_stats.update({
                "Amplicon unique k-mers": sample_amplicon_stats["num_hashes"],
                "Amplicon k-mers total abundance": sample_amplicon_stats["total_abundance"],
                "Amplicon k-mers mean abundance": sample_amplicon_stats["mean_abundance"],
                "Amplicon k-mers median abundance": sample_amplicon_stats["median_abundance"],
                "Amplicon coverage index": (
                    sample_amplicon_stats["num_hashes"] / len(self.amplicon_sig)
                    if len(self.amplicon_sig) > 0 else 0
                ),
            })

            # ============= RELATIVE STATS =============
            amplicon_stats["Relative total abundance"] = (
                amplicon_stats["Amplicon k-mers total abundance"] / genome_stats["Genomic k-mers total abundance"]
                if genome_stats["Genomic k-mers total abundance"] > 0 else 0
            )
            amplicon_stats["Relative coverage"] = (
                amplicon_stats["Amplicon coverage index"] / genome_stats["Genome coverage index"]
                if genome_stats["Genome coverage index"] > 0 else 0
            )

            relative_total_abundance = amplicon_stats["Relative total abundance"]
            if relative_total_abundance <= self.relative_total_abundance_grey_zone[0]:
                predicted_assay_type = "WGS"
            elif relative_total_abundance >= self.relative_total_abundance_grey_zone[1]:
                predicted_assay_type = "WXS"
            else:
                # Assign based on the closest threshold
                distance_to_wgs = abs(relative_total_abundance - self.relative_total_abundance_grey_zone[0])
                distance_to_wxs = abs(relative_total_abundance - self.relative_total_abundance_grey_zone[1])
                predicted_assay_type = "WGS" if distance_to_wgs < distance_to_wxs else "WXS"
            
            self.logger.debug("Predicted assay type: %s", predicted_assay_type)
        
        else:
            self.logger.debug("No amplicon signature provided.")
        
        # ============= Contaminatino/Error STATS =============
        
        self.logger.debug("Calculuating error and contamination indices.")
        try:
            sample_nonref = sample_sig - self.reference_sig
            sample_nonref_singletons = sample_nonref.count_singletons()
            sample_nonref_non_singletons = sample_nonref.total_abundance - sample_nonref_singletons
            sample_total_abundance = sample_sig.total_abundance
            
            predicted_error_index = sample_nonref_singletons / sample_total_abundance
            predicted_contamination_index = sample_nonref_non_singletons / sample_total_abundance

            # predict error and contamination index
            predicted_error_contamination_index["Predicted contamination index"] = predicted_contamination_index
            predicted_error_contamination_index["Sequencing errors index"] = predicted_error_index
        # except zero division error
        except ZeroDivisionError:
            self.logger.error("Please check the sample signature, it seems to be empty.")
            
        
        # ============= Advanced Stats if needed =============
        if advanced:
            # Copy sample signature to avoid modifying the original
            median_trimmed_sample_sig = sample_sig.copy()
            # Trim below median
            median_trimmed_sample_sig.trim_below_median()
            # Get stats
            median_trimmed_sample_stats = median_trimmed_sample_sig.get_sample_stats
            advanced_stats.update({
                "Median-trimmed unique k-mers": median_trimmed_sample_stats["num_hashes"],
                "Median-trimmed total abundance": median_trimmed_sample_stats["total_abundance"],
                "Median-trimmed mean abundance": median_trimmed_sample_stats["mean_abundance"],
                "Median-trimmed median abundance": median_trimmed_sample_stats["median_abundance"],
            })

            # Genome stats for median-trimmed sample
            median_trimmed_sample_genome = median_trimmed_sample_sig & self.reference_sig
            median_trimmed_sample_genome_stats = median_trimmed_sample_genome.get_sample_stats
            advanced_stats.update({
                "Median-trimmed Genomic unique k-mers": median_trimmed_sample_genome_stats["num_hashes"],
                "Median-trimmed Genomic total abundance": median_trimmed_sample_genome_stats["total_abundance"],
                "Median-trimmed Genomic mean abundance": median_trimmed_sample_genome_stats["mean_abundance"],
                "Median-trimmed Genomic median abundance": median_trimmed_sample_genome_stats["median_abundance"],
                "Median-trimmed Genome coverage index": (
                    median_trimmed_sample_genome_stats["num_hashes"] / len(self.reference_sig)
                    if len(self.reference_sig) > 0 else 0
                ),
            })

            if self.amplicon_sig is not None:
                self.logger.debug("Calculating advanced amplicon statistics.")
                # Amplicon stats for median-trimmed sample
                median_trimmed_sample_amplicon = median_trimmed_sample_sig & self.amplicon_sig
                median_trimmed_sample_amplicon_stats = median_trimmed_sample_amplicon.get_sample_stats
                advanced_stats.update({
                    "Median-trimmed Amplicon unique k-mers": median_trimmed_sample_amplicon_stats["num_hashes"],
                    "Median-trimmed Amplicon total abundance": median_trimmed_sample_amplicon_stats["total_abundance"],
                    "Median-trimmed Amplicon mean abundance": median_trimmed_sample_amplicon_stats["mean_abundance"],
                    "Median-trimmed Amplicon median abundance": median_trimmed_sample_amplicon_stats["median_abundance"],
                    "Median-trimmed Amplicon coverage index": (
                        median_trimmed_sample_amplicon_stats["num_hashes"] / len(self.amplicon_sig)
                        if len(self.amplicon_sig) > 0 else 0
                    ),
                })
                # Additional advanced relative metrics
                self.logger.debug("Calculating advanced relative metrics.")
                amplicon_stats["Median-trimmed relative coverage"] = (
                    advanced_stats["Median-trimmed Amplicon coverage index"] / advanced_stats["Median-trimmed Genome coverage index"]
                    if advanced_stats["Median-trimmed Genome coverage index"] > 0 else 0
                )
                amplicon_stats["Median-trimmed relative mean abundance"] = (
                    advanced_stats["Median-trimmed Amplicon mean abundance"] / advanced_stats["Median-trimmed Genomic mean abundance"]
                    if advanced_stats["Median-trimmed Genomic mean abundance"] > 0 else 0
                )
                # Update amplicon_stats with advanced metrics
                amplicon_stats.update({
                    "Median-trimmed relative coverage": amplicon_stats["Median-trimmed relative coverage"],
                    "Median-trimmed relative mean abundance": amplicon_stats["Median-trimmed relative mean abundance"],
                })

                advanced_stats.update(amplicon_stats)
                
        # ============= CHR STATS =============
        
        if self.specific_chr_to_sig:
            self.logger.debug("Calculating mean abundance for each chromosome.")
            for chr_name, chr_sig in self.specific_chr_to_sig.items():
                self.logger.debug("Intersecting %s (%d) with %s (%d)", sample_sig.name, len(sample_sig), chr_name, len(chr_sig))
                chr_sample_sig = sample_sig & chr_sig
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
                f"chr-{chr_name.replace('sex-','').replace('autosome-','')}": chr_to_mean_abundance[chr_name]
                for chr_name in sorted(chr_to_mean_abundance, key=sort_chromosomes)
            }
            
            # Delete the mitochondrial from sorted_chr_to_mean_abundance
            if "mitochondrial-M" in sorted_chr_to_mean_abundance:
                self.logger.debug("Removing mitochondrial-M from sorted_chr_to_mean_abundance.")
                sorted_chr_to_mean_abundance.pop("mitochondrial-M")
            
            chrs_stats.update(sorted_chr_to_mean_abundance)

            # chr_to_mean_abundance but without any chr with partial name sex
            autosomal_chr_to_mean_abundance = {}
            for chr_name, mean_abundance in chr_to_mean_abundance.items():
                if "sex" in chr_name.lower() or "-snipegenome" in chr_name.lower() or "mitochondrial" in chr_name.lower():
                    self.logger.debug("Skipping %s from autosomal_chr_to_mean_abundance.", chr_name)
                    continue

                self.logger.debug("Adding %s to autosomal_chr_to_mean_abundance.", chr_name)
                autosomal_chr_to_mean_abundance[chr_name] = mean_abundance


            # calculate the CV for the whole sample
            if autosomal_chr_to_mean_abundance:
                mean_abundances = np.array(list(autosomal_chr_to_mean_abundance.values()), dtype=np.float64)
                cv = np.std(mean_abundances) / np.mean(mean_abundances) if np.mean(mean_abundances) != 0 else 0.0
                chrs_stats.update({"Autosomal_CV": cv})
                assert "Autosomal_CV" in chrs_stats
                self.logger.debug("Calculated Autosomal CV: %f", cv)
            else:
                self.logger.warning("No autosomal chromosomes were processed. 'Autosomal_CV' set to None.")
                chrs_stats.update({"Autosomal_CV": None})
                assert "Autosomal_CV" in chrs_stats
                
            
            # ============= SEX STATS =============
            
            # Ensure that the chromosome X signature exists
            
            self.logger.debug("Length of genome before subtracting sex chromosomes %s", len(self.reference_sig))
            autosomals_genome_sig = self.reference_sig.copy()
            for chr_name, chr_sig in self.specific_chr_to_sig.items():
                if "sex" in chr_name.lower() or "mitochondrial" in chr_name.lower():
                    self.logger.debug("Removing %s chromosome from the autosomal genome signature.", chr_name)
                    self.logger.debug("Length of autosomals_genome_sig: %s | Length of chr_sig: %s", len(autosomals_genome_sig), len(chr_sig))
                    autosomals_genome_sig -= chr_sig
            self.logger.debug("Length of genome after subtracting sex chromosomes %s", len(autosomals_genome_sig))
            
            if 'sex-x' not in self.specific_chr_to_sig:
                self.logger.warning("Chromosome X ('sex-x') not found in the provided signatures. X-Ploidy score will be set to zero.")
                # set sex-x to an empty signature
                self.specific_chr_to_sig['sex-x'] = SnipeSig.create_from_hashes_abundances(
                    hashes=np.array([], dtype=np.uint64),
                    abundances=np.array([], dtype=np.uint32),
                    ksize= self.specific_chr_to_sig[list( self.specific_chr_to_sig.keys())[0]].ksize,
                    scale= self.specific_chr_to_sig[list( self.specific_chr_to_sig.keys())[0]].scale,
                )
            else:
                self.logger.debug("X chromosome ('sex-x') detected.")
                
            
            # Separate the autosomal genome signature from chromosome-specific signatures

            #! autosomal sig for now is the all of the genome minus sex chrs
            self.logger.debug("Separating autosomal genome signature from chromosome-specific signatures.")
            
            
            # Derive the X chromosome-specific signature by subtracting autosomal genome hashes
            specific_xchr_sig = self.specific_chr_to_sig["sex-x"] - autosomals_genome_sig
            self.logger.debug("\t-Derived X chromosome-specific signature size: %d hashes.", len(specific_xchr_sig))
            
            # Intersect the sample signature with chromosome-specific signatures
            sample_specific_xchr_sig = sample_sig & self.specific_chr_to_sig['sex-x']
            if len(sample_specific_xchr_sig) == 0:
                self.logger.warning("No X chromosome-specific k-mers found in the sample signature.")
            self.logger.debug("\t-Intersected sample signature with X chromosome-specific k-mers = %d hashes.", len(sample_specific_xchr_sig))
            sample_autosomal_sig = sample_sig & autosomals_genome_sig
            self.logger.debug("\t-Intersected sample signature with autosomal genome k-mers = %d hashes.", len(sample_autosomal_sig))
            
            # Retrieve mean abundances
            xchr_mean_abundance = sample_specific_xchr_sig.get_sample_stats.get("mean_abundance", 0.0)
            autosomal_mean_abundance = sample_autosomal_sig.get_sample_stats.get("mean_abundance", 0.0)
            
            xchr_total_abundance = sample_specific_xchr_sig.get_sample_stats.get("total_abundance", 0.0)
            autosomal_total_abundance = sample_autosomal_sig.get_sample_stats.get("total_abundance", 0.0)
            
            x_mean_from_total = xchr_total_abundance / len(sample_specific_xchr_sig)
            autosomal_mean_from_total = autosomal_total_abundance / len(sample_autosomal_sig)
            
            # compare 
            self.logger.debug("X chromosome mean %s | from total %s", xchr_mean_abundance, x_mean_from_total)
            self.logger.debug("Autosomal mean %s | from total %s", autosomal_mean_abundance, autosomal_mean_from_total)
            
            # Calculate X-Ploidy score
            if autosomal_mean_abundance == 0:
                self.logger.warning("Autosomal mean abundance is zero. Setting X-Ploidy score to zero to avoid division by zero.")
                xploidy_score = 0.0
            else:
                # xploidy_score = (xchr_mean_abundance / autosomal_mean_abundance) if len(specific_xchr_sig) > 0 else 0.0
                xploidy_score = (xchr_total_abundance / autosomal_total_abundance) * \
                                                (len(autosomals_genome_sig) / len(specific_xchr_sig) if len(specific_xchr_sig) > 0 else 0.0)    
                                                
                # (yfree_xchr_in_sample.total_abundance / yfree_autosomals_in_sample.total_abundance) * (len(yfree_autosomals_specific) / len(yfree_xchr_specific))        

            self.logger.debug("Calculated X-Ploidy score: %.4f", xploidy_score)
            sex_stats.update({"X-Ploidy score": xploidy_score})
            self.logger.debug("X-Ploidy score: %.4f", sex_stats["X-Ploidy score"])
            
            # Calculate Y-Coverage if Y chromosome is present
            if 'sex-y' in self.specific_chr_to_sig and 'sex-x' in self.specific_chr_to_sig:
                self.logger.debug("Calculating Y-Coverage based on Y chromosome-specific k-mers.")
                
                # Derive Y chromosome-specific k-mers by excluding autosomal and X chromosome k-mers
                ychr_specific_kmers = self.specific_chr_to_sig["sex-y"] - autosomals_genome_sig - specific_xchr_sig
                self.logger.debug("\t-Derived Y chromosome-specific signature size: %d hashes.", len(ychr_specific_kmers))
                
                # Intersect Y chromosome-specific k-mers with the sample signature
                ychr_in_sample = sample_sig & ychr_specific_kmers
                self.logger.debug("\t-Intersected sample signature with Y chromosome-specific k-mers = %d hashes.", len(ychr_in_sample))
                if len(ychr_in_sample) == 0:
                    self.logger.warning("No Y chromosome-specific k-mers found in the sample signature.")
                
                # Derive autosomal-specific k-mers by excluding X and Y chromosome k-mers from the reference signature
                autosomals_specific_kmers = self.reference_sig - self.specific_chr_to_sig["sex-x"] - self.specific_chr_to_sig['sex-y']
                
                # Calculate Y-Coverage metric
                if len(ychr_specific_kmers) == 0 or len(autosomals_specific_kmers) == 0:
                    self.logger.warning("Insufficient k-mers for Y-Coverage calculation. Setting Y-Coverage to zero.")
                    ycoverage = 0.0
                else:
                    ycoverage = (len(ychr_in_sample) / len(ychr_specific_kmers)) / (len(sample_autosomal_sig) / len(autosomals_specific_kmers))
                
                self.logger.debug("Calculated Y-Coverage: %.4f", ycoverage)
                sex_stats.update({"Y-Coverage": ycoverage})
            else:
                self.logger.warning("No Y chromosome-specific signature detected. Y-Coverage will be set to zero.")
                
        # ============= VARS NONREF STATS =============
        if self.variance_sigs:
            self.logger.debug("Consuming non-reference k-mers from provided variables.")
            self.logger.debug("\t-Current size of the sample signature: %d hashes.", len(sample_sig))
            
            sample_nonref = sample_sig - self.reference_sig

            self.logger.debug("\t-Size of non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
            sample_nonref.trim_singletons()
            self.logger.debug("\t-Size of non-reference k-mers after trimming singletons: %d hashes.", len(sample_nonref))
            
            sample_nonref_unique_hashes = len(sample_nonref)
            
            self.logger.debug("\t-Size of non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
            if len(sample_nonref) == 0:
                self.logger.warning("No non-reference k-mers found in the sample signature.")
                return {}
            
            # intersect and report coverage and depth, then subtract from sample_nonref so sum will be 100%
            for variance_sig in self.variance_sigs:
                variance_name = variance_sig.name
                sample_nonref_var: SnipeSig = sample_nonref & variance_sig
                
                if self.export_varsigs:
                    _export_var_name = variance_name.replace(' ','_').lower()
                    _export_sample_name = f"{sample_sig.name}_{_export_var_name}_nonref"
                    _export_name = _export_sample_name + '_' + _export_var_name
                    sample_nonref_var.name = _export_name
                    self.logger.debug("Exporting non-reference k-mers from variable '%s'.", variance_name)
                    sample_nonref_var.export(f"{sample_sig.name}_{variance_name}_nonref.zip")

                sample_nonref_var_total_abundance = sample_nonref_var.total_abundance
                sample_nonref_var_unique_hashes = len(sample_nonref_var)
                sample_nonref_var_coverage_index = sample_nonref_var_unique_hashes / sample_nonref_unique_hashes
                vars_nonref_stats.update({
                    f"{variance_name} non-genomic total k-mer abundance": sample_nonref_var_total_abundance,
                    f"{variance_name} non-genomic coverage index": sample_nonref_var_coverage_index
                })
                
                self.logger.debug("\t-Consuming non-reference k-mers from variable '%s'.", variance_name)
                sample_nonref -= sample_nonref_var
                self.logger.debug("\t-Size of remaining non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
                
            vars_nonref_stats["non-var non-genomic total k-mer abundance"] = sample_nonref.total_abundance
            vars_nonref_stats["non-var non-genomic coverage index"] = len(sample_nonref) / sample_nonref_unique_hashes if sample_nonref_unique_hashes > 0 else 0.0
            
            self.logger.debug(
                "After consuming all vars from the non reference k-mers, the size of the sample signature is: %d hashes, "
                "with total abundance of %s.", 
                len(sample_nonref), sample_nonref.total_abundance
            )
            

        # ============= Coverage Prediction (ROI) =============
        
        if predict_extra_folds:
            predicted_fold_coverage = {}
            predicted_fold_delta_coverage = {}
            nparts = 30
            if isinstance(self.amplicon_sig, SnipeSig):
                roi_reference_sig = self.amplicon_sig
                self.logger.debug("Using amplicon signature as ROI reference.")
            else:
                roi_reference_sig = self.reference_sig
                self.logger.debug("Using reference genome signature as ROI reference.")

            # Get sample signature intersected with the reference
            _sample_sig_genome = sample_sig & self.reference_sig
            hashes = _sample_sig_genome.hashes
            abundances = _sample_sig_genome.abundances
            N = len(hashes)

            # Generate random fractions using Dirichlet distribution
            fractions = np.random.dirichlet([1] * nparts, size=N)  # Shape: (N, nparts)

            # Calculate counts for each part
            counts = np.round(abundances[:, None] * fractions).astype(int)  # Shape: (N, nparts)

            # Adjust counts to ensure sums match original abundances
            differences = abundances - counts.sum(axis=1)
            indices = np.argmax(counts, axis=1)
            counts[np.arange(N), indices] += differences

            # Compute cumulative counts
            counts_cumulative = counts.cumsum(axis=1)  # Shape: (N, nparts)
            cumulative_total_abundances = counts.sum(axis=0).cumsum()

            coverage_depth_data = []

            # Force conversion to GENOME
            roi_reference_sig.sigtype = SigType.GENOME

            for i in range(nparts):
                cumulative_counts = counts_cumulative[:, i]
                idx = cumulative_counts > 0

                cumulative_hashes = hashes[idx]
                cumulative_abundances = cumulative_counts[idx]

                cumulative_snipe_sig = SnipeSig.create_from_hashes_abundances(
                    hashes=cumulative_hashes,
                    abundances=cumulative_abundances,
                    ksize=sample_sig.ksize,
                    scale=sample_sig.scale,
                    name=f"{sample_sig.name}_cumulative_part_{i+1}",
                    filename=sample_sig.filename,
                    enable_logging=self.enable_logging
                )

                # Compute coverage index
                cumulative_qc = ReferenceQC(
                    sample_sig=cumulative_snipe_sig,
                    reference_sig=roi_reference_sig,
                    enable_logging=self.enable_logging
                )
                cumulative_stats = cumulative_qc.get_aggregated_stats()
                cumulative_coverage_index = cumulative_stats.get("Genome coverage index", 0.0)
                cumulative_total_abundance = cumulative_total_abundances[i]

                coverage_depth_data.append({
                    "cumulative_parts": i + 1,
                    "cumulative_total_abundance": cumulative_total_abundance,
                    "cumulative_coverage_index": cumulative_coverage_index,
                })

                self.logger.debug("Added coverage depth data for cumulative part %d.", i + 1)

            self.logger.debug("Coverage vs depth calculation completed.")

            for extra_fold in predict_extra_folds:
                if extra_fold < 1:
                    self.warning.error("Extra fold must be >= 1. Skipping this extra fold prediction.")
                    continue

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
                predicted_fold_coverage[f"Predicted coverage with {extra_fold} extra folds"] = predicted_coverage
                _delta_coverage = predicted_coverage - y_data[-1]
                predicted_fold_delta_coverage[f"Predicted delta coverage with {extra_fold} extra folds"] = _delta_coverage
                if _delta_coverage < 0:
                    self.logger.warning(
                        "Predicted coverage at %.2f-fold increase is less than the current coverage (probably low complexity).",
                        extra_fold
                    )
                self.logger.debug("Predicted coverage at %.2f-fold increase: %f", extra_fold, predicted_coverage)
                self.logger.debug("Predicted delta coverage at %.2f-fold increase: %f", extra_fold, _delta_coverage)

            # Update the ROI stats
            roi_stats.update(predicted_fold_coverage)
            roi_stats.update(predicted_fold_delta_coverage)
                
        # ============= Merging all stats in one dictionary =============
        aggregated_stats = {}
        if sample_stats:
            aggregated_stats.update(sample_stats)
        if genome_stats:
            aggregated_stats.update(genome_stats)
        if amplicon_stats:
            aggregated_stats.update(amplicon_stats)
        if advanced_stats:
            aggregated_stats.update(advanced_stats)
        if chrs_stats: 
            aggregated_stats.update(chrs_stats) 
        else: 
            self.logger.warning("No chromosome stats were processed.")
        if sex_stats:
            aggregated_stats.update(sex_stats)
        else:
            self.logger.warning("No sex-metrics stats were processed.")
        if predicted_error_contamination_index:
            aggregated_stats.update(predicted_error_contamination_index)
        if vars_nonref_stats:
            aggregated_stats.update(vars_nonref_stats)
        if roi_stats:
            aggregated_stats.update(roi_stats)
            
        # update the class with the new sample
        self.sample_to_stats[sample_sig.name] = aggregated_stats
            
        return aggregated_stats
