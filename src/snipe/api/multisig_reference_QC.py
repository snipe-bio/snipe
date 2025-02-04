import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from snipe.api.snipe_sig import SnipeSig
from snipe.api.enums import SigType
import os
from snipe.api.reference_QC import ReferenceQC

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
                 custom_logger: Optional[logging.Logger] = None,
                 **kwargs):
        
        self.logger = custom_logger or logging.getLogger(__name__)
        
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
        
        if self.specific_chr_to_sig is not None and len(self.specific_chr_to_sig) > 0:
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
        roi_stats: Dict[str, Any] = {}


        # ============= SAMPLE Verification =============


        self.logger.debug("Validating ksize and scale across signatures.")
        if sample_sig.ksize != self.reference_sig.ksize:
            self.logger.error("K-mer sizes do not match: sample_sig.ksize=%d vs reference_sig.ksize=%d", sample_sig.ksize, self.reference_sig.ksize)
            raise ValueError(f"sample_sig kszie ({sample_sig.ksize}) does not match reference_sig ksize ({self.reference_sig.ksize}).")
        if sample_sig.scale != self.reference_sig.scale:
            self.logger.error("Scale values do not match: sample_sig.scale=%d vs reference_sig.scale=%d", sample_sig.scale, self.reference_sig.scale)
            raise ValueError(f"sample_sig scale ({sample_sig.scale}) does not match reference_sig scale ({self.reference_sig.scale}).")
        
        if self.amplicon_sig is not None:
            if self.amplicon_sig.ksize != sample_sig.ksize:
                self.logger.error("K-mer sizes do not match: amplicon_sig.ksize=%d vs sample_sig.ksize=%d", self.amplicon_sig.ksize, sample_sig.ksize)
                raise ValueError(f"amplicon_sig ksize ({self.amplicon_sig.ksize}) does not match sample_sig ksize ({sample_sig.ksize}).")
            if self.amplicon_sig.scale != sample_sig.scale:
                self.logger.error("Scale values do not match: amplicon_sig.scale=%d vs sample_sig.scale=%d", self.amplicon_sig.scale, sample_sig.scale)
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
            "Experiment ID": sample_stats_raw["name"],
            "ksize": sample_stats_raw["ksize"],
            "scale": sample_stats_raw["scale"],
            "filename": sample_stats_raw["filename"],
            "Total unique k-mers": sample_stats_raw["num_hashes"],
            "k-mer total abundance": sample_stats_raw["total_abundance"],
            "k-mer mean abundance": sample_stats_raw["mean_abundance"],
            "k-mer median abundance": sample_stats_raw["median_abundance"],
            "singleton k-mers": sample_stats_raw["num_singletons"],
            "snipe bases": sample_stats_raw["snipe_bases"],
            "k-mer-to-bases ratio": (
                (sample_stats_raw["total_abundance"] * sample_stats_raw["scale"]) / sample_stats_raw["snipe_bases"]
                if sample_stats_raw["snipe_bases"] > 0 else 0
            ),

        })

        # ============= GENOME STATS =============

        self.logger.debug("Calculating genome statistics.")
        # Compute intersection of sample and reference genome
        self.logger.debug("Type of sample_sig: %s | Type of reference_sig: %s", sample_sig.sigtype, self.reference_sig.sigtype)
        sample_genome = sample_sig & self.reference_sig

        sample_genome_stats = sample_genome.get_sample_stats

        genome_stats.update({
            "Genomic unique k-mers": sample_genome_stats["num_hashes"],
            "Genomic k-mers total abundance": sample_genome_stats["total_abundance"],
            "Genomic k-mers mean abundance": sample_genome_stats["mean_abundance"],
            "Genomic k-mers median abundance": sample_genome_stats["median_abundance"],

            "Genome coverage index": (
                sample_genome_stats["num_hashes"] / len(self.reference_sig)
                if len(self.reference_sig) > 0 and sample_genome_stats["num_hashes"] is not None else 0
            ),
            "Mapping index": (
                sample_genome_stats["total_abundance"] / sample_stats["k-mer total abundance"]
                if sample_stats.get("k-mer total abundance", 0) > 0 and sample_genome_stats["total_abundance"] is not None else 0
            ),
        })

        # ============= AMPLICON STATS =============
        if self.amplicon_sig is not None:
            _amplicon_ref_sig = self.amplicon_sig & self.reference_sig
            self.logger.debug(f"Amplicon signature is contained by the reference genome: {len(_amplicon_ref_sig) == len(self.amplicon_sig)} with intersection of {len(_amplicon_ref_sig)} hashes.") 
            if len(_amplicon_ref_sig) != len(self.amplicon_sig):
                _sig_to_be_removed = self.amplicon_sig - _amplicon_ref_sig
                _percentage_of_removal = len(_sig_to_be_removed) / len(self.amplicon_sig) * 100
                # if percentage is more than 20% then we should warn the user again
                if _percentage_of_removal > 20:
                    self.logger.error("[!] More than 20% of the amplicon signature is not contained in the reference genome.")
                    raise ValueError("Amplicon signature is poorly contained in the reference genome.")

                self.logger.debug(f"Amplicon signature is not fully contained in the reference genome.\nRemoving {len(_sig_to_be_removed)} hashes ({_percentage_of_removal:.2f}%) from the amplicon signature.")
                self.amplicon_sig.difference_sigs(_sig_to_be_removed)
                self.logger.debug("Amplicon signature has been modified to be fully contained in the reference genome.")
            
            
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
                        if len(self.amplicon_sig) > 0 and sample_amplicon_stats["num_hashes"] is not None else 0
                ),
            })

            # ============= RELATIVE STATS =============
            amplicon_stats["Relative total abundance"] = (
            amplicon_stats["Amplicon k-mers total abundance"] / genome_stats["Genomic k-mers total abundance"]
            if genome_stats.get("Genomic k-mers total abundance", 0) > 0 and 
            amplicon_stats.get("Amplicon k-mers total abundance") is not None else 0
            )

            amplicon_stats["Relative coverage"] = (
                amplicon_stats["Amplicon coverage index"] / genome_stats["Genome coverage index"]
                if genome_stats.get("Genome coverage index", 0) > 0 and 
                amplicon_stats.get("Amplicon coverage index") is not None else 0
            )
        
        else:
            self.logger.debug("No amplicon signature provided.")
        
        # ============= Contamination/Error STATS =============

        self.logger.debug("Calculuating error and contamination indices.")

        sample_nonref = sample_sig - self.reference_sig
        sample_nonref_singletons = sample_nonref.count_singletons()
        sample_nonref_non_singletons = sample_nonref.total_abundance - sample_nonref_singletons
        sample_total_abundance = sample_sig.total_abundance

        predicted_error_index = (
            sample_nonref_singletons / sample_total_abundance
            if sample_total_abundance is not None and sample_total_abundance > 0 else 0
        )

        predicted_contamination_index = (
            sample_nonref_non_singletons / sample_total_abundance
            if sample_total_abundance is not None and sample_total_abundance > 0 else 0
        )

        # predict error and contamination index
        predicted_error_contamination_index["Predicted contamination index"] = predicted_contamination_index
        predicted_error_contamination_index["Sequencing errors index"] = predicted_error_index
        
        #! TODO raw sketch singletons / raw sketch total abundance
        #! DEV NEW TEST PARAM
        _sample_ref = sample_sig & self.reference_sig
        _sample_ref_singletons = _sample_ref.count_singletons()
        _tmp_x = (_sample_ref_singletons / _sample_ref.total_abundance if _sample_ref.total_abundance is not None and _sample_ref.total_abundance > 0 else 0)
        _tmp_y = (sample_nonref_singletons / sample_nonref.total_abundance if sample_nonref.total_abundance is not None and sample_nonref.total_abundance > 0 else 0)
        _tmp_z_error_rate = _tmp_y - _tmp_x
        predicted_error_contamination_index["z_error_rate"] = _tmp_z_error_rate * 100
        
        _z_error_rate_raw_y = (sample_sig.count_singletons() / sample_sig.total_abundance) - _tmp_x
        predicted_error_contamination_index["z_error_rate_raw_y"] = _z_error_rate_raw_y * 100
        
        # genomic singltons
        predicted_error_contamination_index["genomic_singletons"] = _sample_ref_singletons
        
        
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
                    if len(self.reference_sig) > 0 and median_trimmed_sample_genome_stats.get("num_hashes") is not None else 0
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
                        if len(self.amplicon_sig) > 0 and median_trimmed_sample_amplicon_stats.get("num_hashes") is not None else 0
                    ),

                })
                # Additional advanced relative metrics
                self.logger.debug("Calculating advanced relative metrics.")
                amplicon_stats["Median-trimmed relative coverage"] = (
                    advanced_stats["Median-trimmed Amplicon coverage index"] / advanced_stats["Median-trimmed Genome coverage index"]
                    if advanced_stats.get("Median-trimmed Genome coverage index", 0) > 0 and 
                    advanced_stats.get("Median-trimmed Amplicon coverage index") is not None else 0
                )

                amplicon_stats["Median-trimmed relative mean abundance"] = (
                    advanced_stats["Median-trimmed Amplicon mean abundance"] / advanced_stats["Median-trimmed Genomic mean abundance"]
                    if advanced_stats.get("Median-trimmed Genomic mean abundance", 0) > 0 and 
                    advanced_stats.get("Median-trimmed Amplicon mean abundance") is not None else 0
                )

                # Update amplicon_stats with advanced metrics
                amplicon_stats.update({
                    "Median-trimmed relative coverage": amplicon_stats["Median-trimmed relative coverage"],
                    "Median-trimmed relative mean abundance": amplicon_stats["Median-trimmed relative mean abundance"],
                })

                advanced_stats.update(amplicon_stats)

        # ============= CHR STATS =============
        
        def sort_chromosomes(chrom_dict):
            # Sort based on numeric part or fallback to string for non-numeric chromosomes
            sorted_keys = sorted(chrom_dict, key=lambda x: (int(x.split('-')[1]) if x.split('-')[1].isdigit() else float('inf'), x))
            return {k: chrom_dict[k] for k in sorted_keys}
        
        if self.specific_chr_to_sig:
            self.logger.debug("Calculating mean abundance for each chromosome.")
            for chr_name, chr_sig in self.specific_chr_to_sig.items():
                self.logger.debug("Intersecting %s (%d) with %s (%d)", sample_sig.name, len(sample_sig), chr_name, len(chr_sig))
                chr_sample_sig = sample_sig & chr_sig
                chr_stats = chr_sample_sig.get_sample_stats
                chr_to_mean_abundance[chr_name] = chr_sample_sig.total_abundance / len(chr_sig)
                self.logger.debug("\t-Mean abundance for %s: %f", chr_name, chr_stats["mean_abundance"])

            # Create a new dictionary with sorted chromosome names and prefixed with 'chr-'
            sorted_chr_to_mean_abundance = {
                f"chr-{chr_name.replace('sex-','').replace('autosome-','')}": chr_to_mean_abundance[chr_name]
                for chr_name in sort_chromosomes(chr_to_mean_abundance)
            }
            
            # delete any key with mitochondrial in it
            sorted_chr_to_mean_abundance = {k: v for k, v in sorted_chr_to_mean_abundance.items() if "mitochondrial" not in k.lower()}
            
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
                mean = np.mean(mean_abundances)
                cv = np.std(mean_abundances) / mean if mean > 0 and not np.isnan(mean) else 0.0
                chrs_stats.update({"Autosomal k-mer mean abundance CV": cv})
                self.logger.debug("Calculated Autosomal CV: %f", cv)
            else:
                self.logger.warning("No autosomal chromosomes were processed. 'Autosomal_CV' set to None.")
                chrs_stats.update({"Autosomal k-mer mean abundance CV": None})
                
            
            # ============= SEX STATS =============
            
            # condition to see if there is partial lower(sex) match in the chromosome names
            if 'sex-x' in self.specific_chr_to_sig:
                self.logger.debug("Length of genome before subtracting sex chromosomes %s", len(self.reference_sig))
                autosomals_genome_sig = self.reference_sig.copy()
                for chr_name, chr_sig in self.specific_chr_to_sig.items():
                    if "sex" in chr_name.lower() or "mitochondrial" in chr_name.lower():
                        self.logger.debug("Removing %s chromosome from the autosomal genome signature.", chr_name)
                        self.logger.debug("Length of autosomals_genome_sig: %s | Length of chr_sig: %s", len(autosomals_genome_sig), len(chr_sig))
                        autosomals_genome_sig -= chr_sig
                self.logger.debug("Length of genome after subtracting sex chromosomes %s", len(autosomals_genome_sig))
                
                # if 'sex-x' not in self.specific_chr_to_sig:
                #     self.logger.warning("Chromosome X ('sex-x') not found in the provided signatures. chrX Ploidy score will be set to zero.")
                #     # set sex-x to an empty signature
                #     self.specific_chr_to_sig['sex-x'] = SnipeSig.create_from_hashes_abundances(
                #         hashes=np.array([], dtype=np.uint64),
                #         abundances=np.array([], dtype=np.uint32),
                #         ksize= self.specific_chr_to_sig[list( self.specific_chr_to_sig.keys())[0]].ksize,
                #         scale= self.specific_chr_to_sig[list( self.specific_chr_to_sig.keys())[0]].scale,
                #     )
                # else:
                #     self.logger.debug("X chromosome ('sex-x') detected.")
                
                # Derive the X chromosome-specific signature by subtracting autosomal genome hashes
                specific_xchr_sig = self.specific_chr_to_sig["sex-x"] - autosomals_genome_sig
                self.logger.debug("\t-Derived X chromosome-specific signature size: %d hashes.", len(specific_xchr_sig))
                
                # Intersect the sample signature with chromosome-specific signatures
                sample_specific_xchr_sig = sample_sig & self.specific_chr_to_sig['sex-x']
                if len(sample_specific_xchr_sig) == 0:
                    self.logger.warning("No X chromosome-specific k-mers found in the sample signature.")
                self.logger.debug("\t-Intersected sample signature with X chromosome-specific k-mers = %d hashes.", len(sample_specific_xchr_sig))
                sample_autosomal_sig = sample_sig & autosomals_genome_sig #! ( GENOME - SEX - MITO )
                self.logger.debug("\t-Intersected sample signature with autosomal genome k-mers = %d hashes.", len(sample_autosomal_sig))
                
                # Retrieve mean abundances
                xchr_mean_abundance = sample_specific_xchr_sig.total_abundance/ len(self.specific_chr_to_sig['sex-x']) if len(sample_specific_xchr_sig) > 0 else 0.0
                autosomal_mean_abundance = np.mean(list(autosomal_chr_to_mean_abundance.values())) if len(sample_autosomal_sig) > 0 else 0.0

                # Calculate chrX Ploidy score
                if autosomal_mean_abundance == 0:
                    self.logger.warning("Autosomal mean abundance is zero. Setting chrX Ploidy score to zero to avoid division by zero.")
                    xploidy_score = 0.0
                else:
                    xploidy_score = (
                        (xchr_mean_abundance / autosomal_mean_abundance)
                        if len(specific_xchr_sig) > 0 and autosomal_mean_abundance > 0 else 0.0
                    )

                self.logger.debug("Calculated chrX Ploidy score: %.4f", xploidy_score)
                sex_stats.update({"chrX Ploidy score": xploidy_score})
                self.logger.debug("chrX Ploidy score: %.4f", sex_stats["chrX Ploidy score"])
            else:
                self.logger.debug("No X chromosome-specific signature detected. chrX Ploidy score will be set to zero.")
            
            # Calculate chrY Coverage score if Y chromosome is present
            if 'sex-y' in self.specific_chr_to_sig and 'sex-x' in self.specific_chr_to_sig:
                self.logger.debug("Calculating chrY Coverage score based on Y chromosome-specific k-mers.")
                
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
                
                # Calculate chrY Coverage score metric
                if len(ychr_specific_kmers) == 0 or len(autosomals_specific_kmers) == 0:
                    self.logger.warning("Insufficient k-mers for chrY Coverage score calculation. Setting chrY Coverage score to zero.")
                    ycoverage = 0.0
                else:
                    try:
                        ycoverage = (
                            (len(ychr_in_sample) / len(ychr_specific_kmers)) / 
                            (len(sample_autosomal_sig) / len(autosomals_specific_kmers))
                        )
                    except (ZeroDivisionError, TypeError):
                        ycoverage = 0.0

                
                self.logger.debug("Calculated chrY Coverage score: %.4f", ycoverage)
                sex_stats.update({"chrY Coverage score": ycoverage})
            else:
                self.logger.warning("No Y chromosome-specific signature detected. chrY Coverage score will be set to zero.")
                
        # ============= VARS NONREF STATS =============
        if self.variance_sigs:
            self.logger.debug("Consuming non-reference k-mers from provided variables.")
            self.logger.debug("\t-Current size of the sample signature: %d hashes.", len(sample_sig))

            sample_nonref = sample_sig - self.reference_sig

            self.logger.debug("\tSize of non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
            # sample_nonref.trim_singletons()
            self.logger.debug("\tSize of non-reference k-mers after trimming singletons: %d hashes.", len(sample_nonref))

            sample_nonref_unique_hashes = len(sample_nonref)
            sample_nonref_total_abundance = sample_nonref.total_abundance

            self.logger.debug("\t-Size of non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
            if len(sample_nonref) == 0:
                self.logger.warning("No non-reference k-mers found in the sample signature.")
                return {}

            # intersect and report coverage and depth, then subtract from sample_nonref so sum will be 100%
            for variance_sig in self.variance_sigs:
                variance_name = variance_sig.name
                sample_nonref_var: SnipeSig = sample_nonref & variance_sig

                if self.export_varsigs:
                    __sample_name = sample_sig.name.replace(' ','_')
                    __var_name = variance_name.replace(' ','_')
                    __filename = os.path.basename(f"{__sample_name}_{__var_name}_nonref.zip".strip())
                    self.logger.debug("Exporting non-reference k-mers from variable '%s'.", __filename)
                    sample_nonref_var.export(__filename)

                sample_nonref_var_total_abundance = sample_nonref_var.total_abundance
                sample_nonref_var_fraction_total = (
                    sample_nonref_var_total_abundance / sample_nonref_total_abundance
                    if sample_nonref_total_abundance > 0 and sample_nonref_var_total_abundance is not None else 0.0
                )

                vars_nonref_stats.update({
                    f"{variance_name} total k-mer abundance": sample_nonref_var_total_abundance,
                    f"{variance_name} mean abundance": sample_nonref.mean_abundance,
                    f"{variance_name} median abundance": sample_nonref.median_abundance,
                    f"{variance_name} fraction of total abundance": sample_nonref_var_fraction_total
                })

                self.logger.debug("\t-Consuming non-reference k-mers from variable '%s'.", variance_name)
                sample_nonref -= sample_nonref_var
                self.logger.debug("\t-Size of remaining non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))

            vars_nonref_stats["unexplained variance total abundance"] = sample_nonref.total_abundance
            vars_nonref_stats["unexplained variance mean abundance"] = sample_nonref.mean_abundance
            vars_nonref_stats["unexplained variance median abundance"] = sample_nonref.median_abundance
            vars_nonref_stats["unexplained variance fraction of total abundance"] = (
                sample_nonref.total_abundance / sample_nonref_total_abundance
                if sample_nonref_total_abundance > 0 and sample_nonref.total_abundance is not None else 0.0
            )

            
            self.logger.debug(
                "After consuming all vars from the non reference k-mers, the size of the sample signature is: %d hashes, "
                "with total abundance of %s.", 
                len(sample_nonref), sample_nonref.total_abundance
            )
            

        # ============= Coverage Prediction (ROI) =============

        # Check if extra fold predictions are enabled and genome coverage index is sufficient
        if predict_extra_folds and genome_stats["Genome coverage index"] > 0.01:
            # Initialize dictionaries to store predicted coverages and unique hashes
            predicted_fold_coverage = {}
            predicted_fold_delta_coverage = {}
            predicted_unique_hashes = {}
            predicted_delta_unique_hashes = {}
            nparts = 30  # Number of parts to divide the data into for simulation
            roi_reference_sig = self.reference_sig  # Reference signature for ROI

            # Extract genome-specific signatures from the sample
            _sample_sig_genome = sample_sig & self.reference_sig

            hashes = _sample_sig_genome.hashes
            abundances = _sample_sig_genome.abundances
            N = len(hashes)  # Total number of unique hashes

            # Distribute abundances into parts using Dirichlet distribution
            # Instead of using Dirichlet distribution, perform random shuffling and round-robin splitting.
            total_repeats = []  # List to hold repeated hash indices based on their abundances
            for idx, abundance in enumerate(abundances):
                # Append the index 'idx' abundance times to the list
                total_repeats.extend([idx] * abundance)
            
            # Shuffle the list of repeated hash indices randomly
            np.random.shuffle(total_repeats)
            
            # Initialize a matrix to store counts: shape (number of unique hashes, nparts)
            counts = np.zeros((N, nparts), dtype=int)
            
            # Distribute shuffled hashes round-robin into parts
            for j, hash_idx in enumerate(total_repeats):
                part_index = j % nparts
                counts[hash_idx, part_index] += 1
            
            # Compute cumulative counts across parts as before
            counts_cumulative = counts.cumsum(axis=1)
            
            cumulative_total_abundances = counts.sum(axis=0).cumsum()  # Cumulative total abundances

            roi_reference_sig.sigtype = SigType.GENOME  # Set signature type to GENOME

            # Initialize arrays to store cumulative coverage indices and unique hashes
            cumulative_coverage_indices = np.zeros(nparts)
            cumulative_unique_hashes = np.zeros(nparts)
            total_reference_kmers = len(self.reference_sig)  # Total reference k-mers

            # Calculate cumulative coverage and unique k-mers for each part
            for i in range(nparts):
                cumulative_counts = counts_cumulative[:, i]
                idx = cumulative_counts > 0
                num_unique_kmers = np.sum(idx)
                cumulative_unique_hashes[i] = num_unique_kmers
                coverage_index = num_unique_kmers / total_reference_kmers
                cumulative_coverage_indices[i] = coverage_index

            # Compile coverage depth data for modeling
            coverage_depth_data = []
            for i in range(nparts):
                coverage_depth_data.append({
                    "cumulative_parts": i + 1,
                    "cumulative_total_abundance": cumulative_total_abundances[i],
                    "cumulative_coverage_index": cumulative_coverage_indices[i],
                    "cumulative_unique_hashes": cumulative_unique_hashes[i]
                })

            # Define the saturation model function for curve fitting
            def saturation_model(x, a, b):
                return a * x / (b + x)

            # Prepare data for unique k-mers saturation model
            x_unique = np.array([d["cumulative_total_abundance"] for d in coverage_depth_data])
            y_unique = np.array([d["cumulative_unique_hashes"] for d in coverage_depth_data])
            y_unique = np.maximum.accumulate(y_unique)  # Ensure monotonic increase
            initial_guess_unique = [y_unique[-1], x_unique[int(len(x_unique) / 2)]]

            # Fit the saturation model to unique k-mers data
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    params_unique, covariance_unique = curve_fit(
                        saturation_model,
                        x_unique,
                        y_unique,
                        p0=initial_guess_unique,
                        bounds=(0, np.inf),
                        maxfev=10000
                    )
            except (RuntimeError, OptimizeWarning) as exc:
                raise RuntimeError("Saturation model fitting for Genomic Unique k-mers failed.") from exc

            # Check for invalid covariance results
            if np.isinf(covariance_unique).any() or np.isnan(covariance_unique).any():
                raise RuntimeError("Saturation model fitting for Genomic Unique k-mers failed.")
            
            

            a_unique, b_unique = params_unique  # Extract fitted parameters
            # Correct total abundance based on sample data
            if sample_sig._bases_count > 0 and sample_sig._bases_count is not None:
                # The above code is calculating the corrected total abundance by dividing the
                # `_bases_count` attribute of the `sample_sig` object by the `scale` attribute of the
                # same object.
                # corrected_total_abundance = sample_sig._bases_count / sample_sig.scale
                kmer_to_bases_ratio = (sample_total_abundance * sample_sig.scale) / sample_sig._bases_count
                # ! TODO rename to corrected_genomic_total_abundance
                corrected_total_abundance = sample_genome_stats["total_abundance"] / kmer_to_bases_ratio
            else:
                #! TODO: BUG here (this will not be equal to the previous line)
                self.logger.warning("Total bases count is zero or None. This will affect the calculation of adjusted coverage index and k-mer-to-bases ratio.")
                corrected_total_abundance = x_unique[-1]

            # Predict genomic unique k-mers using the saturation model
            # a_unique = params_unique[0] = a = saturation point = max unique hashes
            # b_unique = params_unique[1] = b = total abundance at which saturation occurs
            predicted_genomic_unique_hashes = saturation_model(corrected_total_abundance, a_unique, b_unique)
            current_unique_hashes = y_unique[-1]
            # predicted_genomic_unique_hashes = max(predicted_genomic_unique_hashes, current_unique_hashes) #! BUG
            delta_unique_hashes = predicted_genomic_unique_hashes - current_unique_hashes
            adjusted_genome_coverage_index = predicted_genomic_unique_hashes / total_reference_kmers if total_reference_kmers > 0 else 0.0

            # Store predicted unique hashes information
            predicted_unique_hashes = {
                "Predicted genomic unique k-mers": predicted_genomic_unique_hashes,
                "Delta genomic unique k-mers": delta_unique_hashes,
                "Adjusted genome coverage index": adjusted_genome_coverage_index
            }

            x_total_abundance = x_unique
            y_coverage = cumulative_coverage_indices
            initial_guess_coverage = [y_coverage[-1], x_total_abundance[int(len(x_total_abundance) / 2)]]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    params_coverage, covariance_coverage = curve_fit(
                        saturation_model,
                        x_total_abundance,
                        y_coverage,
                        p0=initial_guess_coverage,
                        bounds=(0, np.inf),
                        maxfev=10000
                    )
            except (RuntimeError, OptimizeWarning) as exc:
                raise RuntimeError("Saturation model fitting for coverage failed.") from exc

            if np.isinf(covariance_coverage).any() or np.isnan(covariance_coverage).any():
                raise RuntimeError("Saturation model fitting for coverage failed.")

            a_coverage, b_coverage = params_coverage  # Extract fitted parameters

            # Predict coverage for each extra fold specified
            for extra_fold in predict_extra_folds:
                if extra_fold < 1:
                    continue  # Skip invalid fold values
                
                #! BETA: Predict the coverage based on the adjusted total abundance
                # total_abundance = x_total_abundance[-1]
                predicted_total_abundance = (1 + extra_fold) * sample_genome_stats["total_abundance"] # * corrected_total_abundance # x_total_abundance[-1]
                predicted_coverage = saturation_model(predicted_total_abundance, a_coverage, b_coverage)
                max_coverage = 1.0  # Maximum possible coverage
                #! CAPPED to maximum coverage
                # predicted_coverage = min(predicted_coverage, max_coverage)  # Cap coverage at max value
                predicted_fold_coverage[f"Predicted coverage with {extra_fold} extra folds"] = predicted_coverage
                _delta_coverage = predicted_coverage - y_coverage[-1]
                predicted_fold_delta_coverage[f"Predicted delta coverage with {extra_fold} extra folds"] = _delta_coverage

                if _delta_coverage < 0:
                    self.logger.warning(
                        "Predicted coverage at %.2f-fold increase is less than the current coverage.",
                        extra_fold
                    )

            roi_stats.update(predicted_fold_coverage)
            roi_stats.update(predicted_fold_delta_coverage)
            roi_stats.update(predicted_unique_hashes)

        else:
            self.logger.warning("Skipping ROI prediction due to zero Genomic Coverage Index.")

                
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
