import json
import numpy as np
import hashlib
import random
from enum import Enum, auto
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class SigType(Enum):
    SAMPLE = auto()
    GENOME = auto()
    AMPLICON = auto()


class RefStats:
    def __init__(self, ref_name):
        self.ref_name = ref_name
        self.stats = self._create_ref_amplicon_stats_template(ref_name)
    
    
    def _create_ref_amplicon_stats_template(self, ref_name):
        return {
            f"{ref_name}_coverage_index": -1.0,
            f"{ref_name}_unique_hashes": -1,
            f"{ref_name}_total_abundance": -1,
            f"{ref_name}_mean_abundance": -1.0,
            f"{ref_name}_median_abundance": -1.0,

            f"{ref_name}_median_trimmed_coverage_index": -1.0,
            f"{ref_name}_median_trimmed_unique_hashes": -1,
            f"{ref_name}_median_trimmed_total_abundance": -1,
            f"{ref_name}_median_trimmed_mean_abundance": -1.0,
            f"{ref_name}_median_trimmed_median_abundance": -1.0,
            
            f"non_{ref_name}_unique_hashes": -1,
            f"non_{ref_name}_total_abundance": -1,
            f"non_{ref_name}_mean_abundance": -1.0,
            f"non_{ref_name}_median_abundance": -1.0,
        }
    
    
    @property
    def coverage_index(self):
        return self.stats[f"{self.ref_name}_coverage_index"]
    
    @coverage_index.setter
    def coverage_index(self, value):
        self.stats[f"{self.ref_name}_coverage_index"] = value
    
    @property
    def unique_hashes(self):
        return self.stats[f"{self.ref_name}_unique_hashes"]
        
    @unique_hashes.setter
    def unique_hashes(self, value):
        self.stats[f"{self.ref_name}_unique_hashes"] = value
        
    @property
    def total_abundance(self):
        return self.stats[f"{self.ref_name}_total_abundance"]
    
    @total_abundance.setter
    def total_abundance(self, value):
        self.stats[f"{self.ref_name}_total_abundance"] = value
    
    @property
    def mean_abundance(self):
        return self.stats[f"{self.ref_name}_mean_abundance"]
        
    @mean_abundance.setter
    def mean_abundance(self, value):
        self.stats[f"{self.ref_name}_mean_abundance"] = value
    
    @property
    def median_abundance(self):
        return self.stats[f"{self.ref_name}_median_abundance"]
        
    @median_abundance.setter
    def median_abundance(self, value):
        self.stats[f"{self.ref_name}_median_abundance"] = value
        
    @property
    def median_trimmed_coverage_index(self):
        return self.stats[f"{self.ref_name}_median_trimmed_coverage_index"]
        
    @median_trimmed_coverage_index.setter
    def median_trimmed_coverage_index(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_coverage_index"] = value
    
    @property
    def median_trimmed_unique_hashes(self):
        return self.stats[f"{self.ref_name}_median_trimmed_unique_hashes"]
    
    @median_trimmed_unique_hashes.setter
    def median_trimmed_unique_hashes(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_unique_hashes"] = value
        
    @property
    def median_trimmed_total_abundance(self):
        return self.stats[f"{self.ref_name}_median_trimmed_total_abundance"]
        
    @median_trimmed_total_abundance.setter
    def median_trimmed_total_abundance(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_total_abundance"] = value
        
    @property
    def median_trimmed_mean_abundance(self):
        return self.stats[f"{self.ref_name}_median_trimmed_mean_abundance"]
    
    @median_trimmed_mean_abundance.setter
    def median_trimmed_mean_abundance(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_mean_abundance"] = value
    
    @property
    def median_trimmed_median_abundance(self):
        return self.stats[f"{self.ref_name}_median_trimmed_median_abundance"]
        
    @median_trimmed_median_abundance.setter
    def median_trimmed_median_abundance(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_median_abundance"] = value
    
    @property
    def non_ref_unique_hashes(self):
        return self.stats[f"non_{self.ref_name}_unique_hashes"]
    
    @non_ref_unique_hashes.setter
    def non_ref_unique_hashes(self, value):
        self.stats[f"non_{self.ref_name}_unique_hashes"] = value
        
    @property
    def non_ref_total_abundance(self):
        return self.stats[f"non_{self.ref_name}_total_abundance"]
        
    @non_ref_total_abundance.setter
    def non_ref_total_abundance(self, value):
        self.stats[f"non_{self.ref_name}_total_abundance"] = value
    
    @property
    def non_ref_mean_abundance(self):
        return self.stats[f"non_{self.ref_name}_mean_abundance"]
        
    @non_ref_mean_abundance.setter
    def non_ref_mean_abundance(self, value):
        self.stats[f"non_{self.ref_name}_mean_abundance"] = value
        
    @property
    def non_ref_median_abundance(self):
        return self.stats[f"non_{self.ref_name}_median_abundance"]
        
    @non_ref_median_abundance.setter
    def non_ref_median_abundance(self, value):
        self.stats[f"non_{self.ref_name}_median_abundance"] = value
        
    def all_stats(self):
        # make sure all stats are set
        if len(self.stats) != len(self._create_ref_amplicon_stats_template(self.ref_name)):
            raise ValueError("All stats must be set before accessing exporting them.")
        return self.stats

    def check_all_stats(self):
        # report the missing stats
        missing_stats = [k for k, v in self.stats.items() if v == -1]
        if missing_stats:
            raise ValueError(f"Missing stats: {missing_stats}")
    
    def __str__(self):
        return json.dumps(self.stats, indent=4)

class AmpliconStats(RefStats):
    def __init__(self, ref_name):
        super().__init__(ref_name)
    
        # add relative coverage_index
        self.stats[f"{ref_name}_relative_coverage_index"] = -1.0
        self.stats[f"{ref_name}_median_trimmed_relative_coverage_index"] = -1.0
        
        # relative abundance
        self.stats[f"{ref_name}_relative_mean_abundance"] = -1.0
        self.stats[f"{ref_name}_median_trimmed_relative_mean_abundance"] = -1.0

    @property
    def relative_coverage_index(self):
        return self.stats[f"{self.ref_name}_relative_coverage_index"]
    
    @relative_coverage_index.setter
    def relative_coverage_index(self, value):
        self.stats[f"{self.ref_name}_relative_coverage_index"] = value
        
    @property
    def median_trimmed_relative_coverage_index(self):
        return self.stats[f"{self.ref_name}_median_trimmed_relative_coverage_index"]
        
    @median_trimmed_relative_coverage_index.setter
    def median_trimmed_relative_coverage_index(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_relative_coverage_index"] = value

    @property
    def relative_mean_abundance(self):
        return self.stats[f"{self.ref_name}_relative_mean_abundance"]
        
    @relative_mean_abundance.setter
    def relative_mean_abundance(self, value):
        self.stats[f"{self.ref_name}_relative_mean_abundance"] = value
    
    @property
    def median_trimmed_relative_mean_abundance(self):
        return self.stats[f"{self.ref_name}_median_trimmed_relative_mean_abundance"]    
    
    @median_trimmed_relative_mean_abundance.setter
    def median_trimmed_relative_mean_abundance(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_relative_mean_abundance"] = value
        
    def all_stats(self):
        # todo: make it more professional
        if len(self.stats) != len(self._create_ref_amplicon_stats_template(self.ref_name)) + 4:
            raise ValueError("All stats must be set before accessing exporting them.")        
        return self.stats
    

class SampleStats:
    def __init__(self, sample_name):
        self.sample_name = sample_name
        self.stats = self._create_sample_stats_template(sample_name)
        
    def _create_sample_stats_template(self, sample_name):
        return {
            "unique_hashes": 0,
            "total_abundance": 0,
            "mean_abundance": 0.0,
            "median_abundance": 0.0,
            "median_trimmed_unique_hashes": 0,
            "median_trimmed_total_abundance": 0,
            "median_trimmed_mean_abundance": 0.0,
            "median_trimmed_median_abundance": 0.0
        }
        


    @property
    def unique_hashes(self):
        return self.stats["unique_hashes"]
        
    @unique_hashes.setter
    def unique_hashes(self, value):
        self.stats["unique_hashes"] = value
    
    @property
    def total_abundance(self):
        return self.stats["total_abundance"]
        
    @total_abundance.setter
    def total_abundance(self, value):
        if not isinstance(value, int):
            raise ValueError("total_abundance must be an integer.")
        self.stats["total_abundance"] = value
        
    @property
    def mean_abundance(self):
        return self.stats["mean_abundance"]
        
    @mean_abundance.setter
    def mean_abundance(self, value):
        self.stats["mean_abundance"] = value
        
    @property
    def median_abundance(self):
        return self.stats["median_abundance"]
        
    @median_abundance.setter
    def median_abundance(self, value):
        self.stats["median_abundance"] = value
    
    @property
    def median_trimmed_unique_hashes(self):
        return self.stats["median_trimmed_unique_hashes"]
    
    @median_trimmed_unique_hashes.setter
    def median_trimmed_unique_hashes(self, value):
        self.stats["median_trimmed_unique_hashes"] = value
        
    @property
    def median_trimmed_total_abundance(self):
        return self.stats["median_trimmed_total_abundance"]

    @median_trimmed_total_abundance.setter
    def median_trimmed_total_abundance(self, value):
        self.stats["median_trimmed_total_abundance"] = value
        
    @property
    def median_trimmed_mean_abundance(self):
        return self.stats["median_trimmed_mean_abundance"]
        
    @median_trimmed_mean_abundance.setter
    def median_trimmed_mean_abundance(self, value):
        self.stats["median_trimmed_mean_abundance"] = value
        
    @property
    def median_trimmed_median_abundance(self):
        return self.stats["median_trimmed_median_abundance"]
        
    @median_trimmed_median_abundance.setter
    def median_trimmed_median_abundance(self, value):
        self.stats["median_trimmed_median_abundance"] = value

    
    @property
    def all_stats(self):
        # make sure all stats are set
        if len(self.stats) != len(self._create_sample_stats_template(self.sample_name)):
            raise ValueError("All stats must be set before accessing exporting them.")        
        return self.stats
        
  

class Signature:
    def __init__(self, k_size: int, signature_type: SigType = SigType.SAMPLE):
        if not isinstance(signature_type, SigType):
            raise ValueError(f"signature_type must be an instance of SignatureType, got {type(signature_type)}")
        
        self._k_size = k_size
        self._hashes = np.array([], dtype=np.uint64)
        self._abundances = np.array([], dtype=np.uint64)
        self._md5sum = ""
        self._scale = 0
        self._name = ""
        self._type = signature_type
        self._reference_signature = None
        self._amplicon_signatures = {}
        self.reference_stats = None
        self.amplicon_stats = {}
        self.genomic_roi_stats_data = None
        self.amplicons_roi_stats_data = {}
        
        

    def load_from_json_string(self, json_string: str):
        try:
            data = json.loads(json_string)
            return self.process_signature_data(data)
        except json.JSONDecodeError as e:
            return None, False, f"Error: The provided string is not in valid JSON format. {str(e)}"
        except Exception as e:
            return None, False, f"An unexpected error occurred: {e}"

    def load_from_path(self, path: str):
        try:
            with open(path, 'r') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):
                        return self.process_signature_data(data[0])
                    else:
                        return None, False, "Error: Expected a list of items in JSON."
                except json.JSONDecodeError:
                    return None, False, "Error: Incomplete or invalid JSON content."
                except IndexError:
                    return None, False, "Error: No items found in JSON."
        except FileNotFoundError:
            return None, False, f"Error: File '{path}' not found."
        except Exception as e:
            return None, False, f"An unexpected error occurred: {e}"

    def process_signature_data(self, data):
        found = False
        if not isinstance(data, list):
            data = [data]
        for d in data:
            for signature in d.get("signatures", []):
                if signature.get("ksize") == self._k_size:
                    found = True
                    self._name = d.get("name", "")
                    self._hashes = np.array(signature.get("mins", []), dtype=np.uint64)
                    self._md5sum = signature.get("md5sum", "")
                    self._scale = 18446744073709551615 // signature.get("max_hash", 1)
                    if "abundances" in signature:
                        self._abundances = np.array(signature["abundances"], dtype=np.uint64)
                        if len(self._hashes) != len(self._abundances):
                            return None, False, "Error: The number of hashes and abundances do not match."
                    else:
                        return None, True, "Note: Abundance data is missing for k-mer size {self._k_size}."
                    return None, True, "Signature loaded successfully."
        if not found:
            return None, False, "Error: k-mer size {self._k_size} not found."
        return None, True, "Signature processed successfully."
    
    

    # once we add a reference signature, we calculate all its relative metrics
    def add_reference_signature(self, reference_signature):
        if self.scale != reference_signature.scale:
            raise ValueError("scale must be the same")
        if self.k_size != reference_signature.k_size:
            raise ValueError("ksize must be the same")
        if reference_signature.type != SigType.GENOME:
            raise ValueError("Reference signature must be of type GENOME")
        
        ref_name = reference_signature.name
        
        # we only support a single reference signature at the time
        if not self._reference_signature:
            self._reference_signature = reference_signature
        else:
            raise ValueError("Reference signature is already set.")
        
        ref_stats = RefStats(ref_name)
        
        intersection_sig = self & reference_signature
        ref_stats.coverage_index = len(intersection_sig) / len(reference_signature)
        ref_stats.unique_hashes = len(intersection_sig)
        ref_stats.total_abundance = intersection_sig.total_abundance
        ref_stats.mean_abundance = intersection_sig.mean_abundance
        ref_stats.median_abundance = intersection_sig.median_abundance
        
        # median trimmed stats
        median_trimmed_sample_signature = self.__copy__()
        median_trimmed_sample_signature.apply_median_trim()
        median_trimmed_intersection_sig = median_trimmed_sample_signature & reference_signature        
        ref_stats.median_trimmed_coverage_index = len(median_trimmed_intersection_sig) / len(reference_signature)
        ref_stats.median_trimmed_unique_hashes = len(median_trimmed_intersection_sig)
        ref_stats.median_trimmed_total_abundance = median_trimmed_intersection_sig.total_abundance
        ref_stats.median_trimmed_mean_abundance = median_trimmed_intersection_sig.mean_abundance
        ref_stats.median_trimmed_median_abundance = median_trimmed_intersection_sig.median_abundance
        
        # calculate non-reference stats
        non_ref_sig = self - reference_signature
        
        if len(non_ref_sig) == 0:
            # set all stats to 0
            ref_stats.non_ref_unique_hashes = 0
            ref_stats.non_ref_total_abundance = 0
            ref_stats.non_ref_mean_abundance = 0.0
            ref_stats.non_ref_median_abundance = 0.0
        else:
            ref_stats.non_ref_unique_hashes = len(non_ref_sig)
            ref_stats.non_ref_total_abundance = non_ref_sig.total_abundance
            ref_stats.non_ref_mean_abundance = non_ref_sig.mean_abundance
            ref_stats.non_ref_median_abundance = non_ref_sig.median_abundance
        
        # make sure all stats are set        
        ref_stats.check_all_stats()
        self.reference_stats = ref_stats
        
        return True


    def add_amplicon_signature(self, amplicon_signature, custom_name: str = None):
        # TODO: make strong action for amplicon signature name settiing
        # TODO: change custom_name to a more meaningful name
        if not self._reference_signature:
            raise ValueError("Reference signature must be set before adding amplicon signatures.")
        if self.scale != amplicon_signature.scale:
            raise ValueError("scale must be the same")
        if self.k_size != amplicon_signature.k_size:
            raise ValueError("ksize must be the same")
        if amplicon_signature.type != SigType.AMPLICON:
            raise ValueError("Amplicon signature must be of type AMPLICON")        
        amplicon_name = custom_name if custom_name else amplicon_signature.name
        # make sure there is no duplicate name or checksum
        if amplicon_name not in self._amplicon_signatures:
            # no duplicate mdsum either in dictionary values
            if amplicon_signature.md5sum not in [x.md5sum for x in self._amplicon_signatures.values()]:            
                self._amplicon_signatures[amplicon_name] = amplicon_signature
        else:
            raise ValueError("Amplicon signature is already added.")
        
        amplicon_stats = AmpliconStats(amplicon_name)
        
        amplicon_on_genome = np.intersect1d(self._reference_signature._hashes, amplicon_signature._hashes)
        if len(amplicon_on_genome) == 0:
            raise ValueError(f"Amplicon {amplicon_name} is not part of the reference genome.")
        amplicon_percentage_in_genome = len(amplicon_on_genome) / len(self._reference_signature)
        if amplicon_percentage_in_genome < 0.01:
            print(f"Warning: Amplicon {amplicon_name} is only {amplicon_percentage_in_genome * 100:.2f}% in the reference genome.")    
        

        intersection_sig = self & amplicon_signature
        amplicon_stats.coverage_index = len(intersection_sig) / len(amplicon_signature)
        amplicon_stats.relative_coverage_index = amplicon_stats.coverage_index / self.reference_stats.coverage_index
        amplicon_stats.unique_hashes = len(intersection_sig)
        amplicon_stats.total_abundance = intersection_sig.total_abundance
        amplicon_stats.mean_abundance = intersection_sig.mean_abundance
        amplicon_stats.median_abundance = intersection_sig.median_abundance
        
        
        subtracted_sig = self - amplicon_signature
        amplicon_stats.non_ref_unique_hashes = len(subtracted_sig)
        amplicon_stats.non_ref_total_abundance = subtracted_sig.total_abundance
        amplicon_stats.non_ref_mean_abundance = subtracted_sig.mean_abundance
        amplicon_stats.non_ref_median_abundance = subtracted_sig.median_abundance
        
        
        amplicon_stats.relative_mean_abundance = amplicon_stats.median_abundance / amplicon_stats.non_ref_mean_abundance
        
        
        
        # median trimmed stats
        median_trimmed_sample_signature = self.__copy__()
        median_trimmed_sample_signature.apply_median_trim()
        median_trimmed_intersection_sig = median_trimmed_sample_signature & amplicon_signature
        amplicon_stats.median_trimmed_coverage_index = len(median_trimmed_intersection_sig) / len(amplicon_signature)
        amplicon_stats.median_trimmed_relative_coverage_index = amplicon_stats.median_trimmed_coverage_index / self.reference_stats.median_trimmed_coverage_index
        amplicon_stats.median_trimmed_unique_hashes = len(median_trimmed_intersection_sig)
        amplicon_stats.median_trimmed_total_abundance = median_trimmed_intersection_sig.total_abundance
        amplicon_stats.median_trimmed_mean_abundance = median_trimmed_intersection_sig.mean_abundance
        amplicon_stats.median_trimmed_median_abundance = median_trimmed_intersection_sig.median_abundance
        amplicon_stats.median_trimmed_relative_mean_abundance = amplicon_stats.median_trimmed_mean_abundance / self.reference_stats.median_trimmed_mean_abundance

        
        # make sure all stats are set
        amplicon_stats.check_all_stats()
        
        _amplicon_final_name = custom_name if custom_name else amplicon_name
        
        # make sure there is no duplicate name or checksum
        if _amplicon_final_name in self.amplicon_stats:
            raise ValueError(f"Amplicon {_amplicon_final_name} is already added.")
        
        self.amplicon_stats[_amplicon_final_name] = amplicon_stats
        return True


    def apply_median_trim(self):
        median_abundance = np.median(self._abundances)
        mask = self._abundances >= median_abundance
        self._hashes = self._hashes[mask]
        self._abundances = self._abundances[mask]
        
        if not len(self._hashes):
            raise ValueError("Median trimmed signature is empty.")
        
        self._checksum()

    
    def apply_abundance_filter(self, min_abundance: int):
        mask = self._abundances >= min_abundance
        self._hashes = self._hashes[mask]
        self._abundances = self._abundances[mask]
        if not len(self._hashes):
            raise ValueError("Abundance filtered signature is empty.")
        self._checksum()
        
    
    
    def _checksum(self):
        md5_ctx = hashlib.md5()
        md5_ctx.update(str(self._k_size).encode('utf-8'))   
        for x in self._hashes:
            md5_ctx.update(str(x).encode('utf-8'))
        self._md5sum = md5_ctx.hexdigest()
        return self._md5sum

    @property
    def scale(self):
        return self._scale
    
    @property
    def md5sum(self):
        return self._md5sum
    
    @property
    def hashes(self):
        return self._hashes
    
    @property
    def abundances(self):
        return self._abundances
    
    @property
    def name(self):
        return self._name
    
    @property
    def type(self):
        return self._type
    
    @property
    def k_size(self):
        return self._k_size
    
    @property
    def unique_hashes(self):
        return len(self._hashes)
    
    @property
    def mean_abundance(self):
        try:
            mean_value = np.mean(self._abundances)
        except ZeroDivisionError:
            raise ValueError("Mean abundance cannot be calculated for an empty signature.")
        
        return mean_value
    
    @property
    def median_abundance(self):
        try:
            median_abundance =  np.median(self._abundances)
        except ZeroDivisionError:
            raise ValueError("Median abundance cannot be calculated for an empty signature.")
        
        return median_abundance
        
    
    @property
    def total_abundance(self):
        return sum(self._abundances)
    
    @property
    def median_trimmed_stats(self):
        median_abundance = np.median(self._abundances)
        mask = self._abundances >= median_abundance
        trimmed_abundances = self._abundances[mask]
        _total_abundance = int(sum(trimmed_abundances))
        _median = np.median(trimmed_abundances)
        _mean = np.mean(trimmed_abundances)
        _std = np.std(trimmed_abundances)
        return {
            "total": _total_abundance,
            "median": _median,
            "mean": _mean,
            "std": _std
        }
    
    @property
    def abundance_stats(self):
        return {
            "total": int(np.sum(self._abundances)),
            "median": int(np.median(self._abundances)),
            "mean": int(np.mean(self._abundances)),
            "std": int(np.std(self._abundances))
        }
        
    @property
    def all_stats(self):
        d = {
            "unique_hashes": self.unique_hashes,
            "total_abundance": int(self.total_abundance),
            "abundance_stats": self.abundance_stats,
            "median_trimmed_stats": self.median_trimmed_stats
        }
        return d
    
    
    def __str__(self):
        d = {
            "k_size": self._k_size,
            "md5sum": self._md5sum,
            "hashes": len(self._hashes),
            "abundances": len(self._abundances),
            "scale": self._scale,
            "name": self._name,
            "type": f"{self._type}"
        }
        return json.dumps(d, indent=4)
    
    def _is_compatible_for_set_operation(self, other):
        """Check if the other signature is compatible for set operations based on its type."""
        # scale must match
        if self.scale != other.scale:
            raise ValueError("scale must be the same")
        
        if self.k_size != other.k_size:
            raise ValueError("ksize must be the same")
        
        compatible_combinations = [
            (SigType.SAMPLE, SigType.GENOME),
            (SigType.SAMPLE, SigType.SAMPLE),
            (SigType.SAMPLE, SigType.AMPLICON)
        ]
        if (self._type, other._type) not in compatible_combinations:
            raise ValueError(f"Signatures of type {self._type} and {other._type} are not compatible for set operations.")
    
    
    def __and__(self, other):
        """Intersection of two signatures, keeping hashes present in both."""
        if not isinstance(other, Signature):
            return NotImplemented
        self._is_compatible_for_set_operation(other)
        new_hashes, indices_self, indices_other = np.intersect1d(self._hashes, other._hashes, return_indices=True)
        new_abundances = self._abundances[indices_self]
        return self._create_new_signature(new_hashes, new_abundances)
    
    def __add__(self, other):
        """Union of two signatures, combining hashes and abundances."""
        if not isinstance(other, Signature):
            return NotImplemented
        
        # other must be of the same type
        if self._type != other._type:
            raise ValueError("Signatures must be of the same type.")

        # union hashes, and add abundances for same hash
        # Combine hashes and corresponding abundances
        combined_hashes = np.concatenate((self._hashes, other._hashes))
        combined_abundances = np.concatenate((self._abundances, other._abundances)).astype(np.uint64)

        # Use np.unique to identify unique hashes and their indices, then sum the abundances
        unique_hashes, indices = np.unique(combined_hashes, return_inverse=True)
        summed_abundances = np.bincount(indices, weights=combined_abundances).astype(np.uint64)

        return self._create_new_signature(unique_hashes, summed_abundances)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __repr__(self):
        return f"Signature(hashes={self._hashes}, abundances={self._abundances}, type={self._type})"


    def __len__(self):
        return len(self._hashes)


    def __eq__(self, other):
        if not isinstance(other, Signature):
            return NotImplemented
        return self.md5sum == other.md5sum
    

    def __or__(self, other):
        """Union of two signatures, combining hashes and abundances."""
        if not isinstance(other, Signature):
            return NotImplemented
        self._is_compatible_for_set_operation(other)
        new_hashes = np.union1d(self._hashes, other._hashes)
        # For abundances, we sum them up where hashes overlap, and keep unique ones as they are.
        new_abundances = self._union_abundances(new_hashes, other)
        return self._create_new_signature(new_hashes, new_abundances)

    def __sub__(self, other):
        """Subtraction of two signatures, removing hashes in the second from the first."""
        if not isinstance(other, Signature):
            return NotImplemented
        self._is_compatible_for_set_operation(other)
        new_hashes = np.setdiff1d(self._hashes, other._hashes)
        new_abundances = self._abundances[np.isin(self._hashes, new_hashes)]
        return self._create_new_signature(new_hashes, new_abundances)
    
    # allow deep copy of the signature
    def __copy__(self):
        return self._create_new_signature(self._hashes.copy(), self._abundances.copy())
    

    def _create_new_signature(self, hashes, abundances, suffix = None):
        new_sig = Signature(self._k_size)
        new_sig._hashes = hashes
        new_sig._abundances = abundances
        new_sig._checksum()
        new_sig._scale = self._scale
        new_sig._name = f"{self._name}"
        new_sig._type = self._type
        new_sig._k_size = self._k_size
        new_sig._reference_signature = None
        new_sig._amplicon_signatures = {}
        # new_sig._reference_signature = self._reference_signature
        # new_sig._amplicon_signatures = self._amplicon_signatures
        # new_sig.reference_stats = self.reference_stats
        # new_sig.amplicon_stats = self.amplicon_stats
        

        if suffix:
            new_sig._name += f"_{suffix}"
        return new_sig

    def _union_abundances(self, new_hashes, other):
        """Compute abundances for the union of two signatures."""
        new_abundances = np.zeros_like(new_hashes, dtype=np.uint64)
        for i, hash_val in enumerate(new_hashes):
            if hash_val in self._hashes:
                idx = np.where(self._hashes == hash_val)[0][0]
                new_abundances[i] += self._abundances[idx]
            if hash_val in other._hashes:
                idx = np.where(other._hashes == hash_val)[0][0]
                new_abundances[i] += other._abundances[idx]
        return new_abundances
    
    def export_to_sourmash(self, output_path: str):
        # try import sourmash
        try:
            import sourmash
        except ImportError:
            raise ImportError("sourmash is required to export to sourmash format.")
        # create a new sourmash signature
        mh = sourmash.MinHash(n=0, scaled=self.scale, ksize=self.k_size, track_abundance=True)
        hash_to_abund = dict(zip(self.hashes, self.abundances))
        mh.set_abundances(hash_to_abund)
        # make sure extensoin is .sig, add it if not present
        if not output_path.endswith(".sig"):
            output_path += ".sig"
            print(f"Warning: Added .sig extension to the output path: {output_path}")

        finalSig = sourmash.SourmashSignature(mh, name=self.name, filename=output_path)
        with sourmash.sourmash_args.FileOutput(output_path, 'wt') as fp:
            sourmash.save_signatures([finalSig], fp=fp)
        
        print(f"Signature exported to Sourmash format: {output_path}")
        
    @staticmethod
    def distribute_kmers_random(original_dict, n):
        
        # Initialize the resulting dictionaries
        distributed_dicts = [{} for _ in range(n)]
        
        # Convert the dictionary to a sorted list of tuples (k, v) by key
        kmer_list = sorted(original_dict.items())
        
        # Flatten the k-mer list according to their abundance
        flat_kmer_list = []
        for k, v in kmer_list:
            flat_kmer_list.extend([k] * v)
        
        # Shuffle the flat list to randomize the distribution
        random.shuffle(flat_kmer_list)
        
        # Distribute the k-mers round-robin into the dictionaries
        for i, k in enumerate(flat_kmer_list):
            dict_index = i % n
            if k in distributed_dicts[dict_index]:
                distributed_dicts[dict_index][k] += np.uint64(1)
            else:
                distributed_dicts[dict_index][k] = np.uint64(1)
        
        return distributed_dicts
    
    
    def split_sig_randomly(self, n):
        # Split the signature into n random signatures
        hash_to_abund = dict(zip(self.hashes, self.abundances))
        random_split_sigs = self.distribute_kmers_random(hash_to_abund, n)
        # split_sigs = [
        #     self._create_new_signature(np.array(list(x.keys()), dtype=np.uint64), np.array(list(x.values()), dtype=np.uint64), f"split_{i}")
        #     for i, x in enumerate(random_split_sigs)
        # ]
        split_sigs = [
            self._create_new_signature(
                np.fromiter(x.keys(), dtype=np.uint64),
                np.fromiter(x.values(), dtype=np.uint64),
                f"{self.name}_split_{i}"
            )
            for i, x in enumerate(random_split_sigs)
        ]
        return split_sigs
    
    def reset_abundance(self, new_abundance = 1):
        # this function set the abundance for all hashes of the signature to a new value
        self._abundances = np.full_like(self._hashes, new_abundance)


    def select_kmers_min_abund(self, min_abundance):
        # keep only k-mers with abundance >= min_abundance
        mask = self._abundances >= min_abundance
        self._hashes = self._hashes[mask]
        self._abundances = self._abundances[mask]
        if not len(self._hashes):
            raise ValueError("No k-mers found with abundance >= min_abundance.")
        self._checksum()
        
    def select_kmers_max_abund(self, max_abundance):
        # keep only k-mers with abundance <= max_abundance
        mask = self._abundances <= max_abundance
        self._hashes = self._hashes[mask]
        self._abundances = self._abundances[mask]
        if not len(self._hashes):
            raise ValueError("No k-mers found with abundance <= max_abundance.")
        self._checksum()
    

    # return on investment ROI calculation
    def calculate_genomic_roi(self, n = 30):
        # check if the signature has a reference signature
        if not self._reference_signature:
            _err = "Reference signature must be set before calculating ROI."
            raise ValueError(_err)
        
        split_sigs = self.split_sig_randomly(n)
        sample_roi_stats_data = []

        # Initialize a cumulative signature for previous parts
        cumulative_snipe_sig = None
        n = len(split_sigs)
        for i in range(n):
            current_part = split_sigs[i]
            
            if cumulative_snipe_sig is None:
                cumulative_snipe_sig = current_part
                continue

            # Add the current part to the cumulative signature
            current_part_snipe_sig = cumulative_snipe_sig + current_part
            
            # Calculate the current part coverage
            current_part_snipe_sig.add_reference_signature(self._reference_signature)
            current_part_coverage_index = current_part_snipe_sig.reference_stats.coverage_index
            current_part_mean_abundance = current_part_snipe_sig.reference_stats.mean_abundance
            
            # Calculate the cumulative coverage_index up to the previous part
            cumulative_snipe_sig.add_reference_signature(self._reference_signature)
            previous_parts_coverage_index = cumulative_snipe_sig.reference_stats.coverage_index
            previous_parts_mean_abundance = cumulative_snipe_sig.reference_stats.mean_abundance
            
            # Calculate delta_coverage_index
            delta_coverage_index = current_part_coverage_index - previous_parts_coverage_index
            
            stats = {
                'current_part_coverage_index': current_part_coverage_index,
                'previous_mean_abundance': previous_parts_mean_abundance,
                'delta_coverage_index': delta_coverage_index
            }
            
            sample_roi_stats_data.append(stats)
            
            # Update the cumulative signature to include the current part
            cumulative_snipe_sig += current_part
        
        # keep the genomic roi stats data for further analysis
        self.genomic_roi_stats_data = sample_roi_stats_data
        return sample_roi_stats_data
    
    def get_genomic_roi_stats(self):
        if not self.genomic_roi_stats_data:
            raise ValueError("Genomic ROI stats data is not available.")
        
        # wasm-friendly format
        return {x['previous_mean_abundance']: x['delta_coverage_index'] for x in self.genomic_roi_stats_data}
    
    def get_amplicon_roi_stats(self, amplicon_name= None):
        
        if not self.amplicons_roi_stats_data:
            raise ValueError("Amplicon stats are not available.")
        if not amplicon_name and len(self.amplicons_roi_stats_data) == 1:
            amplicon_name = list(self.amplicons_roi_stats_data.keys())[0]
        elif amplicon_name:
            if amplicon_name not in self.amplicons_roi_stats_data:
                raise ValueError(f"Amplicon '{amplicon_name}' is not found. Available amplicons are: {list(self.amplicons_roi_stats_data.keys())}")

        
        # wasm-friendly format
        return {x['previous_mean_abundance']: x['delta_coverage_index'] for x in self.amplicons_roi_stats_data[amplicon_name]}
    
    # TODO change the name to calculate_amplicon_roi
    def calculate_exome_roi(self, amplicon_name = None, n = 30):
        # check if the signature has a reference signature
        if not self._reference_signature:
            _err = "Reference signature must be set before calculating ROI."
            raise ValueError(_err)
        
        split_sigs = self.split_sig_randomly(n)
        sample_roi_stats_data = []
        
        AMPLICON_SIGNATURE_for_ROI = None
                
        if not len(self._amplicon_signatures):
            raise ValueError("At least one amplicon signatures must be added before calculating ROI.")
        if not amplicon_name and len(self._amplicon_signatures) == 1:
            amplicon_name = list(self._amplicon_signatures.keys())[0]
        elif amplicon_name:
            if amplicon_name not in self._amplicon_signatures:
                raise ValueError(f"Amplicon signature '{amplicon_name}' is not found. Available amplicons are: {list(self._amplicon_signatures.keys())}")
            else:
                AMPLICON_SIGNATURE_for_ROI = self._amplicon_signatures[amplicon_name]

        # Initialize a cumulative signature for previous parts
        cumulative_snipe_sig = None
        n = len(split_sigs)
        for i in range(n):
            current_part = split_sigs[i]
            
            if cumulative_snipe_sig is None:
                cumulative_snipe_sig = current_part
                continue

            # Add the current part to the cumulative signature
            current_part_snipe_sig = cumulative_snipe_sig + current_part
            
            # Calculate the current part coverage
            # TODO: prevent adding the reference/amplicon signature multiple times
            current_part_snipe_sig.add_reference_signature(self._reference_signature)
            current_part_snipe_sig.add_amplicon_signature(AMPLICON_SIGNATURE_for_ROI, "amplicon")
            
            current_part_coverage_index = current_part_snipe_sig.amplicon_stats["amplicon"].coverage_index
            current_part_mean_abundance = current_part_snipe_sig.amplicon_stats["amplicon"].mean_abundance
            
            # Calculate the cumulative coverage_index up to the previous part
            cumulative_snipe_sig.add_reference_signature(self._reference_signature)
            cumulative_snipe_sig.add_amplicon_signature(AMPLICON_SIGNATURE_for_ROI, "amplicon")
            previous_parts_coverage_index = cumulative_snipe_sig.amplicon_stats["amplicon"].coverage_index
            previous_parts_mean_abundance = cumulative_snipe_sig.amplicon_stats["amplicon"].mean_abundance
            
            # Calculate delta_coverage_index
            delta_coverage_index = current_part_coverage_index - previous_parts_coverage_index
            
            stats = {
                'current_part_coverage_index': current_part_coverage_index,
                'previous_mean_abundance': previous_parts_mean_abundance,
                'delta_coverage_index': delta_coverage_index
            }
            
            sample_roi_stats_data.append(stats)
            
            # Update the cumulative signature to include the current part
            cumulative_snipe_sig += current_part
        
        # keep the genomic roi stats data for further analysis
        self.amplicons_roi_stats_data[amplicon_name] = sample_roi_stats_data
        return sample_roi_stats_data
    

    def _predict_ROI(self, df, n_predict, show_plot=False):
        if n_predict <= len(df):
            raise ValueError(f"n_predict must be greater than the number of training points. Required: more than {len(df)}, Provided: {n_predict}")

        # Check for and handle infinity or NaN values
        if not np.isfinite(df['delta_coverage_index']).all():
            raise ValueError("Input 'delta_coverage_index' contains infinity or NaN values.")
        
        # Handle zero values by adding a small constant
        df['delta_coverage_index'] = df['delta_coverage_index'].replace(0, np.nan)
        df['delta_coverage_index'] = df['delta_coverage_index'].ffill()
        
        if (df['delta_coverage_index'] <= 0).any():
            raise ValueError("Input 'delta_coverage_index' must be positive after zero handling.")

        # Train with all available data points
        X_train = df[['previous_mean_abundance']]
        y_train = np.log(df['delta_coverage_index'])

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Calculate the average distance between points on the x-axis
        x_values = df['previous_mean_abundance'].values
        average_distance = np.mean(np.diff(x_values))

        # Generate the x-values for extrapolation
        last_known_value = x_values[-1]
        n_extra = n_predict - len(df)
        extrapolated_values = np.arange(1, n_extra + 1) * average_distance + last_known_value

        # Print the extra increase in the x-axis
        extra_increase = extrapolated_values[-1] - last_known_value
        # print(f"Extra increase in x-axis to achieve new coverage: {extra_increase}")

        extrapolated_values = extrapolated_values.reshape(-1, 1)
        extrapolated_values_df = pd.DataFrame(extrapolated_values, columns=['previous_mean_abundance'])
        y_pred_extrapolated = np.exp(model.predict(extrapolated_values_df))
        
        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(X_train, np.exp(y_train), color='blue', label='Training points')
            plt.scatter(extrapolated_values, y_pred_extrapolated, color='purple', marker='x', label='Extrapolated points')
            
            # Plot the complete line
            X_all_extended = pd.concat([X_train, extrapolated_values_df])
            y_all_extended = np.concatenate([np.exp(y_train), y_pred_extrapolated])
            plt.plot(X_all_extended, y_all_extended, color='orange', label='Model prediction curve')
            
            plt.xlabel('Previous Mean Abundance')
            plt.ylabel('Delta coverage_index')
            plt.title('Training and Extrapolated Data Points')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        extrapolated_data = pd.DataFrame({
            'previous_mean_abundance': extrapolated_values.flatten(),
            'delta_coverage_index': y_pred_extrapolated
        })
        
        # Calculate the final coverage_index
        last_known_coverage_index = df.iloc[-1]['current_part_coverage_index']
        extrapolated_data['cumulative_coverage_index'] = last_known_coverage_index + extrapolated_data['delta_coverage_index'].cumsum()
        
        final_coverage_index = extrapolated_data.iloc[-1]['cumulative_coverage_index']
        
        combined_data = pd.concat([df, extrapolated_data]).reset_index(drop=True)
        
        predicted_points = {
            'x': extrapolated_data['previous_mean_abundance'].tolist(),
            'y': extrapolated_data['cumulative_coverage_index'].tolist()
        }
        
        return predicted_points, combined_data, final_coverage_index, extra_increase
    
    def predict_genomic_roi(self, n_predict, show_plot=False):
        if not self.genomic_roi_stats_data:
            raise ValueError("Genomic ROI stats data is not available.")
        
        df = pd.DataFrame(self.genomic_roi_stats_data)
        predicted_points, combined_data, final_coverage_index, extra_increase = self._predict_ROI(df, n_predict, show_plot)
        
        return extra_increase, final_coverage_index
    
    def predict_amplicon_roi(self, amplicon_name, n_predict, show_plot=False):
        if not self.amplicons_roi_stats_data:
            raise ValueError("Amplicon stats are not available.")
        if amplicon_name not in self.amplicons_roi_stats_data:
            raise ValueError(f"Amplicon '{amplicon_name}' is not found. Available amplicons are: {list(self.amplicons_roi_stats_data.keys())}")
        
        df = pd.DataFrame(self.amplicons_roi_stats_data[amplicon_name])
        predicted_points, combined_data, final_coverage_index, extra_increase = self._predict_ROI(df, n_predict, show_plot)
        
        return extra_increase, final_coverage_index

class Pangenome:
    def __init__(self):
        self._kSize = None
        self._quantitative_sig = None
        self._scale = None
        # counts how many signatures we added
        self._sigs_counter = 0
    
    
    def add_signature(self, signature):
        if not isinstance(signature, Signature):
            raise ValueError("Signature must be an instance of Signature.")

        # reset the abundance of the signature to 1
        signature.reset_abundance(new_abundance=1)
        
        if self._quantitative_sig is None:
            self._quantitative_sig = signature
            self._kSize = signature.k_size
            self._scale = signature.scale
        
        else:
            if self._kSize != signature.k_size:
                raise ValueError("ksize must be the same")
            if self._scale != signature.scale:
                raise ValueError("scale must be the same")
            self._quantitative_sig += signature
        
        self._sigs_counter += 1


    def get_pangenome_signature(self, percentage = 1.0):
        if self._sigs_counter <= 0:
            raise ValueError("At least one signature must be added to the pangenome.")
        
        if percentage < 0.1 or percentage > 100:
            raise ValueError("Percentage must be between 0.1 and 100.")

        min_abund_threshold = int(self._sigs_counter * percentage / 100)
        return self._quantitative_sig.select_kmers_min_abund(min_abund_threshold)
        
        
        