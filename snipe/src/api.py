import json
import numpy as np
import hashlib

from enum import Enum, auto

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
            f"{ref_name}_saturation": -1.0,
            f"{ref_name}_unique_hashes": -1,
            f"{ref_name}_total_abundance": -1,
            f"{ref_name}_mean_abundance": -1.0,
            f"{ref_name}_median_abundance": -1.0,

            f"{ref_name}_median_trimmed_saturation": -1.0,
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
    def saturation(self):
        return self.stats[f"{self.ref_name}_saturation"]
    
    @saturation.setter
    def saturation(self, value):
        self.stats[f"{self.ref_name}_saturation"] = value
    
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
    def median_trimmed_saturation(self):
        return self.stats[f"{self.ref_name}_median_trimmed_saturation"]
        
    @median_trimmed_saturation.setter
    def median_trimmed_saturation(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_saturation"] = value
    
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
    
        # add relative saturation
        self.stats[f"{ref_name}_relative_saturation"] = -1.0
        self.stats[f"{ref_name}_median_trimmed_relative_saturation"] = -1.0
        
        # relative abundance
        self.stats[f"{ref_name}_relative_mean_abundance"] = -1.0
        self.stats[f"{ref_name}_median_trimmed_relative_mean_abundance"] = -1.0

    @property
    def relative_saturation(self):
        return self.stats[f"{self.ref_name}_relative_saturation"]
    
    @relative_saturation.setter
    def relative_saturation(self, value):
        self.stats[f"{self.ref_name}_relative_saturation"] = value
        
    @property
    def median_trimmed_relative_saturation(self):
        return self.stats[f"{self.ref_name}_median_trimmed_relative_saturation"]
        
    @median_trimmed_relative_saturation.setter
    def median_trimmed_relative_saturation(self, value):
        self.stats[f"{self.ref_name}_median_trimmed_relative_saturation"] = value

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
        self._hashes = np.array([], dtype=int)
        self._abundances = np.array([], dtype=int)
        self._md5sum = ""
        self._scale = 0
        self._name = ""
        self._type = signature_type
        self._reference_signature = None
        self._amplicon_signatures = {}
        self.reference_stats = None
        self.amplicon_stats = {}
        
        

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
                    self._hashes = np.array(signature.get("mins", []), dtype=int)
                    self._md5sum = signature.get("md5sum", "")
                    self._scale = 18446744073709551615 // signature.get("max_hash", 1)
                    if "abundances" in signature:
                        self._abundances = np.array(signature["abundances"], dtype=int)
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
        ref_stats.saturation = len(intersection_sig) / len(reference_signature)
        ref_stats.unique_hashes = len(intersection_sig)
        ref_stats.total_abundance = intersection_sig.total_abundance
        ref_stats.mean_abundance = intersection_sig.mean_abundance
        ref_stats.median_abundance = intersection_sig.median_abundance
        
        # median trimmed stats
        median_trimmed_sample_signature = self.__copy__()
        median_trimmed_sample_signature.apply_median_trim()
        median_trimmed_intersection_sig = median_trimmed_sample_signature & reference_signature        
        ref_stats.median_trimmed_saturation = len(median_trimmed_intersection_sig) / len(reference_signature)
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
        
        
    def add_amplicon_signature(self, amplicon_signature, name: str = None):
        if not self._reference_signature:
            raise ValueError("Reference signature must be set before adding amplicon signatures.")
        if self.scale != amplicon_signature.scale:
            raise ValueError("scale must be the same")
        if self.k_size != amplicon_signature.k_size:
            raise ValueError("ksize must be the same")
        if amplicon_signature.type != SigType.AMPLICON:
            raise ValueError("Amplicon signature must be of type AMPLICON")        
        amplicon_name = amplicon_signature.name
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
        amplicon_stats.saturation = len(intersection_sig) / len(amplicon_signature)
        amplicon_stats.relative_saturation = amplicon_stats.saturation / self.reference_stats.saturation
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
        amplicon_stats.median_trimmed_saturation = len(median_trimmed_intersection_sig) / len(amplicon_signature)
        amplicon_stats.median_trimmed_relative_saturation = amplicon_stats.median_trimmed_saturation / self.reference_stats.median_trimmed_saturation
        amplicon_stats.median_trimmed_unique_hashes = len(median_trimmed_intersection_sig)
        amplicon_stats.median_trimmed_total_abundance = median_trimmed_intersection_sig.total_abundance
        amplicon_stats.median_trimmed_mean_abundance = median_trimmed_intersection_sig.mean_abundance
        amplicon_stats.median_trimmed_median_abundance = median_trimmed_intersection_sig.median_abundance
        amplicon_stats.median_trimmed_relative_mean_abundance = amplicon_stats.median_trimmed_mean_abundance / self.reference_stats.median_trimmed_mean_abundance

        
        # make sure all stats are set
        amplicon_stats.check_all_stats()
        
        _amplicon_final_name = name if name else amplicon_name
        
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
        return np.sum(self._abundances)
    
    @property
    def median_trimmed_stats(self):
        median_abundance = np.median(self._abundances)
        mask = self._abundances >= median_abundance
        trimmed_abundances = self._abundances[mask]
        _total_abundance = np.sum(trimmed_abundances)
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
            "total": np.sum(self._abundances),
            "median": np.median(self._abundances),
            "mean": np.mean(self._abundances),
            "std": np.std(self._abundances)
        }
        
    @property
    def all_stats(self):
        d = {
            "unique_hashes": self.unique_hashes,
            "total_abundance": self.total_abundance,
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
        combined_abundances = np.concatenate((self._abundances, other._abundances))

        # Use np.unique to identify unique hashes and their indices, then sum the abundances
        unique_hashes, indices = np.unique(combined_hashes, return_inverse=True)
        summed_abundances = np.bincount(indices, weights=combined_abundances)

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
        new_abundances = np.zeros_like(new_hashes, dtype=int)
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
        