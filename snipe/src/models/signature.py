import json
import numpy as np
import ijson
import hashlib

from enum import Enum, auto

class SigType(Enum):
    SAMPLE = auto()
    GENOME = auto()
    AMPLICON = auto()

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
            with open(path, 'rb') as file:
                try:
                    data = ijson.items(file, 'item')
                    return self.process_signature_data(next(data))
                except ijson.common.IncompleteJSONError:
                    return None, False, "Error: Incomplete JSON content."
                except StopIteration:
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
    def total_abundance(self):
        return np.sum(self._abundances)
    
    @property
    def median_trimmed_stats(self):
        median_abundance = np.median(self._abundances)
        mask = self._abundances >= median_abundance
        trimmed_abundances = self._abundances[mask]
        _median = np.median(trimmed_abundances)
        _mean = np.mean(trimmed_abundances)
        _std = np.std(trimmed_abundances)
        return {
            "median": _median,
            "mean": _mean,
            "std": _std
        }
    
    @property
    def abundance_stats(self):
        return {
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
        new_abundances = np.minimum(self._abundances[indices_self], other._abundances[indices_other])
        return self._create_new_signature(new_hashes, new_abundances)

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

    def _create_new_signature(self, hashes, abundances, suffix = None):
        new_sig = Signature(self._k_size)
        new_sig._hashes = hashes
        new_sig._abundances = abundances
        new_sig._checksum()
        new_sig._scale = self._scale
        new_sig._name = f"{self._name}"
        new_sig._type = self._type
        new_sig._k_size = self._k_size
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
        
