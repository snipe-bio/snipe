import heapq
import logging

from snipe.api.enums import SigType
from typing import Dict, Iterator, List, Union, Optional
import numpy as np
import sourmash
import os

# Configure the root logger to CRITICAL to suppress unwanted logs by default
logging.basicConfig(level=logging.CRITICAL)


class SnipeSig:
    """
    A class to handle Sourmash signatures with additional functionalities
    such as customized set operations and abundance management.
    """

    def __init__(self, *, 
                 sourmash_sig: Union[str, sourmash.signature.SourmashSignature, sourmash.signature.FrozenSourmashSignature], 
                 sig_type=SigType.SAMPLE, enable_logging: bool = False, **kwargs):
        r"""
        Initialize the SnipeSig with a sourmash signature object or a path to a signature.

        Parameters:
            sourmash_sig (str or sourmash.signature.SourmashSignature): A path to a signature file or a signature object.
            ksize (int): K-mer size.
            scale (int): Scale value.
            sig_type (SigType): Type of the signature.
            enable_logging (bool): Flag to enable detailed logging.
            **kwargs: Additional keyword arguments.
        """
        # Initialize logging based on the flag
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configure the logger
        if enable_logging:
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.hasHandlers():
                # Create console handler
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # Create formatter
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                # Add formatter to handler
                ch.setFormatter(formatter)
                # Add handler to logger
                self.logger.addHandler(ch)
            self.logger.debug("Logging is enabled for SnipeSig.")
        else:
            self.logger.setLevel(logging.CRITICAL)

        # Initialize internal variables
        self.logger.debug("Initializing SnipeSig with sourmash_sig")

        self._scale: int = None
        self._ksize: int = None
        self._md5sum: str = None
        self._hashes = np.array([], dtype=np.uint64)
        self._abundances = np.array([], dtype=np.uint32)
        self._type: SigType = sig_type
        self._name: str = None
        self._filename: str = None
        self._track_abundance: bool = True

        sourmash_sigs: Dict[str, sourmash.signature.SourmashSignature] = {}
        _sourmash_sig: Union[sourmash.signature.SourmashSignature, sourmash.signature.FrozenSourmashSignature] = None
        
        self.chr_to_sig: Dict[str, SnipeSig] = {}

        
        self.logger.debug("Proceeding with a sigtype of %s", sig_type)
        
        if not isinstance(sourmash_sig, (str, sourmash.signature.SourmashSignature, sourmash.signature.FrozenSourmashSignature)):
            # if the str is not a file path
            self.logger.error("Invalid type for sourmash_sig: %s", type(sourmash_sig).__name__)
            raise TypeError(f"sourmash_sig must be a file path, sourmash.signature.SourmashSignature, or Frozensourmash_signature, got {type(sourmash_sig).__name__}")

        # Case 1: If sourmash_sig is already a valid sourmash signature object
        if isinstance(sourmash_sig, (sourmash.signature.FrozenSourmashSignature, sourmash.signature.SourmashSignature)):
            self.logger.debug("Loaded sourmash signature directly from object.")
            sourmash_sigs = {sourmash_sig.name: sourmash_sig}

        # Case 2: If sourmash_sig is a string, try to load as JSON or a file
        elif isinstance(sourmash_sig, str):
            self.logger.debug("Attempting to load sourmash signature from string input.")

            # First, try loading from JSON
            sourmash_sigs = self._try_load_from_json(sourmash_sig)
            self.logger.debug("Loaded sourmash signature from JSON: %s", sourmash_sigs)

            # If JSON loading fails, try loading from file
            if not sourmash_sigs:
                sourmash_sigs = self._try_load_from_file(sourmash_sig)

            # If both attempts fail, raise an error
            if not sourmash_sigs:
                self.logger.error("Failed to load sourmash signature from the provided string.")
                raise ValueError("An unexpected error occurred while loading the sourmash signature.")
        
        if sig_type == SigType.SAMPLE or sig_type == SigType.AMPLICON:
            if len(sourmash_sigs) > 1:
                self.logger.debug("Multiple signatures found in the input. Expected a single sample signature.")
                # not supported at this time
                raise ValueError("Loading multiple sample signatures is not supported at this time.")
            elif len(sourmash_sigs) == 1:
                self.logger.debug("Found a single signature in the sample sig input; Will use this signature.")
                _sourmash_sig = list(sourmash_sigs.values())[0]
            else:
                self.logger.debug("No signature found in the input. Expected a single sample signature.")
                raise ValueError("No signature found in the input. Expected a single sample signature.")
            
        elif sig_type == SigType.GENOME:
            if len(sourmash_sigs) > 1:
                for signame, sig in sourmash_sigs.items():
                    self.logger.debug(f"Iterating over signature: {signame}")
                    if signame.endswith("-snipegenome"):
                        sig = sig.to_mutable()
                        # self.chr_to_sig[sig.name] = SnipeSig(sourmash_sig=sig, sig_type=SigType.GENOME, enable_logging=enable_logging)
                        sig.name = sig.name.replace("-snipegenome", "")
                        self.logger.debug("Found a genome signature with the snipe suffix `-snipegenome`. Restoring original name `%s`.", sig.name)
                        _sourmash_sig = sig
                    elif signame.startswith("sex-"):
                        self.logger.debug("Found a sex chr signature %s", signame)
                        sig = sig.to_mutable()
                        self.chr_to_sig[sig.name] = SnipeSig(sourmash_sig=sig, sig_type=SigType.AMPLICON, enable_logging=enable_logging)
                    elif signame.startswith("autosome-"):
                        self.logger.debug("Found an autosome signature %s", signame)
                        sig = sig.to_mutable()
                        self.chr_to_sig[sig.name] = SnipeSig(sourmash_sig=sig, sig_type=SigType.AMPLICON, enable_logging=enable_logging)
                    elif signame.startswith("mitochondrial-"):
                        self.chr_to_sig[sig.name] = SnipeSig(sourmash_sig=sig, sig_type=SigType.AMPLICON, enable_logging=enable_logging)
                    else:
                        continue
                else:
                    if not _sourmash_sig:
                        self.logger.debug("Found multiple signature per the genome file, but none with the snipe suffix `-snipegenome`.")
                        raise ValueError("Found multiple signature per the genome file, but none with the snipe suffix `-snipegenome`.")
            elif len(sourmash_sigs) == 1:
                self.logger.debug("Found a single signature in the genome sig input; Will use this signature.")
                _sourmash_sig = list(sourmash_sigs.values())[0]
        else:
            self.logger.debug("Unknown sigtype: %s", sig_type)
            raise ValueError(f"Unknown sigtype: {sig_type}")
                
        self.logger.debug("Length of currently loaded signature: %d, with name: %s", len(_sourmash_sig), _sourmash_sig.name)

        # Extract properties from the loaded signature
        self._ksize = _sourmash_sig.minhash.ksize
        self._scale = _sourmash_sig.minhash.scaled
        self._md5sum = _sourmash_sig.md5sum()
        self._name = _sourmash_sig.name
        self._filename = _sourmash_sig.filename
        self._track_abundance = _sourmash_sig.minhash.track_abundance
        
        if self._name.endswith("-snipesample"):
            self._name = self._name.replace("-snipesample", "")
            self.logger.debug("Found a sample signature with the snipe suffix `-snipesample`. Restoring original name `%s`.", self._name)
        elif self._name.endswith("-snipeamplicon"):
            self._name = self._name.replace("-snipeamplicon", "")
            self.logger.debug("Found an amplicon signature with the snipe suffix `-snipeamplicon`. Restoring original name `%s`.", self._name)

        # If the signature does not track abundance, assume abundance of 1 for all hashes
        if not self._track_abundance:
            self.logger.debug("Signature does not track abundance. Setting all abundances to 1.")
            self._abundances = np.ones(len(_sourmash_sig.minhash.hashes), dtype=np.uint32)
            # self._track_abundance = True
        else:
            self._abundances = np.array(list(_sourmash_sig.minhash.hashes.values()), dtype=np.uint32)

        self._hashes = np.array(list(_sourmash_sig.minhash.hashes.keys()), dtype=np.uint64)

        # Sort the hashes and rearrange abundances accordingly
        sorted_indices = np.argsort(self._hashes)
        self._hashes = self._hashes[sorted_indices]
        self._abundances = self._abundances[sorted_indices]

        self.logger.debug(
            "Loaded sourmash signature from file: %s, name: %s, md5sum: %s, ksize: %d, scale: %d, "
            "track_abundance: %s, type: %s, length: %d",
            self._filename, self._name, self._md5sum, self._ksize, self._scale,
            self._track_abundance, self._type, len(self._hashes)
        )
        self.logger.debug("Hashes sorted during initialization.")
        self.logger.debug("Sourmash signature loading completed successfully.")

    def _try_load_from_json(self, sourmash_sig: str) -> Union[List[sourmash.signature.SourmashSignature], None]:
        r"""
        Attempt to load sourmash signature from JSON string.

        Parameters:
            sourmash_sig (str): JSON string representing a sourmash signature.

        Returns:
            sourmash.signature.SourmashSignature or None if loading fails.
        """
        try:
            self.logger.debug("Trying to load sourmash signature from JSON.")
            list_of_sigs = list(sourmash.load_signatures_from_json(sourmash_sig))
            return {sig.name: sig for sig in list_of_sigs}
        except Exception as e:
            self.logger.debug("Loading from JSON failed. Proceeding to file loading.", exc_info=e)
            return None  # Return None to indicate failure

    def _try_load_from_file(self, sourmash_sig_path: str) -> Union[List[sourmash.signature.SourmashSignature], None]:
        r"""
        Attempt to load sourmash signature(s) from a file.

        Parameters:
            sourmash_sig_path (str): File path to a sourmash signature.

        Returns:
            sourmash.signature.SourmashßSignature, list of sourmash.signature.SourmashSignature, or None if loading fails.
        """
        self.logger.debug("Trying to load sourmash signature from file.")
        try:
            signatures = list(sourmash.load_file_as_signatures(sourmash_sig_path))
            self.logger.debug("Loaded %d sourmash signature(s) from file.", len(signatures))
            sigs_dict = {_sig.name: _sig for _sig in signatures}
            self.logger.debug("Loaded sourmash signatures into sigs_dict: %s", sigs_dict)
            return sigs_dict
        except Exception as e:
            self.logger.exception("Failed to load the sourmash signature from the file.", exc_info=e)
            raise ValueError("An unexpected error occurred while loading the sourmash signature.") from e

    # Setters and getters
    @property
    def hashes(self) -> np.ndarray:
        r"""Return a copy of the hashes array."""
        return self._hashes.view()

    @property
    def abundances(self) -> np.ndarray:
        r"""Return a copy of the abundances array."""
        return self._abundances.view()

    @property
    def md5sum(self) -> str:
        r"""Return the MD5 checksum of the signature."""
        return self._md5sum

    @property
    def ksize(self) -> int:
        r"""Return the k-mer size."""
        return self._ksize

    @property
    def scale(self) -> int:
        r"""Return the scale value."""
        return self._scale

    @property
    def name(self) -> str:
        r"""Return the name of the signature."""
        return self._name

    @property
    def filename(self) -> str:
        r"""Return the filename of the signature."""
        return self._filename

    @property
    def sigtype(self) -> SigType:
        r"""Return the type of the signature."""
        return self._type

    @property
    def track_abundance(self) -> bool:
        r"""Return whether the signature tracks abundance."""
        return self._track_abundance

    # Basic class methods
    def get_name(self) -> str:
        r"""Get the name of the signature."""
        return self._name

    # setter sigtype
    @sigtype.setter
    def sigtype(self, sigtype: SigType):
        r"""
        Set the type of the signature.
        """
        self._type = sigtype
        
    @track_abundance.setter
    def track_abundance(self, track_abundance: bool):
        r"""
        Set whether the signature tracks abundance.
        """
        self._track_abundance = track_abundance

    def get_info(self) -> dict:
        r"""
        Get information about the signature.

        Returns:
            dict: A dictionary containing signature information.
        """
        info = {
            "name": self._name,
            "filename": self._filename,
            "md5sum": self._md5sum,
            "ksize": self._ksize,
            "scale": self._scale,
            "track_abundance": self._track_abundance,
            "sigtype": self._type,
            "num_hashes": len(self._hashes)
        }
        return info

    def __len__(self) -> int:
        r"""Return the number of hashes in the signature."""
        return len(self._hashes)

    def __iter__(self) -> Iterator[tuple]:
        r"""
        Iterate over the hashes and their abundances.

        Yields:
            tuple: A tuple containing (hash, abundance).
        """
        for h, a in zip(self._hashes, self._abundances):
            yield (h, a)

    def __contains__(self, hash_value: int) -> bool:
        r"""
        Check if a hash is present in the signature.

        Parameters:
            hash_value (int): The hash value to check.

        Returns:
            bool: True if the hash is present, False otherwise.
        """
        # Utilize binary search since hashes are sorted
        index = np.searchsorted(self._hashes, hash_value)
        if index < len(self._hashes) and self._hashes[index] == hash_value:
            return True
        return False

    def __repr__(self) -> str:
        return (f"SnipeSig(name={self._name}, ksize={self._ksize}, scale={self._scale}, "
                f"type={self._type}, num_hashes={len(self._hashes)})")

    def __str__(self) -> str:
        return self.__repr__()

    def __verify_snipe_signature(self, other: 'SnipeSig'):
        r"""
        Verify that the other object is a SnipeSig instance.

        Parameters:
            other (SnipeSig): The other signature to verify.

        Raises:
            ValueError: If the other object is not a SnipeSig instance.
        """
        if not isinstance(other, SnipeSig):
            msg = f"Provided sig ({type(other).__name__}) is not a SnipeSig instance."
            self.logger.error(msg)
            raise ValueError(msg)

    def __verify_matching_ksize_scale(self, other: 'SnipeSig'):
        r"""
        Verify that the ksize and scale match between two signatures.

        Parameters:
            other (SnipeSig): The other signature to compare.

        Raises:
            ValueError: If ksize or scale do not match.
        """
        if self._ksize != other.ksize:
            _e_msg = f"K-mer size does not match between the two signatures: {self._ksize} vs {other.ksize}."
            self.logger.error(_e_msg)
            raise ValueError(_e_msg)
        if self._scale != other.scale:
            _e_msg = f"Scale value does not match between the two signatures: {self._scale} vs {other.scale}."
            self.logger.error(_e_msg)
            raise ValueError(_e_msg)

    def _validate_abundance_operation(self, value: Union[int, None], operation: str):
        r"""
        Validate that the signature tracks abundance and that the provided value is a non-negative integer.

        Parameters:
            value (int or None): The abundance value to validate. Can be None for operations that don't require a value.
            operation (str): Description of the operation for logging purposes.

        Raises:
            ValueError: If the signature does not track abundance or if the value is invalid.
        """
        if not self._track_abundance and self.sigtype == SigType.SAMPLE:
            self.logger.error("Cannot %s: signature does not track abundance.", operation)
            raise ValueError("Signature does not track abundance.")

        if value is not None:
            if not isinstance(value, int) or value < 0:
                self.logger.error("%s requires a non-negative integer value.", operation.capitalize())
                raise ValueError(f"{operation.capitalize()} requires a non-negative integer value.")

    # Mask application method
    def _apply_mask(self, mask: np.ndarray):
        r"""
        Apply a boolean mask to the hashes and abundances arrays.
        Ensures that the sorted order is preserved.

        Parameters:
            mask (np.ndarray): Boolean array indicating which elements to keep.
        """
        self._hashes = self._hashes[mask]
        self._abundances = self._abundances[mask]

        # Verify that the hashes remain sorted
        if self._hashes.size > 1:
            if not np.all(self._hashes[:-1] <= self._hashes[1:]):
                self.logger.error("Hashes are not sorted after applying mask.")
                raise RuntimeError("Hashes are not sorted after applying mask.")
        self.logger.debug("Applied mask. Hashes remain sorted.")

    # Set operation methods
    def union_sigs(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Combine this signature with another by summing abundances where hashes overlap.

        Given two signatures \( A \) and \( B \) with hash sets \( H_A \) and \( H_B \),
        and their corresponding abundance functions \( a_A \) and \( a_B \), the union
        signature \( C \) is defined as follows:

        - **Hash Set**: 

        $$
        H_C = H_A \cup H_B
        $$

        - **Abundance Function**:

        $$
        a_C(h) =
        \begin{cases} 
            a_A(h) + a_B(h), & \text{if } h \in H_A \cap H_B \\
            a_A(h), & \text{if } h \in H_A \setminus H_B \\
            a_B(h), & \text{if } h \in H_B \setminus H_A
        \end{cases}
        $$
        """
        self.__verify_snipe_signature(other)
        self.__verify_matching_ksize_scale(other)

        self.logger.debug("Unioning signatures (including all unique hashes).")

        # Access internal arrays directly
        self_hashes = self._hashes
        self_abundances = self._abundances
        other_hashes = other._hashes
        other_abundances = other._abundances

        # Handle the case where 'other' does not track abundance
        if not other.track_abundance:
            self.logger.debug("Other signature does not track abundance. Setting abundances to 1.")
            other_abundances = np.ones_like(other_abundances, dtype=np.uint32)

        # Combine hashes and abundances
        combined_hashes = np.concatenate((self_hashes, other_hashes))
        combined_abundances = np.concatenate((self_abundances, other_abundances))

        # Use numpy's unique function with return_inverse to sum abundances efficiently
        unique_hashes, inverse_indices = np.unique(combined_hashes, return_inverse=True)
        summed_abundances = np.zeros_like(unique_hashes, dtype=np.uint32)

        # Sum abundances for duplicate hashes
        np.add.at(summed_abundances, inverse_indices, combined_abundances)

        # Handle potential overflow
        summed_abundances = np.minimum(summed_abundances, np.iinfo(np.uint32).max)

        self.logger.debug("Union operation completed. Total hashes: %d", len(unique_hashes))

        # Create a new SnipeSig instance
        return self.create_from_hashes_abundances(
            hashes=unique_hashes,
            abundances=summed_abundances,
            ksize=self._ksize,
            scale=self._scale,
            name=f"{self._name}_union_{other._name}",
            filename=None,
            enable_logging=self.logger.level <= logging.DEBUG
        )

    def _convert_to_sourmash_signature(self):
        r"""
        Convert the SnipeSig instance to a sourmash.signature.SourmashSignature object.

        Returns:
            sourmash.signature.SourmashSignature: A new sourmash.signature.SourmashSignature instance.
        """
        self.logger.debug("Converting SnipeSig to sourmash.signature.SourmashSignature.")

        mh = sourmash.minhash.MinHash(n=0, ksize=self._ksize, scaled=self._scale, track_abundance=self._track_abundance)
        if self._track_abundance:
            mh.set_abundances(dict(zip(self._hashes, self._abundances)))
        else:
            mh.add_many(self._hashes)
        self.sourmash_sig = sourmash.signature.SourmashSignature(mh, name=self._name, filename=self._filename)
        self.logger.debug("Conversion to sourmash.signature.SourmashSignature completed.")

    def export(self, path, force=False) -> None:
        r"""
        Export the signature to a file.

        Parameters:
            path (str): The path to save the signature to.
            force (bool): Flag to overwrite the file if it already exists.
        """
        self._convert_to_sourmash_signature()
        if path.endswith(".sig"):
            self.logger.debug("Exporting signature to a .sig file.")
            with open(str(path), "wb") as fp:
                sourmash.signature.save_signatures_to_json([self.sourmash_sig], fp)
        # sourmash.save_load.SaveSignatures_SigFile
                
        elif path.endswith(".zip"):
            if os.path.exists(path): 
                raise FileExistsError("Output file already exists.")
            try:
                with sourmash.save_load.SaveSignatures_ZipFile(path) as save_sigs:
                    save_sigs.add(self.sourmash_sig)
            except Exception as e:
                self.logger.error("Failed to export signatures to zip: %s", e)
                raise Exception(f"Failed to export signatures to zip: {e}") from e
        else:
            raise ValueError("Output file must be either a .sig or .zip file.")
        
            

    def export_to_string(self):
        r"""
        Export the signature to a JSON string.

        Returns:
            str: JSON string representation of the signature.
        """
        self._convert_to_sourmash_signature()
        return sourmash.signature.save_signatures_to_json([self.sourmash_sig]).decode('utf-8')

    def intersection_sigs(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Compute the intersection of the current signature with another signature.

        This method keeps only the hashes that are common to both signatures, and retains the abundances from self.

        **Mathematical Explanation**:

        Let \( A \) and \( B \) be two signatures with sets of hashes \( H_A \) and \( H_B \),
        and abundance functions \( a_A(h) \) and \( a_B(h) \), the intersection signature \( C \) has:

        - Hash set:
        $$
        H_C = H_A \cap H_B
        $$

        - Abundance function:
        $$
        a_C(h) = a_A(h), \quad \text{for } h \in H_C
        $$

        **Parameters**:
            - `other (SnipeSig)`: Another `SnipeSig` instance to intersect with.

        **Returns**:
            - `SnipeSig`: A new `SnipeSig` instance representing the intersection of the two signatures.

        **Raises**:
            - `ValueError`: If `ksize` or `scale` do not match between signatures.
        """
        self.__verify_snipe_signature(other)
        self.__verify_matching_ksize_scale(other)

        self.logger.debug("Intersecting signatures.")

        # Use numpy's intersect1d function
        common_hashes, self_indices, _ = np.intersect1d(
            self._hashes, other._hashes, assume_unique=True, return_indices=True
        )

        if common_hashes.size == 0:
            self.logger.debug("No common hashes found. Returning an empty signature.")
            return self.create_from_hashes_abundances(
                hashes=np.array([], dtype=np.uint64),
                abundances=np.array([], dtype=np.uint32),
                ksize=self._ksize,
                scale=self._scale,
                name=f"{self._name}_intersection_{other._name}",
                filename=None,
                enable_logging=self.logger.level <= logging.DEBUG
            )

        # Get the abundances from self
        common_abundances = self._abundances[self_indices]

        self.logger.debug("Intersection operation completed. Total common hashes: %d", len(common_hashes))

        # Create a new SnipeSig instance
        return self.create_from_hashes_abundances(
            hashes=common_hashes,
            abundances=common_abundances,
            ksize=self._ksize,
            scale=self._scale,
            name=f"{self._name}_intersection_{other._name}",
            filename=None,
            enable_logging=self.logger.level <= logging.DEBUG
        )

    def difference_sigs(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Compute the difference of the current signature with another signature.

        This method removes hashes that are present in the other signature from self,
        keeping the abundances from self.

        **Mathematical Explanation**:

        Let \( A \) and \( B \) be two signatures with sets of hashes \( H_A \) and \( H_B \),
        and abundance function \( a_A(h) \), the difference signature \( C \) has:

        - Hash set:
        $$
        H_C = H_A \setminus H_B
        $$

        - Abundance function:
        $$
        a_C(h) = a_A(h), \quad \text{for } h \in H_C
        $$

        **Parameters**:
            - `other (SnipeSig)`: Another `SnipeSig` instance to subtract from the current signature.

        **Returns**:
            - `SnipeSig`: A new `SnipeSig` instance representing the difference of the two signatures.

        **Raises**:
            - `ValueError`: If `ksize` or `scale` do not match between signatures.
            - `RuntimeError`: If zero hashes remain after difference.
        """
        self.__verify_snipe_signature(other)
        self.__verify_matching_ksize_scale(other)

        self.logger.debug("Differencing signatures.")

        # Use numpy's setdiff1d function
        diff_hashes = np.setdiff1d(self._hashes, other._hashes, assume_unique=True)

        if diff_hashes.size == 0:
            _e_msg = f"Difference operation resulted in zero hashes, which is not allowed for {self._name} and {other._name}."
            self.logger.warning(_e_msg)

        # Get the indices of the hashes in self
        mask = np.isin(self._hashes, diff_hashes, assume_unique=True)
        diff_abundances = self._abundances[mask]

        self.logger.debug("Difference operation completed. Remaining hashes: %d", len(diff_hashes))

        # Create a new SnipeSig instance
        return self.create_from_hashes_abundances(
            hashes=diff_hashes,
            abundances=diff_abundances,
            ksize=self._ksize,
            scale=self._scale,
            name=f"{self._name}_difference_{other._name}",
            filename=None,
            enable_logging=self.logger.level <= logging.DEBUG
        )

    def symmetric_difference_sigs(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Compute the symmetric difference of the current signature with another signature.

        This method retains hashes that are unique to each signature, with their respective abundances.

        **Mathematical Explanation**:

        Let \( A \) and \( B \) be two signatures with sets of hashes \( H_A \) and \( H_B \),
        and abundance functions \( a_A(h) \) and \( a_B(h) \), the symmetric difference signature \( C \) has:

        - Hash set:
        $$
        H_C = (H_A \setminus H_B) \cup (H_B \setminus H_A)
        $$

        - Abundance function:
        $$
        a_C(h) =
        \begin{cases}
        a_A(h), & \text{for } h \in H_A \setminus H_B \\
        a_B(h), & \text{for } h \in H_B \setminus H_A \\
        \end{cases}
        $$

        **Parameters**:
            - `other (SnipeSig)`: Another `SnipeSig` instance to compute the symmetric difference with.

        **Returns**:
            - `SnipeSig`: A new `SnipeSig` instance representing the symmetric difference of the two signatures.

        **Raises**:
            - `ValueError`: If `ksize` or `scale` do not match between signatures.
            - `RuntimeError`: If zero hashes remain after symmetric difference.
        """
        self.__verify_snipe_signature(other)
        self.__verify_matching_ksize_scale(other)

        self.logger.debug("Computing symmetric difference of signatures.")

        # Hashes unique to self and other
        unique_self_hashes = np.setdiff1d(self._hashes, other._hashes, assume_unique=True)
        unique_other_hashes = np.setdiff1d(other._hashes, self._hashes, assume_unique=True)

        # Abundances for unique hashes
        mask_self = np.isin(self._hashes, unique_self_hashes, assume_unique=True)
        unique_self_abundances = self._abundances[mask_self]

        mask_other = np.isin(other._hashes, unique_other_hashes, assume_unique=True)
        unique_other_abundances = other._abundances[mask_other]

        # Handle the case where 'other' does not track abundance
        if not other.track_abundance:
            self.logger.debug("Other signature does not track abundance. Setting abundances to 1.")
            unique_other_abundances = np.ones_like(unique_other_abundances, dtype=np.uint32)

        # Combine hashes and abundances
        combined_hashes = np.concatenate((unique_self_hashes, unique_other_hashes))
        combined_abundances = np.concatenate((unique_self_abundances, unique_other_abundances))

        if combined_hashes.size == 0:
            _e_msg = "Symmetric difference operation resulted in zero hashes, which is not allowed."
            self.logger.error(_e_msg)
            raise RuntimeError(_e_msg)

        # Sort combined hashes and abundances
        sorted_indices = np.argsort(combined_hashes)
        combined_hashes = combined_hashes[sorted_indices]
        combined_abundances = combined_abundances[sorted_indices]

        self.logger.debug("Symmetric difference operation completed. Total unique hashes: %d", len(combined_hashes))

        # Create a new SnipeSig instance
        return self.create_from_hashes_abundances(
            hashes=combined_hashes,
            abundances=combined_abundances,
            ksize=self._ksize,
            scale=self._scale,
            name=f"{self._name}_symmetric_difference_{other._name}",
            filename=None,
            enable_logging=self.logger.level <= logging.DEBUG
        )

    # Magic methods for union operations
    def __add__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the + operator.
        Includes all unique hashes from both signatures and sums their abundances where hashes overlap,
        returning a new signature.

        Returns:
            SnipeSig: Union of self and other.
        """
        return self.union_sigs(other)

    def __iadd__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the += operator.
        Includes all unique hashes from both signatures and sums their abundances where hashes overlap,
        modifying self in-place.

        Returns:
            SnipeSig: Updated self after addition.
        """
        union_sig = self.union_sigs(other)
        self._update_from_union(union_sig)
        return self

    def __or__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the | operator.
        Includes all unique hashes from both signatures and sums their abundances where hashes overlap,
        returning a new signature.

        Returns:
            SnipeSig: Union of self and other.
        """
        return self.union_sigs(other)

    def __ior__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the |= operator.
        Includes all unique hashes from both signatures and sums their abundances where hashes overlap,
        modifying self in-place.

        Returns:
            SnipeSig: Updated self after union.
        """
        union_sig = self.union_sigs(other)
        self._update_from_union(union_sig)
        return self

    def __sub__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the - operator.
        Removes hashes present in other from self, keeping abundances from self,
        returning a new signature.

        Returns:
            SnipeSig: Difference of self and other.
        """
        return self.difference_sigs(other)

    def __isub__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the -= operator.
        Removes hashes present in other from self, keeping abundances from self,
        modifying self in-place.

        Returns:
            SnipeSig: Updated self after difference.

        Raises:
            RuntimeError: If zero hashes remain after difference.
        """
        difference_sig = self.difference_sigs(other)
        self._update_from_union(difference_sig)
        return self

    def __xor__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the ^ operator.
        Keeps unique hashes from each signature with their respective abundances, returning a new signature.

        Returns:
            SnipeSig: Symmetric difference of self and other.
        """
        return self.symmetric_difference_sigs(other)

    def __ixor__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the ^= operator.
        Keeps unique hashes from each signature with their respective abundances, modifying self in-place.

        Returns:
            SnipeSig: Updated self after symmetric difference.

        Raises:
            RuntimeError: If zero hashes remain after symmetric difference.
        """
        symmetric_diff_sig = self.symmetric_difference_sigs(other)
        self._update_from_union(symmetric_diff_sig)
        return self

    def __and__(self, other: 'SnipeSig') -> 'SnipeSig':
        r"""
        Implements the & operator.
        Keeps common hashes and retains abundances from self only, returning a new signature.

        Returns:
            SnipeSig: Intersection of self and other.
        """
        return self.intersection_sigs(other)

    def _update_from_union(self, other: 'SnipeSig'):
        r"""
        Update self's hashes and abundances from another SnipeSig instance.

        Parameters:
            other (SnipeSig): The other SnipeSig instance to update from.
        """
        self._hashes = other.hashes
        self._abundances = other.abundances
        self._name = other.name
        self._filename = other.filename
        self._md5sum = other.md5sum
        self._track_abundance = other.track_abundance
        # No need to update ksize and scale since they are verified to match

    @classmethod
    def create_from_hashes_abundances(cls, hashes: np.ndarray, abundances: np.ndarray,
                                      ksize: int, scale: int, name: str = None,
                                      filename: str = None, enable_logging: bool = False, sig_type: SigType = SigType.SAMPLE) -> 'SnipeSig':
        """
        Internal method to create a SnipeSig instance from hashes and abundances.

        Parameters:
            hashes (np.ndarray): Array of hash values.
            abundances (np.ndarray): Array of abundance values corresponding to the hashes.
            ksize (int): K-mer size.
            scale (int): Scale value.
            name (str): Optional name for the signature.
            filename (str): Optional filename for the signature.
            sig_type (SigType): Type of the signature.
            enable_logging (bool): Flag to enable logging.

        Returns:
            SnipeSig: A new SnipeSig instance.
        """
        # Create a mock sourmash signature object
        mh = sourmash.minhash.MinHash(n=0, ksize=ksize, scaled=scale, track_abundance=True)
        mh.set_abundances(dict(zip(hashes, abundances)))
        sig = sourmash.signature.SourmashSignature(mh, name=name or "", filename=filename or "")
        return cls(sourmash_sig=sig, sig_type=sig_type, enable_logging=enable_logging)

    # Aggregation Operations
    @classmethod
    def sum_signatures(cls, signatures: List['SnipeSig'], name: str = "summed_signature",
                       filename: str = None, enable_logging: bool = False) -> 'SnipeSig':

        r"""
        Sum multiple SnipeSig instances by including all unique hashes and summing their abundances where hashes overlap.
        This method utilizes a heap-based multi-way merge algorithm for enhanced efficiency when handling thousands of signatures.

        $$
        \text{Sum}(A_1, A_2, \dots, A_n) = \bigcup_{i=1}^{n} A_i
        $$

        For each hash \( h \), its total abundance is:
        $$
        \text{abundance}(h) = \sum_{i=1}^{n} \text{abundance}_i(h)
        $$

        **Mathematical Explanation**:

        - **Union of Signatures**:
            The summation of signatures involves creating a union of all unique k-mers (hashes) present across the input signatures.

        - **Total Abundance Calculation**:
            For each unique hash \( h \), the total abundance is the sum of its abundances across all signatures where it appears.

        - **Algorithm Efficiency**:
            By using a min-heap to perform a multi-way merge of sorted hash arrays, the method ensures that each hash is processed in ascending order without the need to store all hashes in memory simultaneously.

        **Parameters**:
            - `signatures (List[SnipeSig])`: List of `SnipeSig` instances to sum.
            - `name (str)`: Optional name for the resulting signature.
            - `filename (str)`: Optional filename for the resulting signature.
            - `enable_logging (bool)`: Flag to enable detailed logging.

        **Returns**:
            - `SnipeSig`: A new `SnipeSig` instance representing the sum of the signatures.

        **Raises**:
            - `ValueError`: If the signatures list is empty or if `ksize`/`scale` do not match across signatures.
            - `RuntimeError`: If an error occurs during the summation process.
        """
        if not signatures:
            raise ValueError("No signatures provided for summation.")

        # Verify that all signatures have the same ksize, scale, and track_abundance
        first_sig = signatures[0]
        ksize = first_sig.ksize
        scale = first_sig.scale
        track_abundance = first_sig.track_abundance

        for sig in signatures[1:]:
            if sig.ksize != ksize or sig.scale != scale:
                raise ValueError("All signatures must have the same ksize and scale.")

        # Initialize iterators for each signature's hashes and abundances
        iterators = []
        for sig in signatures:
            it = iter(zip(sig.hashes, sig.abundances))
            try:
                first_hash, first_abundance = next(it)
                iterators.append((first_hash, first_abundance, it))
            except StopIteration:
                continue  # Skip empty signatures

        if not iterators:
            raise ValueError("All provided signatures are empty.")

        # Initialize the heap with the first element from each iterator
        heap = []
        for idx, (hash_val, abundance, it) in enumerate(iterators):
            heap.append((hash_val, abundance, idx))
        heapq.heapify(heap)

        # Prepare lists to collect the summed hashes and abundances
        summed_hashes = []
        summed_abundances = []

        while heap:
            current_hash, current_abundance, idx = heapq.heappop(heap)
            # Initialize total abundance for the current_hash
            total_abundance = current_abundance

            # Check if the next element in the heap has the same hash
            while heap and heap[0][0] == current_hash:
                _, abundance, same_idx = heapq.heappop(heap)
                total_abundance += abundance
                # Push the next element from the same iterator
                try:
                    next_hash, next_abundance = next(iterators[same_idx][2])
                    heapq.heappush(heap, (next_hash, next_abundance, same_idx))
                except StopIteration:
                    pass  # No more elements in this iterator

            # Append the summed hash and abundance
            summed_hashes.append(current_hash)
            summed_abundances.append(total_abundance)

            # Push the next element from the current iterator
            try:
                next_hash, next_abundance = next(iterators[idx][2])
                heapq.heappush(heap, (next_hash, next_abundance, idx))
            except StopIteration:
                pass  # No more elements in this iterator

        # Convert the results to NumPy arrays for efficient storage and processing
        summed_hashes = np.array(summed_hashes, dtype=np.uint64)
        summed_abundances = np.array(summed_abundances, dtype=np.uint32)

        # Handle potential overflow by capping at the maximum value of uint32
        summed_abundances = np.minimum(summed_abundances, np.iinfo(np.uint32).max)

        # Create a new SnipeSig instance from the summed hashes and abundances
        summed_signature = cls.create_from_hashes_abundances(
            hashes=summed_hashes,
            abundances=summed_abundances,
            ksize=ksize,
            scale=scale,
            name=name,
            filename=filename,
            enable_logging=enable_logging
        )

        return summed_signature
    
    @staticmethod
    def get_unique_signatures(signatures: Dict[str, 'SnipeSig']) -> Dict[str, 'SnipeSig']:
        """
        Extract unique signatures from a dictionary of SnipeSig instances.
        
        For each signature, the unique_sig contains only the hashes that do not overlap with any other signature.
        
        Parameters:
            signatures (Dict[str, SnipeSig]): A dictionary mapping signature names to SnipeSig instances.
        
        Returns:
            Dict[str, SnipeSig]: A dictionary mapping signature names to their unique SnipeSig instances.
        
        Raises:
            ValueError: If the input dictionary is empty or if signatures have mismatched ksize/scale.
        """
        if not signatures:
            raise ValueError("The input signatures dictionary is empty.")
        
        # Extract ksize and scale from the first signature
        first_name, first_sig = next(iter(signatures.items()))
        ksize = first_sig.ksize
        scale = first_sig.scale
        
        # Verify that all signatures have the same ksize and scale
        for name, sig in signatures.items():
            if sig.ksize != ksize or sig.scale != scale:
                raise ValueError(f"Signature '{name}' has mismatched ksize or scale.")
        
        # Aggregate all hashes from all signatures
        all_hashes = np.concatenate([sig.hashes for sig in signatures.values()])
        
        # Count the occurrences of each hash
        unique_hashes, counts = np.unique(all_hashes, return_counts=True)
        
        # Identify hashes that are unique across all signatures (count == 1)
        unique_across_all = unique_hashes[counts == 1]
        
        # Convert to a set for faster membership testing
        unique_set = set(unique_across_all)
        
        unique_signatures = {}
        
        for name, sig in signatures.items():
            # Find hashes in the current signature that are unique across all signatures
            mask_unique = np.isin(sig.hashes, list(unique_set))
            
            # Extract unique hashes and their abundances
            unique_hashes_sig = sig.hashes[mask_unique]
            unique_abundances_sig = sig.abundances[mask_unique]
            
            # Create a new SnipeSig instance with the unique hashes and abundances
            unique_sig = SnipeSig.create_from_hashes_abundances(
                hashes=unique_hashes_sig,
                abundances=unique_abundances_sig,
                ksize=ksize,
                scale=scale,
                name=f"{name}_unique",
                filename=None,
                enable_logging=False,  # Set to True if you want logging for the new signatures
                sig_type=SigType.SAMPLE  # Adjust sig_type as needed
            )
            
            unique_signatures[name] = unique_sig
        
        return unique_signatures
    

    @classmethod
    def common_hashes(cls, signatures: List['SnipeSig'], name: str = "common_hashes_signature",
                      filename: str = None, enable_logging: bool = False) -> 'SnipeSig':
        r"""
        Compute the intersection of multiple SnipeSig instances, returning a new SnipeSig containing
        only the hashes present in all signatures, with abundances set to the minimum abundance across signatures.
        
        This method uses a heap-based multi-way merge algorithm for efficient computation,
        especially when handling a large number of signatures with sorted hashes.
        
        **Mathematical Explanation**:
        
        Given signatures \( A_1, A_2, \dots, A_n \) with hash sets \( H_1, H_2, \dots, H_n \),
        the intersection signature \( C \) has:
        
        - Hash set:
        $$
        H_C = \bigcap_{i=1}^{n} H_i
        $$
        
        - Abundance function:
        $$
        a_C(h) = \min_{i=1}^{n} a_i(h), \quad \text{for } h \in H_C
        $$
        
        **Parameters**:
            - `signatures (List[SnipeSig])`: List of `SnipeSig` instances to compute the intersection.
            - `name (str)`: Optional name for the resulting signature.
            - `filename (str)`: Optional filename for the resulting signature.
            - `enable_logging (bool)`: Flag to enable detailed logging.
        
        **Returns**:
            - `SnipeSig`: A new `SnipeSig` instance representing the intersection of the signatures.
        
        **Raises**:
            - `ValueError`: If the signatures list is empty or if `ksize`/`scale` do not match across signatures.
        """
        if not signatures:
            raise ValueError("No signatures provided for intersection.")
        
        # Verify that all signatures have the same ksize and scale
        first_sig = signatures[0]
        ksize = first_sig.ksize
        scale = first_sig.scale
        for sig in signatures[1:]:
            if sig.ksize != ksize or sig.scale != scale:
                raise ValueError("All signatures must have the same ksize and scale.")
        
        num_signatures = len(signatures)
        iterators = []
        for sig in signatures:
            it = iter(zip(sig.hashes, sig.abundances))
            try:
                first_hash, first_abundance = next(it)
                iterators.append((first_hash, first_abundance, it))
            except StopIteration:
                # One of the signatures is empty; intersection is empty
                return cls.create_from_hashes_abundances(
                    hashes=np.array([], dtype=np.uint64),
                    abundances=np.array([], dtype=np.uint32),
                    ksize=ksize,
                    scale=scale,
                    name=name,
                    filename=filename,
                    enable_logging=enable_logging
                )
        
        # Initialize the heap with the first element from each iterator
        heap = []
        for idx, (hash_val, abundance, it) in enumerate(iterators):
            heap.append((hash_val, abundance, idx))
        heapq.heapify(heap)
        
        common_hashes = []
        common_abundances = []
        
        while heap:
            # Pop all entries with the smallest hash
            current_hash, current_abundance, idx = heapq.heappop(heap)
            same_hash_entries = [(current_hash, current_abundance, idx)]
            
            # Collect all entries in the heap that have the same current_hash
            while heap and heap[0][0] == current_hash:
                h, a, i = heapq.heappop(heap)
                same_hash_entries.append((h, a, i))
            
            if len(same_hash_entries) == num_signatures:
                # The current_hash is present in all signatures
                # Take the minimum abundance across signatures
                min_abundance = min(entry[1] for entry in same_hash_entries)
                common_hashes.append(current_hash)
                common_abundances.append(min_abundance)
            
            # Push the next element from each iterator that had the current_hash
            for entry in same_hash_entries:
                h, a, i = entry
                try:
                    next_hash, next_abundance = next(iterators[i][2])
                    heapq.heappush(heap, (next_hash, next_abundance, i))
                except StopIteration:
                    pass  # Iterator exhausted
        
        # Convert the results to NumPy arrays
        if not common_hashes:
            # No common hashes found
            unique_hashes = np.array([], dtype=np.uint64)
            unique_abundances = np.array([], dtype=np.uint32)
        else:
            unique_hashes = np.array(common_hashes, dtype=np.uint64)
            unique_abundances = np.array(common_abundances, dtype=np.uint32)
        
        # Create a new SnipeSig instance from the common hashes and abundances
        common_signature = cls.create_from_hashes_abundances(
            hashes=unique_hashes,
            abundances=unique_abundances,
            ksize=ksize,
            scale=scale,
            name=name,
            filename=filename,
            enable_logging=enable_logging
        )
        
        return common_signature

    def copy(self) -> 'SnipeSig':
        r"""
        Create a copy of the current SnipeSig instance.

        Returns:
            SnipeSig: A new instance that is a copy of self.
        """
        return SnipeSig(sourmash_sig=self.export_to_string(), sig_type=self.sigtype, enable_logging=self.logger.level <= logging.DEBUG)

    # Implement the __radd__ method to support sum()
    def __radd__(self, other: Union[int, 'SnipeSig']) -> 'SnipeSig':
        r"""
        Implements the right-hand + operator to support sum().

        Returns:
            SnipeSig: Union of self and other.
        """
        return self.__radd_sum__(other)

    # Override the __sum__ method
    def __radd_sum__(self, other: Union[int, 'SnipeSig']) -> 'SnipeSig':
        r"""
        Internal helper method to support the sum() function.

        Parameters:
            other (int or SnipeSig): The other object to add. If other is 0, return self.

        Returns:
            SnipeSig: The result of the addition.
        """
        if other == 0:
            return self
        if not isinstance(other, SnipeSig):
            raise TypeError(f"Unsupported operand type(s) for +: 'SnipeSig' and '{type(other).__name__}'")
        return self.union_sigs(other)

    def reset_abundance(self, new_abundance: int = 1):
        r"""
        Reset all abundances to a specified value.

        This method sets the abundance of every hash in the signature to the specified `new_abundance` value.

        **Mathematical Explanation**:

        For each hash \( h \) in the signature, the abundance function is updated to:
        $$
        a(h) = \text{new\_abundance}
        $$

        **Parameters**:
            - `new_abundance (int)`: The new abundance value to set for all hashes. Default is 1.

        **Raises**:
            - `ValueError`: If the signature does not track abundance or if `new_abundance` is invalid.
        """

        self._validate_abundance_operation(new_abundance, "reset abundance")

        self._abundances[:] = new_abundance
        self.track_abundance = True
        self.logger.debug("Reset all abundances to %d.", new_abundance)

    def keep_min_abundance(self, min_abundance: int):
        r"""
        Keep only hashes with abundances greater than or equal to a minimum threshold.

        This method removes hashes whose abundances are less than the specified `min_abundance`.

        **Mathematical Explanation**:

        The updated hash set \( H' \) is:
        $$
        H' = \{ h \in H \mid a(h) \geq \text{min\_abundance} \}
        $$

        **Parameters**:
            - `min_abundance (int)`: The minimum abundance threshold.

        **Raises**:
            - `ValueError`: If the signature does not track abundance or if `min_abundance` is invalid.
        """
        self._validate_abundance_operation(min_abundance, "keep minimum abundance")

        mask = self._abundances >= min_abundance
        self._apply_mask(mask)
        self.logger.debug("Kept hashes with abundance >= %d.", min_abundance)

    def keep_max_abundance(self, max_abundance: int):
        r"""
        Keep only hashes with abundances less than or equal to a maximum threshold.

        This method removes hashes whose abundances are greater than the specified `max_abundance`.

        **Mathematical Explanation**:

        The updated hash set \( H' \) is:
        $$
        H' = \{ h \in H \mid a(h) \leq \text{max\_abundance} \}
        $$

        **Parameters**:
            - `max_abundance (int)`: The maximum abundance threshold.

        **Raises**:
            - `ValueError`: If the signature does not track abundance or if `max_abundance` is invalid.
        """
        self._validate_abundance_operation(max_abundance, "keep maximum abundance")

        mask = self._abundances <= max_abundance
        self._apply_mask(mask)
        self.logger.debug("Kept hashes with abundance <= %d.", max_abundance)

    def trim_below_median(self):
        r"""
        Trim hashes with abundances below the median abundance.

        This method removes all hashes whose abundances are less than the median abundance of the signature.

        **Mathematical Explanation**:

        Let \\( m \\) be the median of \\( \\{ a(h) \mid h \in H \\} \\).
        The updated hash set \\( H' \\) is:

        $$
        H' = \\{ h \in H \mid a(h) \geq m \\}
        $$

        **Raises**:
            - `ValueError`: If the signature does not track abundance.
        """

        self._validate_abundance_operation(None, "trim below median")

        if len(self._abundances) == 0:
            self.logger.debug("No hashes to trim based on median abundance.")
            return

        median = np.median(self._abundances)
        mask = self._abundances >= median
        self._apply_mask(mask)
        self.logger.debug("Trimmed hashes with abundance below median (%f).", median)

    def count_singletons(self) -> int:
        r"""
        Return the number of hashes with abundance equal to 1.

        Returns:
            int: Number of singletons.

        Raises:
            ValueError: If the signature does not track abundance.
        """
        self._validate_abundance_operation(None, "count singletons")

        count = np.sum(self._abundances == 1)
        self.logger.debug("Number of singletons (abundance == 1): %d", count)
        return int(count)

    def trim_singletons(self):
        r"""
        Remove hashes with abundance equal to 1.

        This method removes all hashes that are singletons (abundance equals 1).

        **Mathematical Explanation**:

        The updated hash set \( H' \) is:
        $$
        H' = \{ h \in H \mid a(h) \neq 1 \}
        $$

        **Raises**:
            - `ValueError`: If the signature does not track abundance.
        """
        self._validate_abundance_operation(None, "trim singletons")

        mask = self._abundances != 1
        self.logger.debug("Trimming %d hashes with abundance equal to 1.", np.sum(~mask))
        self._apply_mask(mask)
        self.logger.debug("Size after trimming singletons: %d", len(self._hashes)) 
                          
    # New Properties Implemented as per Request

    @property
    def total_abundance(self) -> int:
        r"""
        Return the total abundance (sum of all abundances).

        Returns:
            int: Total abundance.
        """
        self._validate_abundance_operation(None, "calculate total abundance")

        total = int(np.sum(self._abundances))
        self.logger.debug("Total abundance: %d", total)
        return total

    @property
    def mean_abundance(self) -> float:
        r"""
        Return the mean (average) abundance.

        Returns:
            float: Mean abundance.
        """
        self._validate_abundance_operation(None, "calculate mean abundance")

        if len(self._abundances) == 0:
            self.logger.debug("No abundances to calculate mean.")
            return 0.0

        mean = float(np.mean(self._abundances))  # Changed to float
        self.logger.debug("Mean abundance: %f", mean)
        return mean

    @property
    def get_sample_stats(self) -> dict:
        r"""
        Retrieve statistical information about the signature.

        This property computes and returns a dictionary containing various statistics of the signature, such as total abundance, mean and median abundances, number of singletons, and total number of hashes.

        **Returns**:
            - `dict`: A dictionary containing sample statistics:
                - `total_abundance`: Sum of abundances.
                - `mean_abundance`: Mean abundance.
                - `median_abundance`: Median abundance.
                - `num_singletons`: Number of hashes with abundance equal to 1.
                - `num_hashes`: Total number of hashes.
                - `ksize`: K-mer size.
                - `scale`: Scale value.
                - `name`: Name of the signature.
                - `filename`: Filename of the signature.
        """
        
        # if self.sigtype != SigType.SAMPLE then don't return abundance stats
        
        stats = {
            "num_hashes": len(self._hashes),
            "ksize": self._ksize,
            "scale": self._scale,
            "name": self._name,
            "filename": self._filename
        }
        
        if self.sigtype != SigType.SAMPLE:
            stats["total_abundance"] = None
            stats["mean_abundance"] = None
            stats["median_abundance"] = None
            stats["num_singletons"] = None
        else:
            stats["total_abundance"] = self.total_abundance
            stats["mean_abundance"] = self.mean_abundance
            stats["median_abundance"] = self.median_abundance
            stats["num_singletons"] = self.count_singletons()
            
        return stats

    @property
    def median_abundance(self) -> float:
        r"""
        Return the median abundance.

        Returns:
            float: Median abundance.

        Raises:
            ValueError: If the signature does not track abundance.
        """
        self._validate_abundance_operation(None, "calculate median abundance")

        if len(self._abundances) == 0:
            self.logger.debug("No abundances to calculate median.")
            return 0.0

        median = float(np.median(self._abundances))  # Changed to float
        self.logger.debug("Median abundance: %f", median)
        return median
