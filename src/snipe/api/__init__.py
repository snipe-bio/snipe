from typing import Optional, Dict, Any, List, Union, Iterator
from enum import Enum, auto
import logging
import heapq
import sourmash
from sourmash.signature import SourmashSignature
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import re
import concurrent

# Configure the root logger to CRITICAL to suppress unwanted logs by default
logging.basicConfig(level=logging.CRITICAL)

class SigType(Enum):
    """
    Enumeration representing different types of signatures.
    """
    SAMPLE = auto()
    GENOME = auto()
    AMPLICON = auto()

class SnipeSig:
    """
    A class to handle Sourmash signatures with additional functionalities
    such as customized set operations and abundance management.
    """

    def _try_load_from_json(self, sourmashSig: str) -> Union[SourmashSignature, None]:
        r"""
        Attempt to load sourmash signature from JSON string.

        Parameters:
            sourmashSig (str): JSON string representing a sourmash signature.

        Returns:
            SourmashSignature or None if loading fails.
        """
        try:
            self.logger.debug("Trying to load sourmash signature from JSON.")
            return sourmash.load_one_signature_from_json(sourmashSig)
        except ValueError as e:
            if "expected to load exactly one signature" in str(e):
                self.logger.error("More than one signature found in the JSON input.")
                raise ValueError("Only one sketch per sourmash signature is allowed") from e
            self.logger.debug("Loading from JSON failed. Proceeding to file loading.")
            return None  # Return None to indicate failure

    def _try_load_from_file(self, sourmashSig: str) -> Union[SourmashSignature, None]:
        r"""
        Attempt to load sourmash signature from a file.

        Parameters:
            sourmashSig (str): File path to a sourmash signature.

        Returns:
            SourmashSignature or None if loading fails.
        """
        try:
            self.logger.debug("Trying to load sourmash signature from file.")
            return sourmash.load_file_as_signatures(sourmashSig)
        except ValueError as e:
            if "expected to load exactly one signature" in str(e):
                self.logger.error("More than one signature found in the file.")
                raise ValueError("Only one sketch per sourmash signature is allowed") from e
            self.logger.exception("Failed to load the sourmash signature from the file.")
            return None  # Return None to indicate failure

    def __init__(self, *, sourmashSig: Union[str, SourmashSignature, sourmash.signature.FrozenSourmashSignature],
                 ksize: int = 51, scale: int = 10000, sigType=SigType.SAMPLE, enable_logging: bool = False, **kwargs):
        r"""
        Initialize the SnipeSig with a sourmash signature object or a path to a signature.

        Parameters:
            sourmashSig (str or SourmashSignature): A path to a signature file or a signature object.
            ksize (int): K-mer size.
            scale (int): Scale value.
            sigType (SigType): Type of the signature.
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
        self.logger.debug("Initializing SnipeSig with sourmashSig: %s", sourmashSig)

        self._scale = scale
        self._ksize = ksize
        self._md5sum = None
        self._hashes = np.array([], dtype=np.uint64)
        self._abundances = np.array([], dtype=np.uint32)
        self._type = sigType
        self._name = None
        self._filename = None
        self._track_abundance = False

        _sourmash_sig = None

        # Type checking for sourmashSig
        if not isinstance(sourmashSig, (str, SourmashSignature, sourmash.signature.FrozenSourmashSignature)):
            self.logger.error("Invalid type for sourmashSig: %s", type(sourmashSig).__name__)
            raise TypeError(f"sourmashSig must be a string, SourmashSignature, or FrozenSourmashSignature, got {type(sourmashSig).__name__}")

        # Case 1: If sourmashSig is already a valid signature object
        if isinstance(sourmashSig, (sourmash.signature.FrozenSourmashSignature, SourmashSignature)):
            self.logger.debug("Loaded sourmash signature from object.")
            _sourmash_sig = sourmashSig

        # Case 2: If sourmashSig is a string, try to load as JSON or a file
        elif isinstance(sourmashSig, str):
            self.logger.debug("Attempting to load sourmash signature from string input.")

            # First, try loading from JSON
            _sourmash_sig = self._try_load_from_json(sourmashSig)

            # If JSON loading fails, try loading from file
            if not _sourmash_sig:
                _sourmash_sig = self._try_load_from_file(sourmashSig)

            # If both attempts fail, raise an error
            if not _sourmash_sig:
                self.logger.error("Failed to load sourmash signature from the provided string.")
                raise ValueError("An unexpected error occurred while loading the sourmash signature.")

        # Extract properties from the loaded signature
        self._ksize = _sourmash_sig.minhash.ksize
        self._scale = _sourmash_sig.minhash.scaled
        self._md5sum = _sourmash_sig.md5sum()
        self._name = _sourmash_sig.name
        self._filename = _sourmash_sig.filename
        self._track_abundance = _sourmash_sig.minhash.track_abundance

        # If the signature does not track abundance, assume abundance of 1 for all hashes
        if not self._track_abundance:
            self.logger.debug("Signature does not track abundance. Setting all abundances to 1.")
            self._abundances = np.ones(len(_sourmash_sig.minhash.hashes), dtype=np.uint32)
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
            "sigtype": self._type.name,
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
                f"type={self._type.name}, num_hashes={len(self._hashes)})")

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
            _e_msg = "Only SnipeSig objects can be used for this operation."
            self.logger.error(_e_msg)
            raise ValueError(_e_msg)

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
        if not self._track_abundance:
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
        Convert the SnipeSig instance to a SourmashSignature object.

        Returns:
            SourmashSignature: A new SourmashSignature instance.
        """
        self.logger.debug("Converting SnipeSig to SourmashSignature.")

        mh = sourmash.minhash.MinHash(n=0, ksize=self._ksize, scaled=self._scale, track_abundance=self._track_abundance)
        mh.set_abundances(dict(zip(self._hashes, self._abundances)))
        self.sourmash_sig = sourmash.signature.SourmashSignature(mh, name=self._name, filename=self._filename)
        self.logger.debug("Conversion to SourmashSignature completed.")

    def export(self, path) -> None:
        r"""
        Export the signature to a file.

        Parameters:
            path (str): The path to save the signature to.
        """
        self._convert_to_sourmash_signature()
        with open(str(path), "wb") as fp:
            sourmash.signature.save_signatures_to_json([self.sourmash_sig], fp)

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
            _e_msg = "Difference operation resulted in zero hashes, which is not allowed."
            self.logger.error(_e_msg)
            raise RuntimeError(_e_msg)

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
                                      filename: str = None, enable_logging: bool = False) -> 'SnipeSig':
        """
        Internal method to create a SnipeSig instance from hashes and abundances.

        Parameters:
            hashes (np.ndarray): Array of hash values.
            abundances (np.ndarray): Array of abundance values corresponding to the hashes.
            ksize (int): K-mer size.
            scale (int): Scale value.
            name (str): Optional name for the signature.
            filename (str): Optional filename for the signature.
            enable_logging (bool): Flag to enable logging.

        Returns:
            SnipeSig: A new SnipeSig instance.
        """
        # Create a mock sourmash signature object
        mh = sourmash.minhash.MinHash(n=0, ksize=ksize, scaled=scale, track_abundance=True)
        mh.set_abundances(dict(zip(hashes, abundances)))
        sig = sourmash.signature.SourmashSignature(mh, name=name or "combined_signature", filename=filename or "")
        return cls(sourmashSig=sig, sigType=SigType.SAMPLE, enable_logging=enable_logging)

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
            if sig.track_abundance != track_abundance:
                raise ValueError("All signatures must have the same track_abundance setting.")

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

    def copy(self) -> 'SnipeSig':
        r"""
        Create a copy of the current SnipeSig instance.

        Returns:
            SnipeSig: A new instance that is a copy of self.
        """
        return SnipeSig(sourmashSig=self.export_to_string(), sigType=self.sigtype, enable_logging=self.logger.level <= logging.DEBUG)

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
        self._apply_mask(mask)
        self.logger.debug("Trimmed hashes with abundance equal to 1.")

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

        mean = int(np.mean(self._abundances))
        self.logger.debug("Mean abundance: %d", mean)
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
        stats = {
            "total_abundance": self.total_abundance,
            "mean_abundance": self.mean_abundance,
            "median_abundance": self.median_abundance,
            "num_singletons": self.count_singletons(),
            "num_hashes": len(self._hashes),
            "ksize": self._ksize,
            "scale": self._scale,
            "name": self._name,
            "filename": self._filename,
        }
        return stats

    @property
    def median_abundance(self) -> int:
        r"""
        Return the median abundance.

        Returns:
            int: Median abundance.

        Raises:
            ValueError: If the signature does not track abundance.

        """
        self._validate_abundance_operation(None, "calculate median abundance")

        if len(self._abundances) == 0:
            self.logger.debug("No abundances to calculate median.")
            return 0.0

        median = int(np.median(self._abundances))
        self.logger.debug("Median abundance: %d", median)
        return median

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

        # Verify signature types
        if sample_sig.sigtype != SigType.SAMPLE:
            self.logger.error("Invalid signature type for sample_sig: %s", sample_sig.sigtype.name)
            raise ValueError(f"sample_sig must be of type {SigType.SAMPLE.name}, got {sample_sig.sigtype.name}")

        if reference_sig.sigtype != SigType.GENOME:
            self.logger.error("Invalid signature type for reference_sig: %s", reference_sig.sigtype.name)
            raise ValueError(f"reference_sig must be of type {SigType.GENOME.name}, got {reference_sig.sigtype.name}")

        if amplicon_sig is not None and amplicon_sig.sigtype != SigType.AMPLICON:
            self.logger.error("Invalid signature type for amplicon_sig: %s", amplicon_sig.sigtype.name)
            raise ValueError(f"amplicon_sig must be of type {SigType.AMPLICON.name}, got {amplicon_sig.sigtype.name}")

        if chr_name_to_sig is not None:
            self.flag_ychr_in_reference = flag_ychr_in_reference
            self.logger.debug("Chromosome specific signatures provided.")
            self.flag_activate_sex_metrics = True
            self.chr_name_to_sig = chr_name_to_sig
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
        self._calculate_stats()
    
    def _create_chr_specific_sigs_if_needed(self):
        # process the chromosome specific kmers to make sure no common kmers between them
        
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
