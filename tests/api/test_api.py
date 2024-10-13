import json
import logging
import os
import tempfile
import unittest

import numpy as np
import sourmash

from snipe.api import SnipeSig
from snipe.api.enums import SigType

def sample_signature_json():
    """
    Fixture to provide a sample sourmash signature in JSON format.
    """
    return json.dumps([
        {
            "class": "sourmash_signature",
            "email": "",
            "hash_function": "0.murmur64",
            "filename": "-",
            "name": "test_signature",
            "license": "CC0",
            "signatures": [
                {
                    "num": 0,
                    "ksize": 31,
                    "seed": 42,
                    "max_hash": 18446744073709551615,
                    "mins": [10, 20, 30, 40, 50],
                    "md5sum": "d9be84d0b05d39d231702887169b69a7",
                    "abundances": [1, 2, 3, 4, 5],
                    "molecule": "dna"
                }
            ],
            "version": 0.4
        }
    ])

def sample_signature_json_2():
    """
    Fixture to provide another sample sourmash signature in JSON format for testing set operations.
    """
    return json.dumps([
        {
            "class": "sourmash_signature",
            "email": "",
            "hash_function": "0.murmur64",
            "filename": "-",
            "name": "test_signature_2",
            "license": "CC0",
            "signatures": [
                {
                    "num": 0,
                    "ksize": 31,
                    "seed": 42,
                    "max_hash": 18446744073709551615,
                    "mins": [30, 40, 50, 60, 70],
                    "md5sum": "abcdef1234567890abcdef1234567890",
                    "abundances": [3, 4, 5, 6, 7],
                    "molecule": "dna"
                }
            ],
            "version": 0.4
        }
    ])

class TestSnipeSig(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Load sample signature JSON and parse into SourmashSignature object
        self.sample_json = sample_signature_json()
        self.sample_sig = sourmash.load_one_signature_from_json(self.sample_json)

        self.sample_json_2 = sample_signature_json_2()
        self.sample_sig_2 = sourmash.load_one_signature_from_json(self.sample_json_2)

        # Create SnipeSig instances
        self.snipe_sig = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        self.snipe_sig_2 = SnipeSig(
            sourmash_sig=self.sample_sig_2,
            enable_logging=False
        )

    def test_init_from_signature_object(self):
        """
        Test initialization of SnipeSig from a SourmashSignature object.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        self.assertEqual(snipe.name, "test_signature")
        self.assertEqual(snipe.ksize, 31)
        self.assertEqual(snipe.scale, 1)
        self.assertEqual(snipe.sigtype, SigType.SAMPLE)
        self.assertTrue(snipe.track_abundance)
        self.assertEqual(len(snipe), 5)
        expected_hashes = np.array([10, 20, 30, 40, 50], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(snipe.hashes, expected_hashes)
        np.testing.assert_array_equal(snipe.abundances, expected_abundances)

    def test_init_from_json_string(self):
        """
        Test initialization of SnipeSig from a JSON string.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_json,
            enable_logging=False
        )
        self.assertEqual(snipe.name, "test_signature")
        self.assertEqual(snipe.ksize, 31)
        self.assertEqual(snipe.scale, 1)
        self.assertEqual(snipe.sigtype, SigType.SAMPLE)
        self.assertTrue(snipe.track_abundance)
        self.assertEqual(len(snipe), 5)
        expected_hashes = np.array([10, 20, 30, 40, 50], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(snipe.hashes, expected_hashes)
        np.testing.assert_array_equal(snipe.abundances, expected_abundances)

    def test_init_from_file(self):
        """
        Test initialization of SnipeSig from a file containing a JSON signature.
        """
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            tmp_file.write(self.sample_json)
            tmp_file_path = tmp_file.name
        try:
            snipe = SnipeSig(
                sourmash_sig=tmp_file_path,
                enable_logging=False
            )
            self.assertEqual(snipe.name, "test_signature")
            self.assertEqual(snipe.ksize, 31)
            self.assertEqual(snipe.scale, 1)
            self.assertEqual(snipe.sigtype, SigType.SAMPLE)
            self.assertTrue(snipe.track_abundance)
            self.assertEqual(len(snipe), 5)
            expected_hashes = np.array([10, 20, 30, 40, 50], dtype=np.uint64)
            expected_abundances = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
            np.testing.assert_array_equal(snipe.hashes, expected_hashes)
            np.testing.assert_array_equal(snipe.abundances, expected_abundances)
        finally:
            os.remove(tmp_file_path)

    def test_init_invalid_input_type(self):
        """
        Test initialization with an invalid input type for sourmash_sig.
        """
        with self.assertRaises(TypeError):
            SnipeSig(
                sourmash_sig=12345,  # Invalid type
                enable_logging=False
            )

    def test_init_invalid_json(self):
        """
        Test initialization with an invalid JSON string.
        """
        invalid_json = '{"invalid": "json"}'
        with self.assertRaises(ValueError):
            SnipeSig(
                sourmash_sig=invalid_json,
                enable_logging=False
            )

    def test_properties(self):
        """
        Test access to various properties of the SnipeSig instance.
        """
        self.assertTrue(np.array_equal(
            self.snipe_sig.hashes,
            np.array([10, 20, 30, 40, 50], dtype=np.uint64)
        ))
        self.assertTrue(np.array_equal(
            self.snipe_sig.abundances,
            np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        ))
        self.assertEqual(self.snipe_sig.md5sum, "d9be84d0b05d39d231702887169b69a7")
        self.assertEqual(self.snipe_sig.ksize, 31)
        self.assertEqual(self.snipe_sig.scale, 1)
        self.assertEqual(self.snipe_sig.name, "test_signature")
        self.assertEqual(self.snipe_sig.filename, "-")
        self.assertEqual(self.snipe_sig.sigtype, SigType.SAMPLE)
        self.assertTrue(self.snipe_sig.track_abundance)

    def test_len(self):
        """
        Test the __len__ method to ensure it returns the correct number of hashes.
        """
        self.assertEqual(len(self.snipe_sig), 5)
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        self.assertEqual(len(empty_sig), 0)

    def test_iter(self):
        """
        Test the __iter__ method to ensure it correctly iterates over hashes and abundances.
        """
        expected = list(zip([10, 20, 30, 40, 50], [1, 2, 3, 4, 5]))
        self.assertEqual(list(iter(self.snipe_sig)), expected)

    def test_contains(self):
        """
        Test the __contains__ method to check if specific hashes are present.
        """
        self.assertIn(10, self.snipe_sig)
        self.assertIn(50, self.snipe_sig)
        self.assertNotIn(60, self.snipe_sig)

    def test_repr_str(self):
        """
        Test the __repr__ and __str__ methods for correct string representations.
        """
        expected_repr = (
            "SnipeSig(name=test_signature, ksize=31, scale=1, "
            "type=SAMPLE, num_hashes=5)"
        )
        self.assertEqual(repr(self.snipe_sig), expected_repr)
        self.assertEqual(str(self.snipe_sig), expected_repr)

    def test_union_sigs(self):
        """
        Test the union_sigs method to ensure correct union of two signatures.
        """
        union_sig = self.snipe_sig.union_sigs(self.snipe_sig_2)
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 10, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(union_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(union_sig.abundances, expected_abundances)
        self.assertEqual(union_sig.name, "test_signature_union_test_signature_2")

    def test_intersection_sigs(self):
        """
        Test the intersection_sigs method to ensure correct intersection of two signatures.
        """
        intersection_sig = self.snipe_sig.intersection_sigs(self.snipe_sig_2)
        expected_hashes = np.array([30, 40, 50], dtype=np.uint64)
        expected_abundances = np.array([3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(intersection_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(intersection_sig.abundances, expected_abundances)
        self.assertEqual(intersection_sig.name, "test_signature_intersection_test_signature_2")

    def test_difference_sigs(self):
        """
        Test the difference_sigs method to ensure correct difference of two signatures.
        """
        difference_sig = self.snipe_sig.difference_sigs(self.snipe_sig_2)
        expected_hashes = np.array([10, 20], dtype=np.uint64)
        expected_abundances = np.array([1, 2], dtype=np.uint32)
        np.testing.assert_array_equal(difference_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(difference_sig.abundances, expected_abundances)
        self.assertEqual(difference_sig.name, "test_signature_difference_test_signature_2")

    def test_difference_sigs_result_zero(self):
        """
        Test the difference_sigs method when the result is an empty signature.
        """
        sig1 = self.snipe_sig.intersection_sigs(self.snipe_sig_2)
        # Subtract the second signature again to remove all hashes
        sig3 = sig1.difference_sigs(self.snipe_sig_2)
        # assert zero hashes
        self.assertEqual(len(sig3), 0)

    def test_symmetric_difference_sigs(self):
        """
        Test the symmetric_difference_sigs method to ensure correct symmetric difference.
        """
        sym_diff_sig = self.snipe_sig.symmetric_difference_sigs(self.snipe_sig_2)
        expected_hashes = np.array([10, 20, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(sym_diff_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(sym_diff_sig.abundances, expected_abundances)
        self.assertEqual(sym_diff_sig.name, "test_signature_symmetric_difference_test_signature_2")

    def test_symmetric_difference_sigs_result_zero(self):
        """
        Test the symmetric_difference_sigs method when the result is an empty signature.
        """
        # Create a signature identical to snipe_sig
        sig_copy = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        with self.assertRaises(RuntimeError):
            # Symmetric difference of a signature with itself should be empty
            sig_copy.symmetric_difference_sigs(sig_copy)

    def test_operator_add(self):
        """
        Test the overloaded + operator for union of two signatures.
        """
        union_sig = self.snipe_sig + self.snipe_sig_2
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 10, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(union_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(union_sig.abundances, expected_abundances)

    def test_operator_iadd(self):
        """
        Test the overloaded += operator for in-place union of two signatures.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        snipe += self.snipe_sig_2
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 10, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(snipe.hashes, expected_hashes)
        np.testing.assert_array_equal(snipe.abundances, expected_abundances)
        self.assertEqual(snipe.name, "test_signature_union_test_signature_2")

    def test_operator_sub(self):
        """
        Test the overloaded - operator for difference of two signatures.
        """
        difference_sig = self.snipe_sig - self.snipe_sig_2
        expected_hashes = np.array([10, 20], dtype=np.uint64)
        expected_abundances = np.array([1, 2], dtype=np.uint32)
        np.testing.assert_array_equal(difference_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(difference_sig.abundances, expected_abundances)

    def test_operator_isub(self):
        """
        Test the overloaded -= operator for in-place difference of two signatures.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        snipe -= self.snipe_sig_2
        expected_hashes = np.array([10, 20], dtype=np.uint64)
        expected_abundances = np.array([1, 2], dtype=np.uint32)
        np.testing.assert_array_equal(snipe.hashes, expected_hashes)
        np.testing.assert_array_equal(snipe.abundances, expected_abundances)
        self.assertEqual(snipe.name, "test_signature_difference_test_signature_2")

    def test_operator_or(self):
        """
        Test the overloaded | operator for union of two signatures.
        """
        union_sig = self.snipe_sig | self.snipe_sig_2
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 10, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(union_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(union_sig.abundances, expected_abundances)

    def test_operator_ior(self):
        """
        Test the overloaded |= operator for in-place union of two signatures.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        snipe |= self.snipe_sig_2
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 10, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(snipe.hashes, expected_hashes)
        np.testing.assert_array_equal(snipe.abundances, expected_abundances)
        self.assertEqual(snipe.name, "test_signature_union_test_signature_2")

    def test_operator_xor(self):
        """
        Test the overloaded ^ operator for symmetric difference of two signatures.
        """
        sym_diff_sig = self.snipe_sig ^ self.snipe_sig_2
        expected_hashes = np.array([10, 20, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(sym_diff_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(sym_diff_sig.abundances, expected_abundances)

    def test_operator_ixor(self):
        """
        Test the overloaded ^= operator for in-place symmetric difference of two signatures.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        snipe ^= self.snipe_sig_2
        expected_hashes = np.array([10, 20, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(snipe.hashes, expected_hashes)
        np.testing.assert_array_equal(snipe.abundances, expected_abundances)
        self.assertEqual(snipe.name, "test_signature_symmetric_difference_test_signature_2")

    def test_sum_signatures(self):
        """
        Test the sum_signatures class method to ensure correct aggregation of multiple signatures.
        """
        # Define a third sample signature
        sig3_json = json.dumps([
            {
                "class": "sourmash_signature",
                "email": "",
                "hash_function": "0.murmur64",
                "filename": "-",
                "name": "test_signature_3",
                "license": "CC0",
                "signatures": [
                    {
                        "num": 0,
                        "ksize": 31,
                        "seed": 42,
                        "max_hash": 18446744073709551615,
                        "mins": [50, 60, 70, 80, 90],
                        "md5sum": "1234567890abcdef1234567890abcdef",
                        "abundances": [5, 6, 7, 8, 9],
                        "molecule": "dna"
                    }
                ],
                "version": 0.4
            }
        ])
        sig3 = sourmash.load_one_signature_from_json(sig3_json)
        snipe_sig3 = SnipeSig(
            sourmash_sig=sig3,
            enable_logging=False
        )

        # Sum the three signatures
        summed_sig = SnipeSig.sum_signatures(
            signatures=[self.snipe_sig, self.snipe_sig_2, snipe_sig3],
            name="summed_signature",
            filename=None,
            enable_logging=False
        )
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 15, 12, 14, 8, 9], dtype=np.uint32)
        np.testing.assert_array_equal(summed_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(summed_sig.abundances, expected_abundances)
        self.assertEqual(summed_sig.name, "summed_signature")

    def test_sum_signatures_empty_list(self):
        """
        Test the sum_signatures class method with an empty list of signatures.
        """
        with self.assertRaises(ValueError):
            SnipeSig.sum_signatures(
                signatures=[],
                name="empty_sum",
                filename=None,
                enable_logging=False
            )

    def test_sum_signatures_inconsistent_ksize_scale(self):
        """
        Test the sum_signatures class method with signatures having different ksize or scale.
        """
        # Create a signature with different ksize
        sig_diff_ksize_json = json.dumps([
            {
                "class": "sourmash_signature",
                "email": "",
                "hash_function": "0.murmur64",
                "filename": "-",
                "name": "test_signature_diff_ksize",
                "license": "CC0",
                "signatures": [
                    {
                        "num": 0,
                        "ksize": 25,  # Different ksize
                        "seed": 42,
                        "max_hash": 18446744073709551615,
                        "mins": [100, 110, 120],
                        "md5sum": "fedcba0987654321fedcba0987654321",
                        "abundances": [10, 11, 12],
                        "molecule": "dna"
                    }
                ],
                "version": 0.4
            }
        ])
        sig_diff_ksize = sourmash.load_one_signature_from_json(sig_diff_ksize_json)
        snipe_diff_ksize = SnipeSig(
            sourmash_sig=sig_diff_ksize,
            enable_logging=False
        )

        with self.assertRaises(ValueError):
            SnipeSig.sum_signatures(
                signatures=[self.snipe_sig, snipe_diff_ksize],
                name="inconsistent_sum",
                filename=None,
                enable_logging=False
            )

    def test_radd_sum(self):
        """
        Test the __radd__ method to support the built-in sum() function.
        """
        summed_sig = sum([self.snipe_sig, self.snipe_sig_2], 0)
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 10, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(summed_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(summed_sig.abundances, expected_abundances)
        self.assertEqual(summed_sig.name, "test_signature_union_test_signature_2")

    def test_radd_sum_zero_start(self):
        """
        Test the sum() function starting with 0 to ensure it handles the initial value correctly.
        """
        summed_sig = sum([self.snipe_sig, self.snipe_sig_2], 0)
        expected_hashes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 6, 8, 10, 6, 7], dtype=np.uint32)
        np.testing.assert_array_equal(summed_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(summed_sig.abundances, expected_abundances)
        self.assertEqual(summed_sig.name, "test_signature_union_test_signature_2")

    def test_get_info(self):
        """
        Test the get_info method to ensure it returns correct signature information.
        """
        info = self.snipe_sig.get_info()
        expected_info = {
            "name": "test_signature",
            "filename": "-",
            "md5sum": "d9be84d0b05d39d231702887169b69a7",
            "ksize": 31,
            "scale": 1,
            "track_abundance": True,
            "sigtype": SigType.SAMPLE,
            "num_hashes": 5
        }

        self.assertEqual(info, expected_info)

    def test_get_name(self):
        """
        Test the get_name method to ensure it returns the correct name.
        """
        self.assertEqual(self.snipe_sig.get_name(), "test_signature")

    def test_error_union_inconsistent_ksize_scale(self):
        """
        Test the union_sigs method with signatures having inconsistent ksize or scale.
        """
        # Create a signature with different ksize
        sig_diff_ksize_json = json.dumps([
            {
                "class": "sourmash_signature",
                "email": "",
                "hash_function": "0.murmur64",
                "filename": "-",
                "name": "test_signature_diff_ksize",
                "license": "CC0",
                "signatures": [
                    {
                        "num": 0,
                        "ksize": 25,  # Different ksize
                        "seed": 42,
                        "max_hash": 18446744073709551615,
                        "mins": [100, 110, 120],
                        "md5sum": "fedcba0987654321fedcba0987654321",
                        "abundances": [10, 11, 12],
                        "molecule": "dna"
                    }
                ],
                "version": 0.4
            }
        ])
        sig_diff_ksize = sourmash.load_one_signature_from_json(sig_diff_ksize_json)
        snipe_diff_ksize = SnipeSig(
            sourmash_sig=sig_diff_ksize,
            enable_logging=False
        )

        with self.assertRaises(ValueError):
            self.snipe_sig.union_sigs(snipe_diff_ksize)

    def test_error_contains_non_integer(self):
        """
        Test the __contains__ method with a non-integer input.
        """
        # Since the __contains__ method expects an integer, passing a string should return False
        # or raise a TypeError based on implementation. Adjust the test accordingly.
        # Here, assuming it returns False without raising an error.
        self.assertFalse("not_an_integer" in self.snipe_sig)

    def test_empty_signature_operations(self):
        """
        Test set operations involving empty signatures.
        """
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )

        # Union with empty signature should return the original
        union_sig = self.snipe_sig.union_sigs(empty_sig)
        np.testing.assert_array_equal(union_sig.hashes, self.snipe_sig.hashes)
        np.testing.assert_array_equal(union_sig.abundances, self.snipe_sig.abundances)

        # Intersection with empty signature should return empty
        intersection_sig = self.snipe_sig.intersection_sigs(empty_sig)
        self.assertEqual(len(intersection_sig), 0)

        # Difference with empty signature should return original
        difference_sig = self.snipe_sig.difference_sigs(empty_sig)
        np.testing.assert_array_equal(difference_sig.hashes, self.snipe_sig.hashes)
        np.testing.assert_array_equal(difference_sig.abundances, self.snipe_sig.abundances)

        # Symmetric difference with empty signature should return original
        sym_diff_sig = self.snipe_sig.symmetric_difference_sigs(empty_sig)
        np.testing.assert_array_equal(sym_diff_sig.hashes, self.snipe_sig.hashes)
        np.testing.assert_array_equal(sym_diff_sig.abundances, self.snipe_sig.abundances)

    def testcreate_from_hashes_abundances(self):
        """
        Test the create_from_hashes_abundances class method to ensure correct creation of a SnipeSig instance.
        """
        hashes = np.array([100, 200, 300], dtype=np.uint64)
        abundances = np.array([10, 20, 30], dtype=np.uint32)
        created_sig = SnipeSig.create_from_hashes_abundances(
            hashes=hashes,
            abundances=abundances,
            ksize=31,
            scale=1,
            name="created_signature",
            filename="created.sig",
            enable_logging=False
        )
        self.assertEqual(created_sig.name, "created_signature")
        self.assertEqual(created_sig.filename, "created.sig")
        self.assertEqual(created_sig.ksize, 31)
        self.assertEqual(created_sig.scale, 1)
        self.assertEqual(created_sig.sigtype, SigType.SAMPLE)
        self.assertTrue(created_sig.track_abundance)
        self.assertEqual(len(created_sig), 3)
        np.testing.assert_array_equal(created_sig.hashes, hashes)
        np.testing.assert_array_equal(created_sig.abundances, abundances)

    def test_apply_mask_sorted(self):
        """
        Test the _apply_mask method to ensure that the mask is applied correctly and sorted order is preserved.
        """
        mask = np.array([True, False, True, False, True])
        self.snipe_sig._apply_mask(mask)
        expected_hashes = np.array([10, 30, 50], dtype=np.uint64)
        expected_abundances = np.array([1, 3, 5], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)

    def test_apply_mask_unsorted_result(self):
        """
        Test the _apply_mask method to ensure it raises an error if the resulting hashes are not sorted.
        """
        # Manually shuffle the hashes to make them unsorted after masking
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        snipe._hashes = np.array([30, 10, 50], dtype=np.uint64)  # Unsorted
        snipe._abundances = np.array([3, 1, 5], dtype=np.uint32)
        mask = np.array([True, True, True])
        with self.assertRaises(RuntimeError):
            snipe._apply_mask(mask)

    def test_track_abundance_false(self):
        """
        Test initialization when track_abundance is False.
        """
        # Create a MinHash object that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        # Create a SourmashSignature object
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        expected_abundances = np.ones(5, dtype=np.uint32)
        np.testing.assert_array_equal(snipe.abundances, expected_abundances)
        self.assertFalse(snipe.track_abundance)

    def test_logger_enabled(self):
        """
        Test that logging is enabled when enable_logging=True.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=True
        )
        self.assertEqual(snipe.logger.level, logging.DEBUG)
        # Check that handlers are present
        self.assertTrue(snipe.logger.hasHandlers())

    def test_logger_disabled(self):
        """
        Test that logging is disabled when enable_logging=False.
        """
        snipe = SnipeSig(
            sourmash_sig=self.sample_sig,
            enable_logging=False
        )
        self.assertEqual(snipe.logger.level, logging.CRITICAL)
        # Handlers should not be present except possibly the root handler
        # Assuming that SnipeSig adds handlers only when enable_logging=True
        self.assertFalse(any(handler.level == logging.DEBUG for handler in snipe.logger.handlers))

    def test_reset_abundance(self):
        """
        Test the reset_abundance method to ensure all abundances are set to the new value.
        """
        self.snipe_sig.reset_abundance(new_abundance=5)
        expected_abundances = np.array([5, 5, 5, 5, 5], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)
        self.assertEqual(self.snipe_sig.abundances.dtype, np.uint32)

    def test_reset_abundance_invalid(self):
        """
        Test the reset_abundance method with invalid inputs to ensure it raises ValueError.
        """
        # Test with negative abundance
        with self.assertRaises(ValueError):
            self.snipe_sig.reset_abundance(new_abundance=-1)

        # Test with non-integer abundance
        with self.assertRaises(ValueError):
            self.snipe_sig.reset_abundance(new_abundance=2.5)

        # Test resetting when track_abundance is False
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            snipe_no_abund.reset_abundance(new_abundance=3)

    def test_keep_min_abundance(self):
        """
        Test the keep_min_abundance method to ensure it correctly retains hashes with abundance >= min_abundance.
        """
        self.snipe_sig.keep_min_abundance(min_abundance=3)
        expected_hashes = np.array([30, 40, 50], dtype=np.uint64)
        expected_abundances = np.array([3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)

    def test_keep_min_abundance_invalid(self):
        """
        Test the keep_min_abundance method with invalid inputs to ensure it raises ValueError.
        """
        # Test with negative min_abundance
        with self.assertRaises(ValueError):
            self.snipe_sig.keep_min_abundance(min_abundance=-2)

        # Test with non-integer min_abundance
        with self.assertRaises(ValueError):
            self.snipe_sig.keep_min_abundance(min_abundance=1.5)

        # Test keeping min abundance when track_abundance is False
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            snipe_no_abund.keep_min_abundance(min_abundance=1)

    def test_keep_max_abundance(self):
        """
        Test the keep_max_abundance method to ensure it correctly retains hashes with abundance <= max_abundance.
        """
        self.snipe_sig.keep_max_abundance(max_abundance=3)
        expected_hashes = np.array([10, 20, 30], dtype=np.uint64)
        expected_abundances = np.array([1, 2, 3], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)

    def test_keep_max_abundance_invalid(self):
        """
        Test the keep_max_abundance method with invalid inputs to ensure it raises ValueError.
        """
        # Test with negative max_abundance
        with self.assertRaises(ValueError):
            self.snipe_sig.keep_max_abundance(max_abundance=-5)

        # Test with non-integer max_abundance
        with self.assertRaises(ValueError):
            self.snipe_sig.keep_max_abundance(max_abundance=2.7)

        # Test keeping max abundance when track_abundance is False
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            snipe_no_abund.keep_max_abundance(max_abundance=2)

    def test_trim_below_median(self):
        """
        Test the trim_below_median method to ensure it correctly trims hashes with abundance below the median.
        """
        # Current abundances: [1, 2, 3, 4, 5], median is 3
        self.snipe_sig.trim_below_median()
        expected_hashes = np.array([30, 40, 50], dtype=np.uint64)
        expected_abundances = np.array([3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)

    def test_trim_below_median_empty_abundances(self):
        """
        Test the trim_below_median method when there are no abundances to trim.
        """
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        # Attempt to trim below median on an empty signature should not raise an error
        try:
            empty_sig.trim_below_median()
        except Exception as e:
            self.fail(f"trim_below_median raised an exception on empty signature: {e}")

    def test_trim_below_median_invalid(self):
        """
        Test the trim_below_median method to ensure it raises ValueError when track_abundance is False.
        """
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            snipe_no_abund.trim_below_median()

    def test_count_singletons(self):
        """
        Test the count_singletons method to ensure it correctly counts hashes with abundance == 1.
        """
        count = self.snipe_sig.count_singletons()
        self.assertEqual(count, 1)  # Only hash 10 has abundance 1

    def test_count_singletons_no_singletons(self):
        """
        Test the count_singletons method when there are no singletons.
        """
        # Increase all abundances to >=2
        self.snipe_sig.keep_min_abundance(min_abundance=2)
        count = self.snipe_sig.count_singletons()
        self.assertEqual(count, 0)

    def test_count_singletons_all_singletons(self):
        """
        Test the count_singletons method when all hashes are singletons.
        """
        # Reset all abundances to 1
        self.snipe_sig.reset_abundance(new_abundance=1)
        count = self.snipe_sig.count_singletons()
        self.assertEqual(count, 5)

    def test_count_singletons_invalid(self):
        """
        Test the count_singletons method to ensure it raises ValueError when track_abundance is False.
        """
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            snipe_no_abund.count_singletons()

    def test_trim_singletons(self):
        """
        Test the trim_singletons method to ensure it correctly removes hashes with abundance == 1.
        """
        # Current abundances: [1, 2, 3, 4, 5]
        self.snipe_sig.trim_singletons()
        expected_hashes = np.array([20, 30, 40, 50], dtype=np.uint64)
        expected_abundances = np.array([2, 3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)

    def test_trim_singletons_no_singletons(self):
        """
        Test the trim_singletons method when there are no singletons.
        """
        # Increase all abundances to >=2
        self.snipe_sig.keep_min_abundance(min_abundance=2)
        self.snipe_sig.trim_singletons()
        expected_hashes = np.array([20, 30, 40, 50], dtype=np.uint64)
        expected_abundances = np.array([2, 3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)

    def test_trim_singletons_all_singletons(self):
        """
        Test the trim_singletons method when all hashes are singletons.
        """
        # Reset all abundances to 1
        self.snipe_sig.reset_abundance(new_abundance=1)
        self.snipe_sig.trim_singletons()
        expected_hashes = np.array([], dtype=np.uint64)
        expected_abundances = np.array([], dtype=np.uint32)
        np.testing.assert_array_equal(self.snipe_sig.hashes, expected_hashes)
        np.testing.assert_array_equal(self.snipe_sig.abundances, expected_abundances)

    def test_trim_singletons_invalid(self):
        """
        Test the trim_singletons method to ensure it raises ValueError when track_abundance is False.
        """
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            snipe_no_abund.trim_singletons()

    def test_total_abundance(self):
        """
        Test the total_abundance property to ensure it correctly sums all abundances.
        """
        total = self.snipe_sig.total_abundance
        self.assertEqual(total, 15)  # 1 + 2 + 3 + 4 + 5

    def test_total_abundance_empty(self):
        """
        Test the total_abundance property with an empty signature.
        """
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        total = empty_sig.total_abundance
        self.assertEqual(total, 0)

    def test_total_abundance_invalid(self):
        """
        Test the total_abundance property to ensure it raises ValueError when track_abundance is False.
        """
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            _ = snipe_no_abund.total_abundance

    def test_mean_abundance(self):
        """
        Test the mean_abundance property to ensure it correctly calculates the average abundance.
        """
        mean = self.snipe_sig.mean_abundance
        self.assertAlmostEqual(mean, 3.0)  # (1 + 2 + 3 + 4 + 5) / 5

    def test_mean_abundance_empty(self):
        """
        Test the mean_abundance property with an empty signature.
        """
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        mean = empty_sig.mean_abundance
        self.assertEqual(mean, 0.0)

    def test_mean_abundance_invalid(self):
        """
        Test the mean_abundance property to ensure it raises ValueError when track_abundance is False.
        """
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            _ = snipe_no_abund.mean_abundance

    def test_median_abundance(self):
        """
        Test the median_abundance property to ensure it correctly calculates the median abundance.
        """
        median = self.snipe_sig.median_abundance
        self.assertEqual(median, 3.0)  # Median of [1, 2, 3, 4, 5]

    def test_median_abundance_empty(self):
        """
        Test the median_abundance property with an empty signature.
        """
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        median = empty_sig.median_abundance
        self.assertEqual(median, 0.0)

    def test_median_abundance_invalid(self):
        """
        Test the median_abundance property to ensure it raises ValueError when track_abundance is False.
        """
        # Create a signature that does not track abundance
        mh = sourmash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = sourmash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmash_sig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            _ = snipe_no_abund.median_abundance

if __name__ == '__main__':
    unittest.main()