import json
import logging
import os
import tempfile
import unittest

import numpy as np
import sourmash as smash

from snipe.api import SigType, SnipeSig, ReferenceQC

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
        self.sample_sig = smash.load_one_signature_from_json(self.sample_json)

        self.sample_json_2 = sample_signature_json_2()
        self.sample_sig_2 = smash.load_one_signature_from_json(self.sample_json_2)

        # Create SnipeSig instances
        self.snipe_sig = SnipeSig(
            sourmashSig=self.sample_sig,
            enable_logging=False
        )
        self.snipe_sig_2 = SnipeSig(
            sourmashSig=self.sample_sig_2,
            enable_logging=False
        )

    def test_init_from_signature_object(self):
        """
        Test initialization of SnipeSig from a SourmashSignature object.
        """
        snipe = SnipeSig(
            sourmashSig=self.sample_sig,
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
            sourmashSig=self.sample_json,
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
                sourmashSig=tmp_file_path,
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
        Test initialization with an invalid input type for sourmashSig.
        """
        with self.assertRaises(TypeError):
            SnipeSig(
                sourmashSig=12345,  # Invalid type
                enable_logging=False
            )

    def test_init_invalid_json(self):
        """
        Test initialization with an invalid JSON string.
        """
        invalid_json = '{"invalid": "json"}'
        with self.assertRaises(ValueError):
            SnipeSig(
                sourmashSig=invalid_json,
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
        with self.assertRaises(RuntimeError):
            # Subtract the second signature again to remove all hashes
            sig1.difference_sigs(self.snipe_sig_2)

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
            sourmashSig=self.sample_sig,
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
            sourmashSig=self.sample_sig,
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
            sourmashSig=self.sample_sig,
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
            sourmashSig=self.sample_sig,
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
            sourmashSig=self.sample_sig,
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
        sig3 = smash.load_one_signature_from_json(sig3_json)
        snipe_sig3 = SnipeSig(
            sourmashSig=sig3,
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
        sig_diff_ksize = smash.load_one_signature_from_json(sig_diff_ksize_json)
        snipe_diff_ksize = SnipeSig(
            sourmashSig=sig_diff_ksize,
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
            "sigtype": "SAMPLE",
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
        sig_diff_ksize = smash.load_one_signature_from_json(sig_diff_ksize_json)
        snipe_diff_ksize = SnipeSig(
            sourmashSig=sig_diff_ksize,
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
            sourmashSig=self.sample_sig,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        # Create a SourmashSignature object
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe = SnipeSig(
            sourmashSig=sig_no_abund,
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
            sourmashSig=self.sample_sig,
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
            sourmashSig=self.sample_sig,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30, 40, 50]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
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
        mh = smash.minhash.MinHash(
            n=0,
            ksize=31,
            scaled=1,
            track_abundance=False
        )
        for h in [10, 20, 30]:
            mh.add_hash(h)
        sig_no_abund = smash.signature.SourmashSignature(
            mh,
            name="test_signature_no_abundance",
            filename="-"
        )
        snipe_no_abund = SnipeSig(
            sourmashSig=sig_no_abund,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            _ = snipe_no_abund.median_abundance

class TestReferenceQC(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures for ReferenceQC.
        """
        # Sample signature (sample_sig)
        self.sample_json = sample_signature_json()
        self.sample_sig = SnipeSig(
            sourmashSig=self.sample_json,
            sigType=SigType.SAMPLE,
            enable_logging=False
        )

        # Reference genome signature (reference_sig)
        self.reference_json = json.dumps([
            {
                "class": "sourmash_signature",
                "email": "",
                "hash_function": "0.murmur64",
                "filename": "-",
                "name": "reference_signature",
                "license": "CC0",
                "signatures": [
                    {
                        "num": 0,
                        "ksize": 31,
                        "seed": 42,
                        "max_hash": 18446744073709551615,
                        "mins": [10, 20, 30, 40, 50, 60],
                        "md5sum": "fedcba0987654321fedcba0987654321",
                        "abundances": [2, 3, 4, 5, 6, 7],
                        "molecule": "dna"
                    }
                ],
                "version": 0.4
            }
        ])
        self.reference_sig = SnipeSig(
            sourmashSig=self.reference_json,
            sigType=SigType.GENOME,
            enable_logging=False
        )

        # Amplicon signature (amplicon_sig)
        self.amplicon_json = json.dumps([
            {
                "class": "sourmash_signature",
                "email": "",
                "hash_function": "0.murmur64",
                "filename": "-",
                "name": "amplicon_signature",
                "license": "CC0",
                "signatures": [
                    {
                        "num": 0,
                        "ksize": 31,
                        "seed": 42,
                        "max_hash": 18446744073709551615,
                        "mins": [30, 40, 50],
                        "md5sum": "abcdefabcdefabcdefabcdefabcdefab",
                        "abundances": [3, 4, 5],
                        "molecule": "dna"
                    }
                ],
                "version": 0.4
            }
        ])
        self.amplicon_sig = SnipeSig(
            sourmashSig=self.amplicon_json,
            sigType=SigType.AMPLICON,
            enable_logging=False
        )

    def test_initialization(self):
        """
        Test that ReferenceQC initializes correctly with valid signatures.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        self.assertEqual(qc.sample_sig, self.sample_sig)
        self.assertEqual(qc.reference_sig, self.reference_sig)
        self.assertEqual(qc.amplicon_sig, self.amplicon_sig)

    def test_invalid_sig_types(self):
        """
        Test that ReferenceQC raises ValueError when signatures have incorrect types.
        """
        with self.assertRaises(ValueError):
            ReferenceQC(
                sample_sig=self.reference_sig,  # Should be SAMPLE type
                reference_sig=self.reference_sig,
                amplicon_sig=self.amplicon_sig,
                enable_logging=False
            )

        with self.assertRaises(ValueError):
            ReferenceQC(
                sample_sig=self.sample_sig,
                reference_sig=self.sample_sig,  # Should be GENOME type
                amplicon_sig=self.amplicon_sig,
                enable_logging=False
            )

        with self.assertRaises(ValueError):
            ReferenceQC(
                sample_sig=self.sample_sig,
                reference_sig=self.reference_sig,
                amplicon_sig=self.sample_sig,  # Should be AMPLICON type
                enable_logging=False
            )

    def test_calculate_stats(self):
        """
        Test the calculation of statistics in ReferenceQC.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )

        # Check sample stats
        self.assertEqual(qc.sample_stats["Total unique k-mers"], 5)
        self.assertEqual(qc.sample_stats["k-mer total abundance"], 15)
        self.assertEqual(qc.sample_stats["k-mer mean abundance"], 3)
        self.assertEqual(qc.sample_stats["k-mer median abundance"], 3)
        self.assertEqual(qc.sample_stats["num_singletons"], 1)

        # Check genome stats
        self.assertEqual(qc.genome_stats["Genomic unique k-mers"], 5)
        self.assertAlmostEqual(qc.genome_stats["Genome coverage index"], 5/6)
        self.assertEqual(qc.genome_stats["Genomic k-mers total abundance"], 15)
        self.assertEqual(qc.genome_stats["Genomic k-mers mean abundance"], 3)
        self.assertEqual(qc.genome_stats["Genomic k-mers median abundance"], 3)
        self.assertAlmostEqual(qc.genome_stats["Mapping index"], 1.0)

        # Check amplicon stats
        self.assertEqual(qc.amplicon_stats["Amplicon unique k-mers"], 3)
        self.assertEqual(qc.amplicon_stats["Amplicon coverage index"], 1.0)
        self.assertEqual(qc.amplicon_stats["Amplicon k-mers total abundance"], 12)
        self.assertEqual(qc.amplicon_stats["Amplicon k-mers mean abundance"], 4)
        self.assertEqual(qc.amplicon_stats["Amplicon k-mers median abundance"], 4)
        self.assertAlmostEqual(qc.amplicon_stats["Relative total abundance"], 12/15)
        self.assertAlmostEqual(qc.amplicon_stats["Relative coverage"], (1.0)/(5/6))

        # Check predicted assay type
        self.assertEqual(qc.predicted_assay_type, "WXS")

    def test_get_aggregated_stats(self):
        """
        Test the get_aggregated_stats method without advanced stats.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        stats = qc.get_aggregated_stats()
        self.assertIn("Total unique k-mers", stats)
        self.assertIn("Genomic unique k-mers", stats)
        self.assertIn("Amplicon unique k-mers", stats)
        self.assertIn("Predicted Assay Type", stats)
        self.assertNotIn("Median-trimmed unique k-mers", stats)

    def test_get_aggregated_stats_with_advanced(self):
        """
        Test the get_aggregated_stats method with advanced stats.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )

        stats = qc.get_aggregated_stats(include_advanced=True)
        # self.assertIn("Median-trimmed unique k-mers", stats)
        # self.assertIn("Median-trimmed Genomic unique k-mers", stats)
        # self.assertIn("Median-trimmed Amplicon unique k-mers", stats)

    def test_no_amplicon_sig(self):
        """
        Test ReferenceQC without providing an amplicon signature.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        stats = qc.get_aggregated_stats()
        self.assertIn("Total unique k-mers", stats)
        self.assertIn("Genomic unique k-mers", stats)
        self.assertNotIn("Amplicon unique k-mers", stats)
        self.assertNotIn("Predicted Assay Type", stats)

    def test_enable_logging(self):
        """
        Test that logging is enabled when enable_logging is True.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=True
        )
        self.assertEqual(qc.logger.level, logging.DEBUG)
        self.assertTrue(qc.logger.hasHandlers())

    def test_calculate_stats_with_empty_sample(self):
        """
        Test ReferenceQC with an empty sample signature.
        """
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        empty_sig.sigtype = SigType.SAMPLE
        qc = ReferenceQC(
            sample_sig=empty_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        self.assertEqual(qc.sample_stats["Total unique k-mers"], 0)
        self.assertEqual(qc.genome_stats["Genomic unique k-mers"], 0)
        self.assertEqual(qc.genome_stats["Genome coverage index"], 0)
        self.assertEqual(qc.genome_stats["Mapping index"], 0)

    def test_calculate_stats_with_empty_reference(self):
        """
        Test ReferenceQC with an empty reference signature.
        """
        empty_ref_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        empty_ref_sig.sigtype = SigType.GENOME
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=empty_ref_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        self.assertEqual(qc.genome_stats["Genomic unique k-mers"], 0)
        self.assertEqual(qc.genome_stats["Genome coverage index"], 0)

    def test_calculate_stats_with_empty_amplicon(self):
        """
        Test ReferenceQC with an empty amplicon signature.
        """
        empty_amp_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        empty_amp_sig.sigtype = SigType.AMPLICON
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=empty_amp_sig,
            enable_logging=False
        )
        self.assertEqual(qc.amplicon_stats["Amplicon unique k-mers"], 0)
        self.assertEqual(qc.amplicon_stats["Amplicon coverage index"], 0)

    def test_inconsistent_ksize_scale(self):
        """
        Test that ReferenceQC raises an error when signatures have different ksize or scale.
        """
        # Create a signature with different ksize
        diff_ksize_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([100, 200], dtype=np.uint64),
            abundances=np.array([1, 1], dtype=np.uint32),
            ksize=25,
            scale=1,
            enable_logging=False
        )
        diff_ksize_sig.sigtype = SigType.GENOME
        with self.assertRaises(ValueError):
            ReferenceQC(
                sample_sig=self.sample_sig,
                reference_sig=diff_ksize_sig,
                amplicon_sig=self.amplicon_sig,
                enable_logging=False
            )

    def test_mismatched_scales(self):
        """
        Test that ReferenceQC raises an error when signatures have different scales.
        """
        # Create a signature with different scale
        diff_scale_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([100, 200], dtype=np.uint64),
            abundances=np.array([1, 1], dtype=np.uint32),
            ksize=31,
            scale=2,
            enable_logging=False
        )
        diff_scale_sig.sigtype = SigType.GENOME
        with self.assertRaises(ValueError):
            ReferenceQC(
                sample_sig=self.sample_sig,
                reference_sig=diff_scale_sig,
                amplicon_sig=self.amplicon_sig,
                enable_logging=False
            )

    def test_trim_below_median(self):
        """
        Test the advanced stats after median trimming.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        stats = qc.get_aggregated_stats(include_advanced=True)
        self.assertEqual(stats["Median-trimmed unique k-mers"], 3)
        self.assertEqual(stats["Median-trimmed total abundance"], 12)
        self.assertEqual(stats["Median-trimmed mean abundance"], 4)
        self.assertEqual(stats["Median-trimmed median abundance"], 4)
        self.assertEqual(stats["Median-trimmed Genomic unique k-mers"], 3)
        self.assertEqual(stats["Median-trimmed Amplicon unique k-mers"], 3)

    def test_predicted_assay_type(self):
        """
        Test the predicted assay type based on relative total abundance.
        """
        # Adjust the sample signature abundances to change relative total abundance
        self.sample_sig.reset_abundance(new_abundance=1)
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        # Now, the amplicon k-mers total abundance is 3 (since all abundances are 1)
        # Genomic k-mers total abundance is 5
        # Relative total abundance = 3/5 = 0.6
        # Should be predicted as WXS
        self.assertEqual(qc.predicted_assay_type, "WXS")

        # ---------------------------------------------------------------------
        # ------------- Testing ROI -------------------------------------------
        # ---------------------------------------------------------------------

    def test_predict_coverage_with_valid_input(self):
        """
        Test the predict_coverage method with valid input and verify the predicted coverage.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        predicted_coverage = qc.predict_coverage(extra_fold=1.0, n=10)
        # Since we have small sample data, the predicted coverage may not be meaningful
        # But we can at least check that it returns a float between 0 and 1
        self.assertIsInstance(predicted_coverage, float)
        self.assertGreaterEqual(predicted_coverage, 0.0)
        self.assertLessEqual(predicted_coverage, 1.0)

    def test_predict_coverage_with_zero_extra_fold(self):
        """
        Test the predict_coverage method with extra_fold=0 to ensure it returns current coverage.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        
        # makre sure this test thown ValueError: extra_fold must be >= 1.0.
        with self.assertRaises(ValueError):
            qc.predict_coverage(extra_fold=0.0, n=10)


    def test_predict_coverage_with_negative_extra_fold(self):
        """
        Test the predict_coverage method with negative extra_fold to ensure it raises ValueError.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            qc.predict_coverage(extra_fold=-1.0, n=10)

    def test_predict_coverage_with_large_extra_fold(self):
        """
        Test the predict_coverage method with a large extra_fold value.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        predicted_coverage = qc.predict_coverage(extra_fold=10.0, n=10)
        self.assertIsInstance(predicted_coverage, float)
        self.assertGreaterEqual(predicted_coverage, 0.0)
        self.assertLessEqual(predicted_coverage, 1.0)

    def test_predict_coverage_with_small_n(self):
        """
        Test the predict_coverage method with small n value to check for sufficient data points.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )

        with self.assertRaises(RuntimeError):
            qc.predict_coverage(extra_fold=1.0, n=2)
        


    def test_predict_coverage_with_large_n(self):
        """
        Test the predict_coverage method with large n value to check for performance.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        predicted_coverage = qc.predict_coverage(extra_fold=1.0, n=100)
        self.assertIsInstance(predicted_coverage, float)
        self.assertGreaterEqual(predicted_coverage, 0.0)
        self.assertLessEqual(predicted_coverage, 1.0)

    def test_predict_coverage_with_empty_sample(self):
        """
        Test the predict_coverage method with an empty sample signature.
        """
        empty_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            enable_logging=False
        )
        empty_sig.sigtype = SigType.SAMPLE
        qc = ReferenceQC(
            sample_sig=empty_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        predicted_coverage = qc.predict_coverage(extra_fold=1.0, n=10)
        self.assertEqual(predicted_coverage, 0.0)

    def test_predict_coverage_with_non_integer_n(self):
        """
        Test the predict_coverage method with non-integer n value to ensure it raises ValueError.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            qc.predict_coverage(extra_fold=1.0, n=10.5)

    def test_predict_coverage_with_zero_n(self):
        """
        Test the predict_coverage method with n=0 to ensure it raises ValueError.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        with self.assertRaises(ValueError):
            qc.predict_coverage(extra_fold=1.0, n=0)

    def test_predict_coverage_insufficient_data_points(self):
        """
        Test the predict_coverage method with n=1, which may be insufficient for curve fitting.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        # With n=1, only one data point, curve fitting should fail
        with self.assertRaises(RuntimeError):
            qc.predict_coverage(extra_fold=1.0, n=1)

    def test_predict_coverage_does_not_exceed_one(self):
        """
        Test that the predicted coverage does not exceed 1.0.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        # Use a large extra_fold to attempt to push predicted coverage over 1.0
        predicted_coverage = qc.predict_coverage(extra_fold=100.0, n=10)
        self.assertLessEqual(predicted_coverage, 1.0)
        self.assertGreaterEqual(predicted_coverage, 0.0)

    def test_predict_coverage_already_full_coverage(self):
        """
        Test predict_coverage when current coverage is already at 1.0.
        """
        # Create a sample signature identical to the reference signature
        sample_sig = self.reference_sig.copy()
        sample_sig.sigtype = SigType.SAMPLE
        qc = ReferenceQC(
            sample_sig=sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        current_coverage = qc.genome_stats["Genome coverage index"]
        self.assertEqual(current_coverage, 1.0)
        predicted_coverage = qc.predict_coverage(extra_fold=1.0, n=10)
        # Predicted coverage should still be 1.0
        self.assertEqual(predicted_coverage, 1.0)

if __name__ == '__main__':
    unittest.main()