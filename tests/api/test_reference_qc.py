import json
import logging
import os
import tempfile
import unittest

import numpy as np
import sourmash

from snipe.api import SnipeSig
from snipe.api.enums import SigType
from snipe.api.reference_QC import ReferenceQC


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

class TestReferenceQC(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures for ReferenceQC.
        """
        # Sample signature (sample_sig)
        self.sample_json = sample_signature_json()
        self.sample_sig = SnipeSig(
            sourmash_sig=self.sample_json,
            sig_type=SigType.SAMPLE,
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
            sourmash_sig=self.reference_json,
            sig_type=SigType.GENOME,
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
            sourmash_sig=self.amplicon_json,
            sig_type=SigType.AMPLICON,
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