import json
import logging
import os
import tempfile
import unittest

import numpy as np
import sourmash

from snipe.api.snipe_sig import SnipeSig
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
                        "abundances": [2, 2, 2, 3, 3, 3],
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

        # Autosomal genome signature
        self.autosomal_genome_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=np.uint64),
            abundances=np.array([2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=np.uint32),
            ksize=31,
            scale=1,
            name="autosomal-snipegenome",
            sig_type=SigType.GENOME,
            enable_logging=False
        )

        # Chromosome-specific signatures
        self.chr_signatures = {
            "autosome-1": SnipeSig.create_from_hashes_abundances(
                hashes=np.array([10, 20, 30], dtype=np.uint64),
                abundances=np.array([2, 2, 2], dtype=np.uint32),
                ksize=31,
                scale=1,
                name="autosome-1",
                sig_type=SigType.SAMPLE,
                enable_logging=False
            ),
            "autosome-2": SnipeSig.create_from_hashes_abundances(
                hashes=np.array([40, 50, 60], dtype=np.uint64),
                abundances=np.array([3, 3, 3], dtype=np.uint32),
                ksize=31,
                scale=1,
                name="autosome-2",
                sig_type=SigType.SAMPLE,
                enable_logging=False
            ),
            "sex-x": SnipeSig.create_from_hashes_abundances(
                hashes=np.array([70, 80, 90], dtype=np.uint64),
                abundances=np.array([4, 4, 4], dtype=np.uint32),
                ksize=31,
                scale=1,
                name="sex-x",
                sig_type=SigType.SAMPLE,
                enable_logging=False
            ),
            "sex-y": SnipeSig.create_from_hashes_abundances(
                hashes=np.array([100, 110, 120], dtype=np.uint64),
                abundances=np.array([5, 5, 5], dtype=np.uint32),
                ksize=31,
                scale=1,
                name="sex-y",
                sig_type=SigType.SAMPLE,
                enable_logging=False
            )
        }

        # Combined genome and chromosome signatures
        self.genome_and_chr_signatures = {
            "autosomal-snipegenome": self.autosomal_genome_sig,
            **self.chr_signatures
        }

        
    def create_test_signature(self, hashes, abundances, name, sig_type):
        """
        Helper method to create a SnipeSig instance for testing.
        """
        return SnipeSig.create_from_hashes_abundances(
            hashes=np.array(hashes, dtype=np.uint64),
            abundances=np.array(abundances, dtype=np.uint32),
            ksize=31,
            scale=1,
            name=name,
            sig_type=sig_type,
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
        predicted_coverage = qc.predict_coverage(extra_fold=2.0)
        # Predicted coverage should still be 1.0
        self.assertEqual(predicted_coverage, 1.0)


    def test_get_aggregated_stats_with_advanced(self):
        """
        Test the get_aggregated_stats method with advanced stats included.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=True
        )

        stats = qc.get_aggregated_stats(include_advanced=True)
        
        # Check advanced stats
        self.assertIn("Median-trimmed unique k-mers", stats)
        self.assertIn("Median-trimmed total abundance", stats)
        self.assertIn("Median-trimmed mean abundance", stats)
        self.assertIn("Median-trimmed median abundance", stats)
        self.assertIn("Median-trimmed Genomic unique k-mers", stats)
        self.assertIn("Median-trimmed Genome coverage index", stats)
        self.assertIn("Median-trimmed Amplicon unique k-mers", stats)
        self.assertIn("Median-trimmed Amplicon coverage index", stats)
        self.assertIn("Median-trimmed relative coverage", stats)
        self.assertIn("Median-trimmed relative mean abundance", stats)
        
    def test_calculate_stats_with_empty_amplicon(self):
        """
        Test ReferenceQC with an empty amplicon signature.
        """
        empty_amp_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([], dtype=np.uint64),
            abundances=np.array([], dtype=np.uint32),
            ksize=31,
            scale=1,
            name="empty_amplicon",
            enable_logging=False,
            sig_type=SigType.AMPLICON
        )
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=empty_amp_sig,
            enable_logging=False
        )
        self.assertEqual(qc.amplicon_stats["Amplicon unique k-mers"], 0)
        self.assertEqual(qc.amplicon_stats["Amplicon coverage index"], 0)
        self.assertEqual(qc.amplicon_stats["Amplicon k-mers total abundance"], 0)
        self.assertEqual(qc.amplicon_stats["Amplicon k-mers mean abundance"], 0)
        self.assertEqual(qc.amplicon_stats["Amplicon k-mers median abundance"], 0)
        self.assertEqual(qc.amplicon_stats["Relative total abundance"], 0)
        self.assertEqual(qc.amplicon_stats["Relative coverage"], 0)
        self.assertEqual(qc.predicted_assay_type, "WGS")
            
    def test_trim_below_median(self):
        """
        Test the advanced stats after median trimming.
        """
        # Create a sample signature with even abundances
        sample_json_even = json.dumps([
            {
                "class": "sourmash_signature",
                "email": "",
                "hash_function": "0.murmur64",
                "filename": "-",
                "name": "test_signature_even",
                "license": "CC0",
                "signatures": [
                    {
                        "num": 0,
                        "ksize": 31,
                        "seed": 42,
                        "max_hash": 18446744073709551615,
                        "mins": [10, 20, 30, 40, 50],
                        "md5sum": "d9be84d0b05d39d231702887169b69a7",
                        "abundances": [2, 2, 2, 2, 2],
                        "molecule": "dna"
                    }
                ],
                "version": 0.4
            }
        ])
        sample_sig_even = SnipeSig(
            sourmash_sig=sample_json_even,
            sig_type=SigType.SAMPLE,
            enable_logging=False
        )
        qc = ReferenceQC(
            sample_sig=sample_sig_even,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        stats = qc.get_aggregated_stats(include_advanced=True)
        self.assertEqual(stats["Median-trimmed unique k-mers"], 5)
        self.assertEqual(stats["Median-trimmed total abundance"], 10)
        self.assertEqual(stats["Median-trimmed mean abundance"], 2.0)
        self.assertEqual(stats["Median-trimmed median abundance"], 2.0)
        self.assertEqual(stats["Median-trimmed Genomic unique k-mers"], 5)
        self.assertEqual(stats["Median-trimmed Amplicon unique k-mers"], 3)
        

    def test_split_sig_randomly(self):
        """
        Test that split_sig_randomly correctly splits the sample signature into n parts.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=False
        )
        n = 3
        split_sigs = qc.split_sig_randomly(n)
        
        # Check that the number of returned signatures is n
        self.assertEqual(len(split_sigs), n)
        
        # Check that the total unique k-mers and total abundance are preserved
        total_unique = len(SnipeSig.sum_signatures(split_sigs))
        self.assertEqual(total_unique, len(qc.sample_sig))
        total_abundance = sum(sig.get_sample_stats["total_abundance"] for sig in split_sigs)
        self.assertEqual(total_abundance, qc.sample_stats["k-mer total abundance"])


    def test_calculate_coverage_vs_depth_with_valid_input(self):
        """
        Test the calculate_coverage_vs_depth method with valid input and verify the coverage vs depth data.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=True
        )
        coverage_depth_data = qc.calculate_coverage_vs_depth(n=2)
        
        # Check that two data points are returned
        self.assertEqual(len(coverage_depth_data), 2)
        
        # Check that cumulative_parts increment correctly
        self.assertEqual(coverage_depth_data[0]["cumulative_parts"], 1)
        self.assertEqual(coverage_depth_data[1]["cumulative_parts"], 2)
        
        # Check that cumulative_total_abundance is non-decreasing
        self.assertLessEqual(coverage_depth_data[0]["cumulative_total_abundance"],
                            coverage_depth_data[1]["cumulative_total_abundance"])
        
        # Check that coverage index is between 0 and 1
        for data_point in coverage_depth_data:
            self.assertGreaterEqual(data_point["cumulative_coverage_index"], 0.0)
            self.assertLessEqual(data_point["cumulative_coverage_index"], 1.0)





    def test_predict_coverage_with_insufficient_data_points(self):
        """
        Test the predict_coverage method with n=1, which is insufficient for curve fitting.
        """
        qc = ReferenceQC(
            sample_sig=self.sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        with self.assertRaises(RuntimeError):
            qc.predict_coverage(extra_fold=1.0, n=1)
            
    def test_calculate_sex_chrs_metrics_only_autosomal(self):
        """
        Test calculate_sex_chrs_metrics when only autosomal signatures are provided.
        """
        # Create chromosome-specific signatures without sex chromosomes
        chr1_sig = self.create_test_signature([10, 20, 30], [2, 2, 2], "autosome-1", SigType.SAMPLE)
        chr2_sig = self.create_test_signature([40, 50, 60], [3, 3, 3], "autosome-2", SigType.SAMPLE)
        genome_and_chr_signatures = {
            "autosomal-snipegenome": self.reference_sig,  # Using reference_sig as autosomal_genome_sig
            "autosome-1": chr1_sig,
            "autosome-2": chr2_sig
            # No sex chromosomes
        }

        # Initialize ReferenceQC with updated_sample_sig
        updated_sample_sig = self.create_test_signature(
            [10, 20, 30, 40, 50],
            [1, 2, 3, 4, 5],
            "test_signature_autosomal",
            SigType.SAMPLE
        )

        qc = ReferenceQC(
            sample_sig=updated_sample_sig,
            reference_sig=self.reference_sig,
            enable_logging=True
        )

        # Calculate sex chromosome metrics
        metrics = qc.calculate_sex_chrs_metrics(genome_and_chr_to_sig=genome_and_chr_signatures)

        # Verify metrics
        self.assertIn("X-Ploidy score", metrics)
        self.assertNotIn("Y-Coverage", metrics)

        # X-Ploidy should be 0.0 since no sex chromosomes are provided
        self.assertEqual(metrics["X-Ploidy score"], 0.0)


    def test_calculate_coverage_vs_depth_with_empty_splits(self):
        """
        Test the calculate_coverage_vs_depth method when some split signatures are empty.
        """
        # Create a sample signature with some zero abundances
        sample_json_zero = json.dumps([
            {
                "class": "sourmash_signature",
                "email": "",
                "hash_function": "0.murmur64",
                "filename": "-",
                "name": "test_signature_zero",
                "license": "CC0",
                "signatures": [
                    {
                        "num": 0,
                        "ksize": 31,
                        "seed": 42,
                        "max_hash": 18446744073709551615,
                        "mins": [10, 20, 30],
                        "md5sum": "d9be84d0b05d39d231702887169b69a7",
                        "abundances": [0, 0, 0],
                        "molecule": "dna"
                    }
                ],
                "version": 0.4
            }
        ])
        sample_sig_zero = SnipeSig(
            sourmash_sig=sample_json_zero,
            sig_type=SigType.SAMPLE,
            enable_logging=False
        )
        qc = ReferenceQC(
            sample_sig=sample_sig_zero,
            reference_sig=self.reference_sig,
            enable_logging=False
        )
        coverage_depth_data = qc.calculate_coverage_vs_depth(n=3)
        
        # All coverage indices should be 0
        for data_point in coverage_depth_data:
            self.assertEqual(data_point["cumulative_coverage_index"], 0.0)
            self.assertEqual(data_point["cumulative_total_abundance"], 0)

            
    def test_calculate_sex_chrs_metrics(self):
        """
        Test the calculate_sex_chrs_metrics method with valid chromosome signatures.
        """
        # Update sample_sig to include sex-x k-mers
        updated_sample_sig = SnipeSig.create_from_hashes_abundances(
            hashes=np.array([10, 20, 30, 40, 50, 70, 80, 90], dtype=np.uint64),
            abundances=np.array([1, 2, 3, 4, 5, 4, 4, 4], dtype=np.uint32),
            ksize=31,
            scale=1,
            name="test_signature_updated",
            sig_type=SigType.SAMPLE,
            enable_logging=False
        )

        # Initialize ReferenceQC with updated_sample_sig
        qc = ReferenceQC(
            sample_sig=updated_sample_sig,
            reference_sig=self.autosomal_genome_sig,
            amplicon_sig=self.amplicon_sig,
            enable_logging=True
        )

        # Calculate sex chromosome metrics
        metrics = qc.calculate_sex_chrs_metrics(genome_and_chr_to_sig=self.genome_and_chr_signatures)

        # Verify metrics
        self.assertIn("X-Ploidy score", metrics)
        self.assertIn("Y-Coverage", metrics)

        # Calculate expected X-Ploidy score
        # X-Ploidy = (mean_abundance_x / mean_abundance_autosomal) * (len(autosomal_genome_after_removal) / len(sex_x_sig))
        # mean_abundance_x = 4.0
        # mean_abundance_autosomal = 3.0
        # len(autosomal_genome_after_removal) = 6
        # len(sex_x_sig) = 3
        # X-Ploidy = (4.0 / 3.0) * (6 / 3) = 1.3333 * 2 = 2.6667
        expected_xploidy = (4.0 / 3.0) * (6 / 3)
        self.assertAlmostEqual(metrics["X-Ploidy score"], expected_xploidy, places=4)

        # Calculate expected Y-Coverage
        # Y-Coverage = (len(Y in sample) / len(Y specific)) / (len(autosomal in sample) / len(autosomal specific))
        # len(Y in sample) = 0
        # len(Y specific) = 3
        # len(autosomal in sample) = 5
        # len(autosomal specific) = 6
        # Y-Coverage = (0/3) / (5/6) = 0 / 0.8333 = 0.0
        expected_ycoverage = 0.0
        self.assertAlmostEqual(metrics["Y-Coverage"], expected_ycoverage, places=4)
        
    def test_nonref_consume_from_vars_basic(self):
        """
        Test that nonref_consume_from_vars correctly assigns non-reference k-mers to variables in vars_order.
        """
        # Create a sample signature with some non-reference k-mers
        # Reference signature has [10,20,30,40,50,60]
        # Sample signature has [10,20,30,40,50,70,80,90]
        sample_sig_nonref = self.create_test_signature(
            hashes=[10, 20, 30, 40, 50, 70, 80, 90],
            abundances=[1, 2, 3, 4, 5, 6, 7, 8],
            name="test_sample_nonref",
            sig_type=SigType.SAMPLE
        )
        # Reference signature as per setUp: [10,20,30,40,50,60]
        # Non-reference k-mers: [70,80,90]
        
        # Define variables with overlapping k-mers
        vars_signatures = {
            "var_A": self.create_test_signature(
                hashes=[70, 80],
                abundances=[6, 7],
                name="var_A",
                sig_type=SigType.SAMPLE
            ),
            "var_B": self.create_test_signature(
                hashes=[80, 90],
                abundances=[7, 8],
                name="var_B",
                sig_type=SigType.SAMPLE
            )
        }
        vars_order = ["var_A", "var_B"]
        
        qc = ReferenceQC(
            sample_sig=sample_sig_nonref,
            reference_sig=self.reference_sig,
            amplicon_sig=None,
            enable_logging=False
        )
        
        nonref_stats = qc.nonref_consume_from_vars(vars=vars_signatures, vars_order=vars_order)
        
        # Expected:
        # var_A consumes [70,80]: total abundance = 6 + 7 = 13
        # Coverage index for var_A: 2 / 3 ≈ 0.6667
        # var_B consumes [90]: total abundance = 8
        # Coverage index for var_B: 1 / 3 ≈ 0.3333
        # non-var: 0
        expected_stats = {
            "var_A non-genomic total k-mer abundance": 13,
            "var_A non-genomic coverage index": 2 / 3,
            "var_B non-genomic total k-mer abundance": 8,
            "var_B non-genomic coverage index": 1 / 3,
            "non-var non-genomic total k-mer abundance": 0,
            "non-var non-genomic coverage index": 0 / 3
        }
        
        # Verify the stats
        for key, value in expected_stats.items():
            self.assertAlmostEqual(nonref_stats.get(key, None), value, places=4, msg=f"Mismatch in {key}")
        
    def test_nonref_consume_from_vars_overlapping_vars(self):
        """
        Test that nonref_consume_from_vars handles overlapping variables correctly, consuming k-mers only once.
        """
        # Create a sample signature with some non-reference k-mers
        sample_sig_nonref = self.create_test_signature(
            hashes=[70, 80, 90, 100],
            abundances=[6, 7, 8, 9],
            name="test_sample_nonref_overlap",
            sig_type=SigType.SAMPLE
        )
        # Non-reference k-mers: [70,80,90,100]
        
        # Define variables with overlapping k-mers
        vars_signatures = {
            "var_A": self.create_test_signature(
                hashes=[70, 80],
                abundances=[6, 7],
                name="var_A",
                sig_type=SigType.SAMPLE
            ),
            "var_B": self.create_test_signature(
                hashes=[80, 90],
                abundances=[7, 8],
                name="var_B",
                sig_type=SigType.SAMPLE
            ),
            "var_C": self.create_test_signature(
                hashes=[90, 100],
                abundances=[8, 9],
                name="var_C",
                sig_type=SigType.SAMPLE
            )
        }
        vars_order = ["var_A", "var_B", "var_C"]
        
        qc = ReferenceQC(
            sample_sig=sample_sig_nonref,
            reference_sig=self.reference_sig,
            amplicon_sig=None,
            enable_logging=False
        )
        
        nonref_stats = qc.nonref_consume_from_vars(vars=vars_signatures, vars_order=vars_order)
        
        # Expected:
        # var_A consumes [70,80]: total abundance = 6 + 7 = 13
        # Coverage index for var_A: 2 / 4 = 0.5
        # var_B consumes [90]: total abundance = 8
        # Coverage index for var_B: 1 / 4 = 0.25
        # var_C consumes [100]: total abundance = 9
        # Coverage index for var_C: 1 / 4 = 0.25
        # non-var: 0
        expected_stats = {
            "var_A non-genomic total k-mer abundance": 13,
            "var_A non-genomic coverage index": 2 / 4,
            "var_B non-genomic total k-mer abundance": 8,
            "var_B non-genomic coverage index": 1 / 4,
            "var_C non-genomic total k-mer abundance": 9,
            "var_C non-genomic coverage index": 1 / 4,
            "non-var non-genomic total k-mer abundance": 0,
            "non-var non-genomic coverage index": 0 / 4
        }
        
        # Verify the stats
        for key, value in expected_stats.items():
            self.assertAlmostEqual(nonref_stats.get(key, None), value, places=4, msg=f"Mismatch in {key}")
        
    def test_nonref_consume_from_vars_empty_vars(self):
        """
        Test that nonref_consume_from_vars handles empty vars correctly.
        """
        # Create a sample signature with some non-reference k-mers
        sample_sig_nonref = self.create_test_signature(
            hashes=[70, 80, 90],
            abundances=[6, 7, 8],
            name="test_sample_nonref_empty_vars",
            sig_type=SigType.SAMPLE
        )
        # Non-reference k-mers: [70,80,90]
        
        # Define empty vars
        vars_signatures = {}
        vars_order = []
        
        qc = ReferenceQC(
            sample_sig=sample_sig_nonref,
            reference_sig=self.reference_sig,
            amplicon_sig=None,
            enable_logging=False
        )
        
        nonref_stats = qc.nonref_consume_from_vars(vars=vars_signatures, vars_order=vars_order)
        
        # Expected:
        # non-var consumes all k-mers
        expected_stats = {
            "non-var non-genomic total k-mer abundance": 21,  # 6 + 7 + 8
            "non-var non-genomic coverage index": 3 / 3  # 3 k-mers
        }
        
        # Verify the stats
        self.assertEqual(len(nonref_stats), 2)
        self.assertAlmostEqual(nonref_stats.get("non-var non-genomic total k-mer abundance"), 21, places=4)
        self.assertAlmostEqual(nonref_stats.get("non-var non-genomic coverage index"), 1.0, places=4)
        
    def test_nonref_consume_from_vars_vars_order_not_matching_vars(self):
        """
        Test that nonref_consume_from_vars raises ValueError when vars_order contains variables not in vars.
        """
        # Create a sample signature with some non-reference k-mers
        sample_sig_nonref = self.create_test_signature(
            hashes=[70, 80, 90],
            abundances=[6, 7, 8],
            name="test_sample_nonref_invalid_order",
            sig_type=SigType.SAMPLE
        )
        # Non-reference k-mers: [70,80,90]
        
        # Define vars with one variable
        vars_signatures = {
            "var_A": self.create_test_signature(
                hashes=[70],
                abundances=[6],
                name="var_A",
                sig_type=SigType.SAMPLE
            )
        }
        # vars_order includes a variable not in vars_signatures
        vars_order = ["var_A", "var_B"]
        
        qc = ReferenceQC(
            sample_sig=sample_sig_nonref,
            reference_sig=self.reference_sig,
            amplicon_sig=None,
            enable_logging=False
        )
        
        with self.assertRaises(ValueError):
            qc.nonref_consume_from_vars(vars=vars_signatures, vars_order=vars_order)
    
    def test_nonref_consume_from_vars_no_nonref_kmers(self):
        """
        Test that nonref_consume_from_vars returns empty dict when there are no non-reference k-mers.
        """
        # Create a sample signature identical to reference signature
        sample_sig_identical = self.create_test_signature(
            hashes=[10,20,30,40,50,60],
            abundances=[1,2,3,4,5,6],
            name="test_sample_identical",
            sig_type=SigType.SAMPLE
        )
        # Non-reference k-mers: none
        
        # Define variables
        vars_signatures = {
            "var_A": self.create_test_signature(
                hashes=[70],
                abundances=[7],
                name="var_A",
                sig_type=SigType.SAMPLE
            )
        }
        vars_order = ["var_A"]
        
        qc = ReferenceQC(
            sample_sig=sample_sig_identical,
            reference_sig=self.reference_sig,
            amplicon_sig=None,
            enable_logging=False
        )
        
        nonref_stats = qc.nonref_consume_from_vars(vars=vars_signatures, vars_order=vars_order)
        
        # Expected: empty dict since no non-reference k-mers
        self.assertEqual(nonref_stats, {})
        
    def test_nonref_consume_from_vars_all_kmers_consumed(self):
        """
        Test that nonref_consume_from_vars correctly reports when all non-reference k-mers are consumed by variables.
        """
        # Create a sample signature with some non-reference k-mers
        sample_sig_nonref = self.create_test_signature(
            hashes=[70, 80, 90],
            abundances=[6, 7, 8],
            name="test_sample_nonref_all_consumed",
            sig_type=SigType.SAMPLE
        )
        # Non-reference k-mers: [70,80,90]
        
        # Define variables that consume all non-reference k-mers
        vars_signatures = {
            "var_A": self.create_test_signature(
                hashes=[70, 80],
                abundances=[6, 7],
                name="var_A",
                sig_type=SigType.SAMPLE
            ),
            "var_B": self.create_test_signature(
                hashes=[90],
                abundances=[8],
                name="var_B",
                sig_type=SigType.SAMPLE
            )
        }
        vars_order = ["var_A", "var_B"]
        
        qc = ReferenceQC(
            sample_sig=sample_sig_nonref,
            reference_sig=self.reference_sig,
            amplicon_sig=None,
            enable_logging=False
        )
        
        nonref_stats = qc.nonref_consume_from_vars(vars=vars_signatures, vars_order=vars_order)
        
        # Expected:
        # var_A consumes [70,80]: total abundance = 6 + 7 = 13
        # Coverage index for var_A: 2 / 3 ≈ 0.6667
        # var_B consumes [90]: total abundance = 8
        # Coverage index for var_B: 1 / 3 ≈ 0.3333
        # non-var: 0
        expected_stats = {
            "var_A non-genomic total k-mer abundance": 13,
            "var_A non-genomic coverage index": 2 / 3,
            "var_B non-genomic total k-mer abundance": 8,
            "var_B non-genomic coverage index": 1 / 3,
            "non-var non-genomic total k-mer abundance": 0,
            "non-var non-genomic coverage index": 0 / 3
        }
        
        # Verify the stats
        for key, value in expected_stats.items():
            self.assertAlmostEqual(nonref_stats.get(key, None), value, places=4, msg=f"Mismatch in {key}")





if __name__ == '__main__':
    unittest.main()