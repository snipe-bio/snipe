import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, poisson, chisquare
from scipy.optimize import OptimizeWarning, curve_fit, fsolve
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

from snipe.api.enums import SigType
from snipe.api.snipe_sig import SnipeSig

# Keep my important commits and do not remove or rename this function:
def check_poisson_fit(frequencies):
    """
    Check the goodness-of-fit of frequencies to a Poisson distribution
    using a chi-square test. Returns the p-value.
    """
    lambda_hat = np.mean(frequencies)

    # Generate expected Poisson probabilities
    max_val = max(frequencies)
    x_vals = np.arange(0, max_val + 1)
    expected_probs = poisson.pmf(x_vals, lambda_hat)

    # Compute expected counts
    total_count = len(frequencies)
    expected_counts = expected_probs * total_count

    # Ensure no expected count is too small (combine small bins if needed)
    threshold = 5  # Minimum expected count threshold
    observed_counts, _ = np.histogram(frequencies, bins=np.arange(max_val + 2))

    mask = expected_counts >= threshold
    if not np.all(mask):
        observed_counts = np.append(observed_counts[mask], np.sum(observed_counts[~mask]))
        expected_counts = np.append(expected_counts[mask], np.sum(expected_counts[~mask]))

    # Normalize counts to ensure sum consistency
    expected_counts *= np.sum(observed_counts) / np.sum(expected_counts)

    # Perform chi-square test
    chi2_stat, p_value = chisquare(observed_counts, expected_counts)
    return p_value


class MultiSigReferenceQC:
    """
    A refactored and modular version of the original class.
    It retains the same structure, logic, and outputs as requested,
    but reorganizes code to remove duplication and clarify flow.

    # !Important commits and comments are preserved.
    # The code is kept in a single class since class data are large.
    """

    def __init__(
        self,
        *,
        reference_sig,
        amplicon_sig=None,
        ychr=None,
        varsigs=None,
        enable_logging=False,
        export_varsigs=False,
        repetitive_aware=True,
        custom_logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initialize the MultiSigReferenceQC with references, optional amplicon, sex chromosomes, etc.
        """
        # -------------- Initialize Logger --------------
        self.logger = custom_logger or logging.getLogger(__name__)
        self._initialize_logger(enable_logging)

        self.repetitive_aware_flag = repetitive_aware

        # Initialize split cache
        self._split_cache: Dict[int, List[Any]] = {}
        self.logger.debug("Initialized split cache.")

        # Log all passed parameters except self
        self.logger.debug("passed parameters:")
        for key, value in locals().items():
            if key != "self":
                self.logger.debug("\t%s: %s", key, value)

        # -------------- Validate and Store Signatures --------------
        self.reference_sig = reference_sig
        self.REFERENCE_UNIQUE_KMERS = len(reference_sig)
        self.REFERENCE_TOTAL_KMERS = reference_sig.total_abundance        
        self.amplicon_sig = amplicon_sig
        self.AMPLICON_UNIQUE_KMERS = len(amplicon_sig) if amplicon_sig else 0
        self.AMPLICON_TOTAL_KMERS = amplicon_sig.total_abundance if amplicon_sig else 0
        self._validate_signatures(reference_sig, amplicon_sig, ychr)

        # -------------- Prepare Chromosome-Specific Signatures --------------
        self.specific_chr_to_sig: Optional[Dict[str, Any]] = reference_sig.chr_to_sig
        if ychr is not None and self.specific_chr_to_sig is not None:
            self.logger.debug("Y chromosome signature provided and passed to the specific_kmers function.")
            self.specific_chr_to_sig["sex-y"] = ychr

        self._prepare_chromosome_signatures()

        # -------------- Prepare Variance Signatures --------------
        self.variance_sigs: Optional[List[Any]] = None
        if varsigs is not None:
            self._prepare_variance_signatures(varsigs)

        self.logger.debug("Chromosome specific signatures provided.")
        self.flag_activate_sex_metrics = True

        self.enable_logging = enable_logging
        self.export_varsigs = export_varsigs
        self.sample_to_stats = {}

        # ! New: repetitive k-mers awareness
        self.repetitive_hashes_sig = self.reference_sig.copy()
        self.repetitive_hashes_sig.trim_singletons()
        self.logger.debug(f"Repetitive k-mers: {self.repetitive_hashes_sig}")

        if len(self.repetitive_hashes_sig) > 0:
            self.reference_without_repeats = self.reference_sig - self.repetitive_hashes_sig
        else:
            self.reference_without_repeats = self.reference_sig.copy()

        self.logger.debug(f"Reference without repeats: {len(self.reference_without_repeats)}")

    # ----------------------- PRIVATE HELPER METHODS -----------------------

    def _initialize_logger(self, enable_logging: bool) -> None:
        """
        Configure logger level and handler.
        """
        if enable_logging:
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.hasHandlers():
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
            self.logger.debug("Logging is enabled for ReferenceQC.")
        else:
            self.logger.setLevel(logging.CRITICAL)

    def _validate_signatures(self, reference_sig, amplicon_sig, ychr) -> None:
        """
        Validate that the reference signature is of type GENOME
        and that amplicon_sig and ychr (if provided) have acceptable types.
        """
        if reference_sig.sigtype != SigType.GENOME:
            self.logger.error("Invalid signature type for reference_sig: %s", reference_sig.sigtype)
            raise ValueError(f"reference_sig must be of type {SigType.GENOME}, got {reference_sig.sigtype}")

        if amplicon_sig is not None and amplicon_sig.sigtype != SigType.AMPLICON:
            self.logger.error("Invalid signature type for amplicon_sig: %s", amplicon_sig.sigtype)
            raise ValueError(f"amplicon_sig must be of type {SigType.AMPLICON}, got {amplicon_sig.sigtype}")

        if ychr and not hasattr(ychr, "sigtype"):
            self.logger.error("Invalid signature type for ychr: %s", ychr)
            raise ValueError("ychr must be a valid SnipeSig type reference.")

    def _prepare_chromosome_signatures(self) -> None:
        """
        Deduplicate and remove unwanted signatures (e.g. those ending with '-snipegenome')
        from the chromosome-specific signatures.
        """
        if self.specific_chr_to_sig is not None and len(self.specific_chr_to_sig) > 0:
            self.logger.debug("Computing specific chromosome hashes for %s.", 
                              ",".join(self.specific_chr_to_sig.keys()))
            self.logger.debug(
                f"\t-All hashes for chromosomes before getting unique sigs "
                f"{len(SnipeSig.sum_signatures(list(self.specific_chr_to_sig.values())))}"
            )
            self.specific_chr_to_sig = SnipeSig.get_unique_signatures({
                sig_name: sig for sig_name, sig in self.specific_chr_to_sig.items()
                if not sig_name.endswith("-snipegenome")
            })
            self.logger.debug(
                f"\t-All hashes for chromosomes after getting unique sigs "
                f"{len(SnipeSig.sum_signatures(list(self.specific_chr_to_sig.values())))}"
            )

    def _prepare_variance_signatures(self, varsigs) -> None:
        """
        Validate that each variance signature matches the reference signature's ksize and scale.
        """
        self.logger.debug("Variance signatures provided.")
        for sig in varsigs:
            if sig.ksize != self.reference_sig.ksize:
                self.logger.error(
                    "K-mer sizes do not match: varsigs.ksize=%d vs reference_sig.ksize=%d",
                    sig.ksize, self.reference_sig.ksize,
                )
                raise ValueError(f"varsigs ksize ({sig.ksize}) does not match reference_sig ksize ({self.reference_sig.ksize}).")
            if sig.scale != self.reference_sig.scale:
                self.logger.error(
                    "Scale values do not match: varsigs.scale=%d vs reference_sig.scale=%d",
                    sig.scale, self.reference_sig.scale,
                )
                raise ValueError(f"varsigs scale ({sig.scale}) does not match reference_sig scale ({self.reference_sig.scale}).")
        self.variance_sigs = varsigs

    def _fit_two_component_gmm(self, abundances) -> Tuple[float, float, float]:
        """
        Fit a 2-component Gaussian Mixture Model in log10 space and return d_prime, overlap, threshold.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                with np.errstate(divide="raise", invalid="raise"):
                    abundances = np.array(abundances)
                    abundances = abundances[abundances > 0]
                    if len(abundances) == 0:
                        return 0, 0, 0

                    log_abund = np.log10(abundances).reshape(-1, 1)
                    if len(np.unique(log_abund)) < 2:
                        return 0, 0, 0

                    gmm = GaussianMixture(n_components=2, random_state=42)
                    gmm.fit(log_abund)

                    means = gmm.means_.flatten()
                    sigmas = np.sqrt(gmm.covariances_.flatten())
                    weights = gmm.weights_.flatten()

                    if np.any(np.isclose(sigmas, 0)) or np.any(np.isclose(weights, 0)):
                        return 0, 0, 0

                    if means[0] < means[1]:
                        mu_low, sigma_low, w_low = means[0], sigmas[0], weights[0]
                        mu_high, sigma_high, w_high = means[1], sigmas[1], weights[1]
                    else:
                        mu_low, sigma_low, w_low = means[1], sigmas[1], weights[1]
                        mu_high, sigma_high, w_high = means[0], sigmas[0], weights[0]

                    if (np.isclose(sigma_low, 0) or np.isclose(sigma_high, 0) or 
                        np.isclose(w_low, 0) or np.isclose(w_high, 0)):
                        return 0, 0, 0

                    D = np.log((w_low * sigma_high) / (w_high * sigma_low))
                    A = 1 / (2 * sigma_low**2) - 1 / (2 * sigma_high**2)
                    B = -mu_low / (sigma_low**2) + mu_high / (sigma_high**2)
                    C = (mu_low**2 / (2 * sigma_low**2)) - (mu_high**2 / (2 * sigma_high**2)) - D

                    if np.isclose(B, 0):
                        return 0, 0, 0

                    if np.abs(A) < 1e-8:
                        intersection_log = -C / B
                    else:
                        roots = np.roots([A, B, C])
                        real_roots = roots[np.isreal(roots)].real
                        if real_roots.size == 0:
                            return 0, 0, 0
                        intersection_log = None
                        for r in real_roots:
                            if r >= min(mu_low, mu_high) and r <= max(mu_low, mu_high):
                                intersection_log = r
                                break
                        if intersection_log is None:
                            intersection_log = real_roots[np.argmin(np.abs(real_roots - (mu_low + mu_high) / 2))]
                    threshold = 10 ** intersection_log
                    d_prime = abs(mu_high - mu_low) / np.sqrt((sigma_low**2 + sigma_high**2) / 2)
                    overlap = 2 * norm.cdf(-d_prime / 2)
                    return d_prime, overlap, threshold
        except Exception:
            return 0, 0, 0

    @staticmethod
    def _bases_covered(L, lambda_, G):
        """
        Helper: returns the expected covered bases.
        """
        return L * (1 - np.exp(-lambda_ * (L / G)))

    @staticmethod
    def _solve_for_L(L_guess, lambda_, G, N_empirical):
        """
        Solve for L such that the expected covered bases matches N_empirical.
        """
        return MultiSigReferenceQC._bases_covered(L_guess, lambda_, G) - N_empirical

    def _grok_calcs(self, lambda_current, coverage_index, prefix, reference_len):
        """
        Compute coverage estimates via fsolve using _solve_for_L.
        """
        local_stats = {}
        G = reference_len
        N_empirical = reference_len * coverage_index
        L_initial_guess = N_empirical
        L_estimated = fsolve(self._solve_for_L, L_initial_guess, args=(lambda_current, G, N_empirical))[0]
        local_stats[f"{prefix}grok_L_estimate"] = L_estimated
        local_stats[f"{prefix}grok_sat_point"] = min(1, L_estimated / reference_len)
        return local_stats

    @staticmethod
    def _saturation_model(x, a, b):
        """
        Saturation model: f(x) = a*x/(b+x)
        """
        return a * x / (b + x)

    @staticmethod
    def _sort_chromosomes(chrom_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sort chromosome keys based on the numeric part (if available) or lexicographically.
        """
        sorted_keys = sorted(chrom_dict, key=lambda x: (int(x.split("-")[1]) if len(x.split("-")) > 1 and x.split("-")[1].isdigit() else float('inf'), x))
        return {k: chrom_dict[k] for k in sorted_keys}


    def _predict_roi_stats(self, sample_sig, kmers_to_bases: float, predict_extra_folds: list) -> dict:
        roi_stats = {}
        predicted_fold_coverage = {}
        predicted_fold_delta_coverage = {}
        predicted_unique_hashes = {}

        np.random.seed(42)

        _sample_sig_genome = sample_sig & self.reference_sig
        abundances = np.array(_sample_sig_genome.abundances, dtype=float)
        N = len(abundances)
        total_reference_kmers = len(self.reference_sig)

        if N == 0 or total_reference_kmers == 0:
            self.logger.warning("Sample or reference has zero k-mers. Returning empty predictions.")
            for extra_fold in predict_extra_folds:
                predicted_fold_coverage[f"Predicted coverage with {extra_fold} extra folds"] = 0.0
                predicted_fold_delta_coverage[f"Predicted delta coverage with {extra_fold} extra folds"] = 0.0
            predicted_unique_hashes = {
                "Predicted genomic unique k-mers": 0,
                "Delta genomic unique k-mers": 0,
                "Adjusted genome coverage index": 0.0,
            }
            roi_stats.update(predicted_fold_coverage)
            roi_stats.update(predicted_fold_delta_coverage)
            roi_stats.update(predicted_unique_hashes)
            return roi_stats

        abundances.sort()
        cum_kmers = np.cumsum(abundances)
        cum_unique = np.arange(1, N + 1)
        cum_fraction = cum_unique / float(total_reference_kmers)

        x_data = np.concatenate([[0], cum_kmers])
        y_data = np.concatenate([[0], cum_fraction])

        # Model definitions
        def michaelis_menten(x, K): return x / (K + x)
        def hill(x, K, n): return (x**n) / (K**n + x**n)
        def exponential(x, lam): return 1.0 - np.exp(-lam * x)
        def weibull(x, lam, k): return 1.0 - np.exp(-(lam * x)**k)

        models = {
            'Michaelis-Menten': (michaelis_menten, [np.median(cum_kmers)]),
            'Hill': (hill, [np.median(cum_kmers), 1.0]),
            'Exponential': (exponential, [-np.log(1 - y_data[-1]) / max(x_data[-1], 1e-6)]),
            'Weibull': (weibull, [1e-3, 1.0]),
        }

        best_model = None
        best_params = None
        best_aic = np.inf
        n = len(x_data)

        def compute_aic(rss, k):
            return n * np.log(rss / n + 1e-12) + 2 * k

        for name, (model_func, p0) in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    params, _ = curve_fit(model_func, x_data, y_data, p0=p0, bounds=(1e-8, np.inf), maxfev=10000)
                pred = model_func(x_data, *params)
                rss = np.sum((pred - y_data)**2)
                aic = compute_aic(rss, len(params))
                if aic < best_aic:
                    best_aic = aic
                    best_model = (model_func, params)
            except Exception:
                continue

        if best_model is None:
            self.logger.warning("All model fittings failed. Falling back to Michaelis-Menten heuristic.")
            best_model = (michaelis_menten, [np.median(cum_kmers)])

        model_func, params = best_model

        # Current sequencing abundance
        if sample_sig._bases_count is not None and sample_sig._bases_count > 0:
            corrected_total_abundance = sample_sig._bases_count / sample_sig.scale
        else:
            corrected_total_abundance = float(np.sum(abundances))

        # Predict now
        predicted_now = model_func(corrected_total_abundance, *params)
        predicted_now = np.clip(predicted_now, 0.0, 1.0)

        current_unique_hashes = cum_unique[-1]
        predicted_genomic_unique_hashes = predicted_now * total_reference_kmers
        delta_unique_hashes = predicted_genomic_unique_hashes - current_unique_hashes
        adjusted_genome_coverage_index = predicted_genomic_unique_hashes / total_reference_kmers if total_reference_kmers else 0.0

        predicted_unique_hashes = {
            "Predicted genomic unique k-mers": predicted_genomic_unique_hashes,
            "Delta genomic unique k-mers": delta_unique_hashes,
            "Adjusted genome coverage index": adjusted_genome_coverage_index,
        }

        # Predict for each fold
        for extra_fold in predict_extra_folds:
            if extra_fold < 1:
                continue
            predicted_total = corrected_total_abundance * (1 + extra_fold)
            pred_coverage = model_func(predicted_total, *params)
            pred_coverage = np.clip(pred_coverage, 0.0, 1.0)
            predicted_fold_coverage[f"Predicted coverage with {extra_fold} extra folds"] = pred_coverage
            _delta = pred_coverage - predicted_now
            predicted_fold_delta_coverage[f"Predicted delta coverage with {extra_fold} extra folds"] = max(0.0, _delta)

        roi_stats.update(predicted_fold_coverage)
        roi_stats.update(predicted_fold_delta_coverage)
        roi_stats.update(predicted_unique_hashes)
        roi_stats["Best model"] = best_model[0].__name__
        return roi_stats

    def _exp_predict_roi_stats(self, sample_sig, kmers_to_bases: float, predict_extra_folds: list) -> dict:
        """
        - Uses a 2-parameter Weibull curve with asymptote fixed at 1 (100 % genome coverage)
        """
        roi_stats: Dict[str, Any] = {}
        pred_fold_cov: Dict[str, float] = {}
        pred_fold_delta: Dict[str, float] = {}
        pred_unique_hashes: Dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # 0. consistent RNG for reproducibility                              #
        # ------------------------------------------------------------------ #
        np.random.seed(42)

        # ------------------------------------------------------------------ #
        # 1. extract genome‑intersected abundances                            #
        # ------------------------------------------------------------------ #
        _sample_sig_genome = sample_sig & self.reference_sig
        abundances = np.asarray(_sample_sig_genome.abundances, dtype=float)
        N          = abundances.size
        total_reference_kmers = len(self.reference_sig)

        if N == 0 or total_reference_kmers == 0:
            self.logger.warning("Sample or reference has zero k-mers. Returning empty predictions.")
            for fld in predict_extra_folds:
                pred_fold_cov[f"Predicted coverage with {fld} extra folds"] = 0.0
                pred_fold_delta[f"Predicted delta coverage with {fld} extra folds"] = 0.0
            pred_unique_hashes = {
                "Predicted genomic unique k-mers": 0,
                "Delta genomic unique k-mers": 0,
                "Adjusted genome coverage index": 0.0,
            }
            roi_stats.update(pred_fold_cov)
            roi_stats.update(pred_fold_delta)
            roi_stats.update(pred_unique_hashes)
            return roi_stats

        # ------------------------------------------------------------------ #
        # 2. build empirical rarefaction curve                              #
        # ------------------------------------------------------------------ #
        abundances.sort()                          # ascending
        cum_abund = np.cumsum(abundances)          # cumulative total abundance per new k-mer discovered
        cum_unique = np.arange(1, N + 1)           # cumulative count of unique k-mers discovered
        y_frac = cum_unique / float(total_reference_kmers)  # coverage fraction

        x_data = np.concatenate([[0.0], cum_abund])
        y_data = np.concatenate([[0.0], y_frac])

        # ------------------------------------------------------------------ #
        # 3. fit 2-parameter Weibull (asymptote = 1)                        #
        # ------------------------------------------------------------------ #
        def _weibull(x, lam, k):
            return 1.0 - np.exp(-np.power(x / lam, k))

        lam_init = max(np.median(cum_abund), 1e-3)
        k_init   = 0.8
        p0       = [lam_init, k_init]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, _ = curve_fit(
                    _weibull,
                    x_data,
                    y_data,
                    p0=p0,
                    bounds=(1e-10, np.inf),
                    maxfev=20_000,
                )
        except RuntimeError:
            # fall back to exponential (k=1)
            params = [lam_init, 1.0]
        lam_fit, k_fit = params

        # helper
        def predict_cov(total_abund: float) -> float:
            val = _weibull(total_abund, lam_fit, k_fit)
            return float(min(max(val, 0.0), 1.0))

        # ------------------------------------------------------------------ #
        # 4. compute current abundance & baseline coverage                  #
        # ------------------------------------------------------------------ #
        if getattr(sample_sig, "_bases_count", None) and sample_sig._bases_count > 0:
            corrected_total_abund = sample_sig._bases_count / sample_sig.scale
        else:
            corrected_total_abund = float(abundances.sum())

        cov_now = predict_cov(corrected_total_abund)

        # unique k‑mers predictions
        current_unique_kmers = cum_unique[-1]
        predicted_unique = cov_now * total_reference_kmers
        delta_unique     = predicted_unique - current_unique_kmers
        adj_cov_index    = predicted_unique / total_reference_kmers if total_reference_kmers else 0.0

        pred_unique_hashes = {
            "Predicted genomic unique k-mers": predicted_unique,
            "Delta genomic unique k-mers": delta_unique,
            "Adjusted genome coverage index": adj_cov_index,
        }

        # ------------------------------------------------------------------ #
        # 5. predict coverage for requested extra folds                     #
        # ------------------------------------------------------------------ #
        for fld in predict_extra_folds:
            if fld < 1:
                continue
            new_total = corrected_total_abund * (1 + fld)
            cov_pred  = predict_cov(new_total)
            pred_fold_cov[f"Predicted coverage with {fld} extra folds"] = cov_pred
            pred_fold_delta[f"Predicted delta coverage with {fld} extra folds"] = max(0.0, cov_pred - cov_now)

        # ------------------------------------------------------------------ #
        # 6. assemble output                                                #
        # ------------------------------------------------------------------ #
        roi_stats.update(pred_fold_cov)
        roi_stats.update(pred_fold_delta)
        roi_stats.update(pred_unique_hashes)
        return roi_stats

    # ----------------------- MAIN PUBLIC METHOD -----------------------

    def process_sample(self, sample_sig, predict_extra_folds: Optional[List[int]] = None, advanced: Optional[bool] = False) -> Dict[str, Any]:
        """
        Process the sample signature to compute a large variety of stats:
          - Sample stats
          - Genome stats (with repetitive-aware corrections)
          - Amplicon stats
          - Error/contamination indices
          - Advanced metrics (if advanced=True)
          - Chromosome statistics
          - Variance signature coverage
          - ROI coverage predictions

        Returns a dictionary with aggregated results.
        """
        sample_stats: Dict[str, Any] = {}
        genome_stats: Dict[str, Any] = {}
        amplicon_stats: Dict[str, Any] = {}
        advanced_stats: Dict[str, Any] = {}
        chrs_stats: Dict[str, Any] = {}
        sex_stats: Dict[str, Any] = {}
        predicted_error_contamination_index: Dict[str, Any] = {}
        vars_nonref_stats: Dict[str, Any] = {}
        roi_stats: Dict[str, Any] = {}
        chr_to_mean_abundance: Dict[str, np.float64] = {}

        # ============= SAMPLE Verification =============
        self.logger.debug("Validating ksize and scale across signatures.")
        if sample_sig.ksize != self.reference_sig.ksize:
            self.logger.error(
                "K-mer sizes do not match: sample_sig.ksize=%d vs reference_sig.ksize=%d",
                sample_sig.ksize, self.reference_sig.ksize,
            )
            raise ValueError(
                f"sample_sig ksize ({sample_sig.ksize}) does not match reference_sig ksize ({self.reference_sig.ksize})."
            )
        if sample_sig.scale != self.reference_sig.scale:
            self.logger.error(
                "Scale values do not match: sample_sig.scale=%d vs reference_sig.scale=%d",
                sample_sig.scale, self.reference_sig.scale,
            )
            raise ValueError(
                f"sample_sig scale ({sample_sig.scale}) does not match reference_sig scale ({self.reference_sig.scale})."
            )
        if self.amplicon_sig is not None:
            if self.amplicon_sig.ksize != sample_sig.ksize:
                self.logger.error(
                    "K-mer sizes do not match: amplicon_sig.ksize=%d vs sample_sig.ksize=%d",
                    self.amplicon_sig.ksize, sample_sig.ksize,
                )
                raise ValueError(
                    f"amplicon_sig ksize ({self.amplicon_sig.ksize}) does not match sample_sig ksize ({sample_sig.ksize})."
                )
            if self.amplicon_sig.scale != sample_sig.scale:
                self.logger.error(
                    "Scale values do not match: amplicon_sig.scale=%d vs sample_sig.scale=%d",
                    self.amplicon_sig.scale, sample_sig.scale,
                )
                raise ValueError(
                    f"amplicon_sig scale ({self.amplicon_sig.scale}) does not match sample_sig scale ({sample_sig.scale})."
                )
        from snipe.api.enums import SigType
        if sample_sig._type != SigType.SAMPLE:
            self.logger.error(
                "Invalid signature type for sample_sig: %s | %s",
                sample_sig.sigtype, sample_sig._type,
            )
            raise ValueError(
                f"sample_sig must be of type {SigType.SAMPLE}, got {sample_sig.sigtype}"
            )
        self.logger.debug("All signatures have matching ksize and scale.")

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
            "snipe valid k-mers": sample_sig.valid_kmers,
        })

        FRACMINHASH_PRECISION = 1.00
        if sample_sig.valid_kmers is not None and sample_sig.valid_kmers > 0:
            FRACMINHASH_PRECISION = (sample_sig.total_abundance * sample_sig.scale) / sample_sig.valid_kmers

        kmer_yield = 1
        kmers_to_bases = 1
        if sample_sig.bases is not None and sample_sig.bases > 0:
            kmers_to_bases = ((sample_sig.total_abundance * sample_sig.scale) / sample_sig.bases
                              if sample_sig.total_abundance is not None and sample_sig.total_abundance > 0 else 0)
            if sample_sig.valid_kmers is not None and sample_sig.valid_kmers > 0:
                kmer_yield = sample_sig.valid_kmers / sample_sig.bases

        normalized_fracminhash_precision = FRACMINHASH_PRECISION / kmer_yield
        sample_stats["kmer_yield"] = kmer_yield
        sample_stats["FRACMINHASH_PRECISION"] = FRACMINHASH_PRECISION
        sample_stats["NORMALIZED_FRACMINHASH_PRECISION"] = normalized_fracminhash_precision
        sample_stats["kmers_to_bases"] = kmers_to_bases

        # ============= GENOME STATS =============
        self.logger.debug("Calculating genome statistics.")
        sample_genome = sample_sig & self.reference_sig
        sample_genome_stats = sample_genome.get_sample_stats
        genome_stats.update({
            "Genomic unique k-mers": sample_genome_stats["num_hashes"],
            "Genomic k-mers total abundance": sample_genome_stats["total_abundance"],
            "Genomic k-mers mean abundance": sample_genome_stats["total_abundance"] / self.REFERENCE_TOTAL_KMERS,
            "Genomic k-mer variance abundance": np.var(sample_genome.abundances),
            "Genomic k-mers mean abundance - no_zero_cov": sample_genome_stats["mean_abundance"],
            "Genomic k-mers median abundance - no_zero_cov": sample_genome_stats["median_abundance"],
            "Genome coverage index": (sample_genome_stats["num_hashes"] / self.REFERENCE_UNIQUE_KMERS
                                      if self.REFERENCE_UNIQUE_KMERS > 0 and sample_genome_stats["num_hashes"] is not None else 0),
            "Mapping index": (sample_genome_stats["total_abundance"] / sample_stats["k-mer total abundance"]
                              if sample_stats.get("k-mer total abundance", 0) > 0 and sample_genome_stats["total_abundance"] is not None else 0),
        })

        if self.repetitive_aware_flag:
            sample_genome_non_repetitive = sample_genome.copy()
            sample_genome_non_repetitive &= self.reference_without_repeats
            abundance_based_sample_genome_stats = sample_genome_non_repetitive.get_sample_stats
            
            ### --------- variance calculation heuristics -------------
            _projected_repfree_genomic_abundance_array_length = len(self.reference_without_repeats)
            # Extend np array with zeros to match reference length
            pad_length = _projected_repfree_genomic_abundance_array_length - len(sample_genome_non_repetitive.abundances)
            _custom_genomic_repfree_abundances = np.concatenate([
                sample_genome_non_repetitive.abundances,
                np.zeros(pad_length)
            ])
            # Remove top 1% of the abundances by value
            _custom_genomic_repfree_abundances = np.sort(_custom_genomic_repfree_abundances)
            _custom_genomic_repfree_abundances = _custom_genomic_repfree_abundances[:int(len(_custom_genomic_repfree_abundances) * 0.99)]


            genome_stats.update({
                "Genomic repfree unique k-mers": abundance_based_sample_genome_stats["num_hashes"],
                "Genomic repfree k-mers total abundance": abundance_based_sample_genome_stats["total_abundance"],
                "Genomic repfree k-mers mean abundance": (abundance_based_sample_genome_stats["total_abundance"] / len(self.reference_without_repeats) if len(self.reference_without_repeats) > 0 and abundance_based_sample_genome_stats["total_abundance"] is not None else 0),
                "Genomic repfree k-mers variance abundance": np.var(_custom_genomic_repfree_abundances),
                "Genomic repfree k-mers mean abundance - no_zero_cov": abundance_based_sample_genome_stats["mean_abundance"],
                "Genomic repfree k-mers median abundance - no_zero_cov": abundance_based_sample_genome_stats["median_abundance"],
                "Genomic repfree k-mers coverage index": (abundance_based_sample_genome_stats["num_hashes"] / len(self.reference_without_repeats)
                                                          if len(self.reference_without_repeats) > 0 and abundance_based_sample_genome_stats["num_hashes"] is not None else 0),
            })
        else:
            abundance_based_sample_genome_stats = sample_genome_stats

        sample_total_abundance = sample_sig.total_abundance
        sample_nonref = sample_sig - self.reference_sig
        sample_nonref_singletons = sample_nonref.count_singletons()
        predicted_error_index = (sample_nonref_singletons / sample_total_abundance
                                  if sample_total_abundance is not None and sample_total_abundance > 0 else 0)

        _genetic_var = 0.1

        if kmers_to_bases > 0 and sample_genome_stats["num_hashes"] is not None:
            genome_stats["tmp corrected genome coverage index"] = (
                1 - (1 - genome_stats["Genome coverage index"]) ** ((1 + predicted_error_index + _genetic_var) / kmer_yield)
            )
            genome_stats["razan corrected genome coverage index"] = (
                1 - (1 - min(genome_stats["Genome coverage index"] * normalized_fracminhash_precision, 1)) ** ((1 + predicted_error_index + _genetic_var) / kmers_to_bases)
            )
                        
            lambda_1_stats = self._grok_calcs(
                genome_stats["Genomic k-mers mean abundance"],
                genome_stats["Genome coverage index"],
                "raw_",
                self.REFERENCE_UNIQUE_KMERS
            )
            genome_stats.update(lambda_1_stats)


            genome_stats["corrected genomic total abundance"] = (
                (sample_genome_stats["total_abundance"] + sample_genome_stats["total_abundance"] * predicted_error_index + sample_genome_stats["total_abundance"] * _genetic_var)
                / kmers_to_bases
                if sample_genome_stats["total_abundance"] is not None else 0
            )
            genome_stats["corrected genomic mean abundance"] = (
                genome_stats["corrected genomic total abundance"] / self.REFERENCE_TOTAL_KMERS
            )
            denominator = genome_stats["razan corrected genome coverage index"] * self.REFERENCE_UNIQUE_KMERS
            if denominator == 0:
                genome_stats["corrected genomic mean abundance - no_zero_cov"] = 0
            else:
                genome_stats["corrected genomic mean abundance - no_zero_cov"] = (
                    genome_stats["corrected genomic total abundance"] / denominator
                )
            corrected_lambda = genome_stats["corrected genomic mean abundance"]
            corrected_cov_index = genome_stats["razan corrected genome coverage index"]
            lambda_1_stats_corr = self._grok_calcs(
                corrected_lambda,
                corrected_cov_index,
                "corrected_",
                self.REFERENCE_UNIQUE_KMERS
            )
            genome_stats.update(lambda_1_stats_corr)
            genome_stats["corrected_mapping_index"] = (
                (sample_genome_stats["total_abundance"] + sample_genome_stats["total_abundance"] * predicted_error_index + sample_genome_stats["total_abundance"] * _genetic_var)
                / sample_stats["k-mer total abundance"]
                if sample_stats.get("k-mer total abundance", 0) > 0 and sample_stats["k-mer total abundance"] is not None else 0
            )
            genome_stats["predicted_base_error_percentage"] = 100 * (predicted_error_index / sample_sig.ksize)
            
            
            if self.repetitive_aware_flag:
                genome_stats["tmp corrected genome repfree coverage index"] = (
                    1 - (1 - genome_stats["Genomic repfree k-mers coverage index"]) ** ((1 + predicted_error_index + _genetic_var) / kmer_yield)
                )
                genome_stats["razan corrected genome repfree coverage index"] = (
                    1 - (1 - min(genome_stats["Genomic repfree k-mers coverage index"] * normalized_fracminhash_precision, 1)) ** ((1 + predicted_error_index + _genetic_var) / kmers_to_bases)
                )
    
                genome_stats["corrected repfree genomic total abundance"] = (
                    (abundance_based_sample_genome_stats["total_abundance"] +
                     abundance_based_sample_genome_stats["total_abundance"] * predicted_error_index +
                     abundance_based_sample_genome_stats["total_abundance"] * _genetic_var) / kmers_to_bases
                    if abundance_based_sample_genome_stats["total_abundance"] is not None else 0
                )
                genome_stats["corrected repfree genomic mean abundance"] = (
                    genome_stats["corrected repfree genomic total abundance"] / len(self.reference_without_repeats)
                )
                denominator = genome_stats["razan corrected genome repfree coverage index"] * len(self.reference_without_repeats)
                genome_stats["corrected repfree genomic mean abundance - no_zero_cov"] = np.divide(
                    genome_stats["corrected repfree genomic total abundance"],
                    denominator,
                    out=np.zeros_like(genome_stats["corrected repfree genomic total abundance"]),
                    where=denominator != 0
                )

        (genome_stats["genomic_dprime"],
         genome_stats["genomic_overlap"],
         genome_stats["genomic_threshold"]) = self._fit_two_component_gmm(sample_genome.abundances)
        (sample_stats["sample_dprime"],
         sample_stats["sample_overlap"],
         sample_stats["sample_threshold"]) = self._fit_two_component_gmm(sample_sig.abundances)

        # ============= AMPLICON STATS =============
        if self.amplicon_sig is not None:
            _amplicon_ref_sig = self.amplicon_sig & self.reference_sig
            self.logger.debug(
                "Amplicon signature is contained by the reference genome: %s with intersection of %d hashes.",
                (len(_amplicon_ref_sig) == len(self.amplicon_sig)), len(_amplicon_ref_sig)
            )
            if len(_amplicon_ref_sig) != len(self.amplicon_sig):
                _sig_to_be_removed = self.amplicon_sig - _amplicon_ref_sig
                _percentage_of_removal = (len(_sig_to_be_removed) / len(self.amplicon_sig) * 100)
                if _percentage_of_removal > 20:
                    self.logger.error("[!] More than 20%% of the amplicon signature is not contained in the reference genome.")
                    raise ValueError("Amplicon signature is poorly contained in the reference genome.")
                self.logger.debug(
                    "Amplicon signature is not fully contained in the reference genome.\n"
                    "Removing %d hashes (%.2f%%) from the amplicon signature.",
                    len(_sig_to_be_removed), _percentage_of_removal
                )
                self.amplicon_sig.difference_sigs(_sig_to_be_removed)
                self.logger.debug("Amplicon signature has been modified to be fully contained in the reference genome.")

            self.logger.debug("Calculating amplicon statistics.")
            sample_amplicon = sample_sig & self.amplicon_sig
            sample_amplicon_stats = sample_amplicon.get_sample_stats

            # NEW: Repetitive-free amplicon metrics
            if self.repetitive_aware_flag:
                sample_amplicon_non_repetitive = sample_amplicon.copy()
                sample_amplicon_non_repetitive &= self.reference_without_repeats
                self.amplicon_without_repeats = self.amplicon_sig & self.reference_without_repeats
                abundance_based_sample_amplicon_stats = sample_amplicon_non_repetitive.get_sample_stats

                ### --------- amplicon variance calculation heuristics -------------
                _projected_repfree_amplicon_abundance_array_length = len(self.amplicon_without_repeats)
                # Extend np array with zeros to match amplicon reference length
                pad_length = _projected_repfree_amplicon_abundance_array_length - len(sample_amplicon_non_repetitive.abundances)
                _custom_amplicon_repfree_abundances = np.concatenate([
                    sample_amplicon_non_repetitive.abundances,
                    np.zeros(pad_length)
                ])
                # Remove top 1% of the abundances by value
                _custom_amplicon_repfree_abundances = np.sort(_custom_amplicon_repfree_abundances)
                _custom_amplicon_repfree_abundances = _custom_amplicon_repfree_abundances[:int(len(_custom_amplicon_repfree_abundances) * 0.99)]

                amplicon_stats["Amplicon repfree k-mers total abundance"] = \
                    abundance_based_sample_amplicon_stats["total_abundance"]
                amplicon_stats["Amplicon repfree k-mers mean abundance"] = (
                    abundance_based_sample_amplicon_stats["total_abundance"] / len(self.amplicon_without_repeats)
                    if len(self.amplicon_without_repeats) > 0 and abundance_based_sample_amplicon_stats["total_abundance"] is not None else 0
                )
                amplicon_stats["Amplicon repfree k-mers variance abundance"] = np.var(_custom_amplicon_repfree_abundances)
                amplicon_stats["Amplicon repfree k-mers mean abundance - no_zero_cov"] = \
                    abundance_based_sample_amplicon_stats["mean_abundance"]
                amplicon_stats["Amplicon repfree k-mers median abundance - no_zero_cov"] = \
                    abundance_based_sample_amplicon_stats["median_abundance"]
                
                amplicon_stats["Amplicon repfree k-mers coverage index"] = (
                    abundance_based_sample_amplicon_stats["num_hashes"] / len(self.amplicon_without_repeats)
                    if len(self.amplicon_without_repeats) > 0 and abundance_based_sample_amplicon_stats["num_hashes"] is not None else 0
                )    
                
            else:
                abundance_based_sample_amplicon_stats = sample_amplicon_stats

            amplicon_stats.update({
                "Amplicon unique k-mers": sample_amplicon_stats["num_hashes"],
                "Amplicon k-mers total abundance": abundance_based_sample_amplicon_stats["total_abundance"],
                "Amplicon k-mers mean abundance": abundance_based_sample_amplicon_stats["mean_abundance"],
                "Amplicon k-mer variance abundance": np.var(sample_amplicon.abundances),
                "Amplicon k-mers median abundance": abundance_based_sample_amplicon_stats["median_abundance"],
                "Amplicon coverage index": (
                    sample_amplicon_stats["num_hashes"] / len(self.amplicon_sig)
                    if len(self.amplicon_sig) > 0 and sample_amplicon_stats["num_hashes"] is not None else 0
                ),
            })

            # NEW: multiple corrected coverage indices for amplicons (similar to genome’s tmp & razan coverage)
            if kmers_to_bases > 0:
                # tmp corrected coverage index
                amplicon_stats["tmp corrected amplicon coverage index"] = (
                    1 - (1 - amplicon_stats["Amplicon coverage index"]) ** 
                    ((1 + predicted_error_index + _genetic_var) / kmer_yield)
                    if amplicon_stats["Amplicon coverage index"] > 0 else 0
                )

                # razan corrected coverage index
                # (using 'normalized_fracminhash_precision' to parallel the genome logic)
                min_cov = amplicon_stats["Amplicon coverage index"] * normalized_fracminhash_precision
                amplicon_stats["razan corrected amplicon coverage index"] = (
                    1 - (1 - min(min_cov, 1.0)) ** ((1 + predicted_error_index + _genetic_var) / kmers_to_bases)
                    if amplicon_stats["Amplicon coverage index"] > 0 else 0
                )

                # corrected total abundance
                amplicon_stats["corrected amplicon total abundance"] = (
                    (abundance_based_sample_amplicon_stats["total_abundance"] +
                    abundance_based_sample_amplicon_stats["total_abundance"] * predicted_error_index +
                    abundance_based_sample_amplicon_stats["total_abundance"] * _genetic_var)
                    / kmers_to_bases
                    if abundance_based_sample_amplicon_stats["total_abundance"] is not None else 0
                )

                # corrected mean abundance (overall)
                amplicon_stats["corrected amplicon mean abundance"] = (
                    amplicon_stats["corrected amplicon total abundance"] / len(self.amplicon_sig)
                    if len(self.amplicon_sig) > 0 else 0
                )

                # corrected mean abundance - no_zero_cov
                amplicon_denominator = sample_amplicon_stats["num_hashes"]
                if amplicon_denominator == 0:
                    amplicon_stats["corrected amplicon mean abundance - no_zero_cov"] = 0
                else:
                    amplicon_stats["corrected amplicon mean abundance - no_zero_cov"] = \
                        amplicon_stats["corrected amplicon total abundance"] / amplicon_denominator

                # corrected mapping index for amplicons
                amplicon_stats["corrected amplicon mapping index"] = (
                    (sample_amplicon_stats["total_abundance"] + sample_amplicon_stats["total_abundance"] * predicted_error_index + sample_amplicon_stats["total_abundance"] * _genetic_var)
                    / sample_stats["k-mer total abundance"]
                    if (sample_stats.get("k-mer total abundance", 0) > 0
                        and sample_amplicon_stats["total_abundance"] is not None) else 0
                )
                
                if self.repetitive_aware_flag:
                    amplicon_stats["corrected repfree amplicon total abundance"] = (
                        (abundance_based_sample_amplicon_stats["total_abundance"] +
                        abundance_based_sample_amplicon_stats["total_abundance"] * predicted_error_index +
                        abundance_based_sample_amplicon_stats["total_abundance"] * _genetic_var)
                        / kmers_to_bases
                        if abundance_based_sample_amplicon_stats["total_abundance"] is not None else 0
                    )
                    amplicon_stats["corrected repfree amplicon mean abundance"] = (
                        amplicon_stats["corrected repfree amplicon total abundance"] / len(self.amplicon_without_repeats)
                        if len(self.amplicon_sig) > 0 else 0
                    )
                    amplicon_denominator = abundance_based_sample_amplicon_stats["num_hashes"]
                    if amplicon_denominator == 0:
                        amplicon_stats["corrected repfree amplicon mean abundance - no_zero_cov"] = 0
                    else:
                        amplicon_stats["corrected repfree amplicon mean abundance - no_zero_cov"] = \
                            amplicon_stats["corrected repfree amplicon total abundance"] / amplicon_denominator


                # NEW: saturation model (“grok”) for amplicons
                # raw
                lambda_amplicon_raw = amplicon_stats["Amplicon k-mers mean abundance"]
                cov_amplicon_raw = amplicon_stats["Amplicon coverage index"]
                amplicon_len = len(self.amplicon_sig)

                lambda_amplicon_raw_stats = self._grok_calcs(
                    lambda_amplicon_raw,
                    cov_amplicon_raw,
                    "raw_amplicon_",
                    amplicon_len
                )
                amplicon_stats.update(lambda_amplicon_raw_stats)

                # corrected
                corrected_amplicon_lambda = amplicon_stats["corrected amplicon mean abundance"]
                corrected_amplicon_cov_index = amplicon_stats["razan corrected amplicon coverage index"]
                lambda_amplicon_stats_corr = self._grok_calcs(
                    corrected_amplicon_lambda,
                    corrected_amplicon_cov_index,
                    "corrected_amplicon_",
                    amplicon_len
                )
                amplicon_stats.update(lambda_amplicon_stats_corr)

            # Maintain the old relative stats as-is
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
        self.logger.debug("Calculating error and contamination indices.")
        sample_nonref_non_singletons = sample_nonref.total_abundance - sample_nonref_singletons


        predicted_contamination_index = (
            sample_nonref_non_singletons / sample_total_abundance
            if sample_total_abundance is not None and sample_total_abundance > 0 else 0
        )
        predicted_error_contamination_index["Predicted contamination index"] = predicted_contamination_index
        predicted_error_contamination_index["Sequencing errors index"] = predicted_error_index

        # ! DEV NEW TEST PARAM
        _sample_ref = sample_sig & self.reference_sig
        _sample_ref_singletons = _sample_ref.count_singletons()
        _tmp_x = (
            _sample_ref_singletons / _sample_ref.total_abundance
            if _sample_ref.total_abundance is not None and _sample_ref.total_abundance > 0 else 0
        )
        _tmp_y = (
            sample_nonref_singletons / sample_nonref.total_abundance
            if sample_nonref.total_abundance is not None and sample_nonref.total_abundance > 0 else 0
        )
        _tmp_z_error_rate = _tmp_y - _tmp_x
        predicted_error_contamination_index["z_error_rate"] = _tmp_z_error_rate * 100
        _z_error_rate_raw_y = (sample_sig.count_singletons() / sample_sig.total_abundance) - _tmp_x
        predicted_error_contamination_index["z_error_rate_raw_y"] = _z_error_rate_raw_y * 100
        predicted_error_contamination_index["genomic_singletons"] = _sample_ref_singletons

        # ============= Advanced Stats if needed =============
        if advanced:
            self.logger.debug("Calculating advanced statistics.")
            median_trimmed_sample_sig = sample_sig.copy()
            median_trimmed_sample_sig.trim_below_median()
            median_trimmed_sample_stats = median_trimmed_sample_sig.get_sample_stats
            advanced_stats.update({
                "Median-trimmed unique k-mers": median_trimmed_sample_stats["num_hashes"],
                "Median-trimmed total abundance": median_trimmed_sample_stats["total_abundance"],
                "Median-trimmed mean abundance": median_trimmed_sample_stats["mean_abundance"],
                "Median-trimmed median abundance": median_trimmed_sample_stats["median_abundance"],
            })
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
                advanced_stats["Median-trimmed relative coverage"] = (
                    advanced_stats["Median-trimmed Amplicon coverage index"]
                    / advanced_stats["Median-trimmed Genome coverage index"]
                    if advanced_stats.get("Median-trimmed Genome coverage index", 0) > 0 and advanced_stats.get("Median-trimmed Amplicon coverage index") is not None else 0
                )
                advanced_stats["Median-trimmed relative mean abundance"] = (
                    advanced_stats["Median-trimmed Amplicon mean abundance"]
                    / advanced_stats["Median-trimmed Genomic mean abundance"]
                    if advanced_stats.get("Median-trimmed Genomic mean abundance", 0) > 0 and advanced_stats.get("Median-trimmed Amplicon mean abundance") is not None else 0
                )
        # ============= CHROMOSOME STATS =============
        def sort_chromosomes(chrom_dict):
            sorted_keys = sorted(
                chrom_dict,
                key=lambda x: (int(x.split("-")[1]) if x.split("-")[1].isdigit() else float("inf"), x),
            )
            return {k: chrom_dict[k] for k in sorted_keys}

        if self.specific_chr_to_sig:
            self.logger.debug("Calculating mean abundance for each chromosome.")
            for chr_name, chr_sig in self.specific_chr_to_sig.items():
                chr_sample_sig = sample_sig & chr_sig
                chr_to_mean_abundance[chr_name] = chr_sample_sig.total_abundance / len(chr_sig)
            sorted_chr_to_mean_abundance = {
                f"chr-{chr_name.replace('sex-', '').replace('autosome-', '')}": mean_abundance
                for chr_name, mean_abundance in sort_chromosomes(chr_to_mean_abundance).items()
                if "mitochondrial" not in chr_name.lower()
            }
            chrs_stats.update(sorted_chr_to_mean_abundance)

            autosomal_chr_to_mean_abundance = {}
            for chr_name, mean_abundance in chr_to_mean_abundance.items():
                if "sex" in chr_name.lower() or "-snipegenome" in chr_name.lower() or "mitochondrial" in chr_name.lower():
                    self.logger.debug("Skipping %s from autosomal_chr_to_mean_abundance.", chr_name)
                    continue
                autosomal_chr_to_mean_abundance[chr_name] = mean_abundance

            if autosomal_chr_to_mean_abundance:
                mean_abundances = np.array(list(autosomal_chr_to_mean_abundance.values()), dtype=np.float64)
                mean = np.mean(mean_abundances)
                cv = np.std(mean_abundances) / mean if mean > 0 and not np.isnan(mean) else 0.0
                chrs_stats.update({"Autosomal k-mer mean abundance CV": cv})
                self.logger.debug("Calculated Autosomal CV: %f", cv)
            else:
                self.logger.warning("No autosomal chromosomes were processed. 'Autosomal_CV' set to None.")
                chrs_stats.update({"Autosomal k-mer mean abundance CV": None})

            if "sex-x" in self.specific_chr_to_sig:
                self.logger.debug("Calculating sex stats (chrX and chrY).")
                autosomals_genome_sig = self.reference_sig.copy()
                for chr_name, chr_sig in self.specific_chr_to_sig.items():
                    if "sex" in chr_name.lower() or "mitochondrial" in chr_name.lower():
                        self.logger.debug("Removing %s from autosomal genome signature.", chr_name)
                        autosomals_genome_sig -= chr_sig
                specific_xchr_sig = self.specific_chr_to_sig["sex-x"] - autosomals_genome_sig
                self.logger.debug("\t-Derived X chromosome-specific signature size: %d hashes.", len(specific_xchr_sig))
                sample_specific_xchr_sig = sample_sig & self.specific_chr_to_sig["sex-x"]
                self.logger.debug("\t-Intersected sample signature with X-specific k-mers = %d hashes.", len(sample_specific_xchr_sig))
                sample_autosomal_sig = sample_sig & autosomals_genome_sig
                self.logger.debug("\t-Intersected sample signature with autosomal genome k-mers = %d hashes.", len(sample_autosomal_sig))
                xchr_mean_abundance = (sample_specific_xchr_sig.total_abundance / len(self.specific_chr_to_sig["sex-x"])
                                       if len(sample_specific_xchr_sig) > 0 else 0.0)
                autosomal_mean_abundance = (np.mean(list(autosomal_chr_to_mean_abundance.values()))
                                            if len(sample_autosomal_sig) > 0 else 0.0)
                if autosomal_mean_abundance == 0:
                    self.logger.warning("Autosomal mean abundance is zero. Setting chrX Ploidy score to zero.")
                    xploidy_score = 0.0
                else:
                    xploidy_score = (xchr_mean_abundance / autosomal_mean_abundance
                                     if len(specific_xchr_sig) > 0 and autosomal_mean_abundance > 0 else 0.0)
                self.logger.debug("Calculated chrX Ploidy score: %.4f", xploidy_score)
                sex_stats.update({"chrX Ploidy score": xploidy_score})
            else:
                self.logger.debug("No X chromosome-specific signature detected. chrX Ploidy score will be set to zero.")

            if "sex-y" in self.specific_chr_to_sig and "sex-x" in self.specific_chr_to_sig:
                self.logger.debug("Calculating chrY Coverage score based on Y chromosome-specific k-mers.")
                ychr_specific_kmers = self.specific_chr_to_sig["sex-y"] - autosomals_genome_sig - (self.specific_chr_to_sig["sex-x"] - autosomals_genome_sig)
                self.logger.debug("\t-Derived Y chromosome-specific signature size: %d hashes.", len(ychr_specific_kmers))
                ychr_in_sample = sample_sig & ychr_specific_kmers
                self.logger.debug("\t-Intersected sample signature with Y chromosome-specific k-mers = %d hashes.", len(ychr_in_sample))
                autosomals_specific_kmers = self.reference_sig - self.specific_chr_to_sig["sex-x"] - self.specific_chr_to_sig["sex-y"]
                if len(ychr_specific_kmers) == 0 or len(autosomals_specific_kmers) == 0:
                    self.logger.warning("Insufficient k-mers for chrY Coverage score calculation. Setting chrY Coverage score to zero.")
                    ycoverage = 0.0
                else:
                    try:
                        ycoverage = ((len(ychr_in_sample) / len(ychr_specific_kmers)) /
                                     (len(sample_autosomal_sig) / len(autosomals_specific_kmers)))
                    except (ZeroDivisionError, TypeError):
                        ycoverage = 0.0
                self.logger.debug("Calculated chrY Coverage score: %.4f", ycoverage)
                sex_stats.update({"chrY Coverage score": ycoverage})
            else:
                self.logger.warning("No Y chromosome-specific signature detected. chrY Coverage score will be set to zero.")

        # ============= VARIANCE NONREF STATS =============
        if self.variance_sigs:
            self.logger.debug("Consuming non-reference k-mers from provided variables.")
            self.logger.debug("\tSize of non-reference k-mers in the sample signature: %d hashes.", len(sample_nonref))
            sample_nonref_total_abundance = sample_nonref.total_abundance
            if len(sample_nonref) == 0:
                self.logger.warning("No non-reference k-mers found in the sample signature.")
            else:
                for variance_sig in self.variance_sigs:
                    variance_name = variance_sig.name
                    sample_nonref_var = sample_nonref & variance_sig
                    if self.export_varsigs:
                        __sample_name = sample_sig.name.replace(" ", "_")
                        __var_name = variance_name.replace(" ", "_")
                        __filename = __sample_name + "_" + __var_name + "_nonref.zip"
                        self.logger.debug("Exporting non-reference k-mers from variable '%s'.", __filename)
                        sample_nonref_var.export(__filename)
                    var_total = sample_nonref_var.total_abundance
                    sample_nonref_var_fraction_total = (var_total / sample_nonref_total_abundance
                                                        if sample_nonref_total_abundance > 0 and var_total is not None else 0.0)
                    vars_nonref_stats.update({
                        f"{variance_name} total k-mer abundance": var_total,
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
                    sample_nonref.total_abundance / sample_nonref_total_abundance if sample_nonref_total_abundance > 0 and sample_nonref.total_abundance is not None else 0.0
                )
                self.logger.debug("After consuming all vars from the non reference k-mers, the size of the sample signature is: %d hashes, with total abundance of %s.",
                                  len(sample_nonref), sample_nonref.total_abundance)

        # ============= Coverage Prediction (ROI) =============
        if (predict_extra_folds and 
            genome_stats.get("Genome coverage index") is not None and 
            genome_stats["Genome coverage index"] > 0.01):
            roi_stats = self._predict_roi_stats(sample_sig, kmers_to_bases, predict_extra_folds)
        else:
            self.logger.warning("Skipping ROI prediction due to zero Genomic Coverage Index.")

        # ============= Merging all stats into one dictionary =============
        aggregated_stats = {}
        for d in [sample_stats, genome_stats, amplicon_stats, advanced_stats, chrs_stats, sex_stats, predicted_error_contamination_index, vars_nonref_stats, roi_stats]:
            aggregated_stats.update(d)
        self.sample_to_stats[sample_sig.name] = aggregated_stats
        return aggregated_stats
