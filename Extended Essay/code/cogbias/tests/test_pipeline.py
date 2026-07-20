"""test_pipeline.py — Unit and integration tests for pipeline.py.

Run from the cogbias/ directory:
    pytest tests/test_pipeline.py -v

Tests are designed to run on the synthetic dataset so no real Supabase
connection is needed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BIAS_TYPE, CV_K_FALLBACK, FEATURE_COLS, RT_TOO_FAST_MS, RT_TOO_SLOW_MS,
    SESSION_MIN_SECS, TARGET_COL,
)
from pipeline import (
    cv_evaluate, fit_logreg, fit_mlp, flag_quality, load_and_prepare,
    run_full, wilson_ci,
)

_SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_data" / "synthetic_responses.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    df = pd.read_csv(_SAMPLE_CSV)
    assert not df.empty, "synthetic_responses.csv is missing or empty"
    return df


@pytest.fixture(scope="module")
def flagged_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return flag_quality(raw_df)


@pytest.fixture(scope="module")
def prepared(raw_df: pd.DataFrame):
    return load_and_prepare(raw_df)


@pytest.fixture(scope="module")
def full_results(raw_df: pd.DataFrame):
    return run_full(raw_df, exclude_flagged=False)


# ─────────────────────────────────────────────────────────────────────────────
# flag_quality
# ─────────────────────────────────────────────────────────────────────────────

class TestFlagQuality:
    def test_returns_copy_not_inplace(self, raw_df):
        out = flag_quality(raw_df)
        assert "any_flag" not in raw_df.columns
        assert "any_flag" in out.columns

    def test_flag_columns_exist(self, flagged_df):
        for col in ["rt_too_fast", "rt_too_slow", "session_too_fast", "attention_failed", "any_flag"]:
            assert col in flagged_df.columns, f"Missing flag column: {col}"

    def test_rt_too_fast_correct(self, raw_df, flagged_df):
        expected = raw_df["reaction_time_ms"] < RT_TOO_FAST_MS
        pd.testing.assert_series_equal(
            flagged_df["rt_too_fast"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_rt_too_slow_correct(self, raw_df, flagged_df):
        expected = raw_df["reaction_time_ms"] > RT_TOO_SLOW_MS
        pd.testing.assert_series_equal(
            flagged_df["rt_too_slow"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_speedrunners_flagged(self, flagged_df):
        # Synthetic data plants 1-in-20 speedrunners (all RT < RT_TOO_FAST_MS)
        n_flagged_pids = flagged_df.loc[flagged_df["rt_too_fast"], "participant_id"].nunique()
        assert n_flagged_pids >= 2, f"Expected >=2 speedrunners flagged, got {n_flagged_pids}"

    def test_attention_failed_propagates_to_all_participant_rows(self, flagged_df):
        failed_pids = set(
            flagged_df.loc[flagged_df["attention_check_passed"] == False, "participant_id"]
        )
        if not failed_pids:
            pytest.skip("No attention-check failures in synthetic data")
        for pid in failed_pids:
            rows = flagged_df[flagged_df["participant_id"] == pid]
            assert rows["attention_failed"].all(), (
                f"Participant {pid} has attention failure not propagated to all rows"
            )

    def test_any_flag_is_union(self, flagged_df):
        expected = (
            flagged_df["rt_too_fast"]
            | flagged_df["rt_too_slow"]
            | flagged_df["session_too_fast"]
            | flagged_df["attention_failed"]
        )
        pd.testing.assert_series_equal(
            flagged_df["any_flag"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_all_flag_cols_are_bool(self, flagged_df):
        for col in ["rt_too_fast", "rt_too_slow", "attention_failed", "any_flag"]:
            assert flagged_df[col].dtype == bool, f"{col} is not bool"


# ─────────────────────────────────────────────────────────────────────────────
# load_and_prepare
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAndPrepare:
    def test_returns_four_tuple(self, prepared):
        assert len(prepared) == 4

    def test_no_attention_check_rows_in_modeling_df(self, prepared):
        modeling_df, y, groups, meta = prepared
        if "attention_check_passed" in modeling_df.columns:
            assert modeling_df["attention_check_passed"].isna().all(), (
                "Attention-check rows leaked into modeling_df"
            )

    def test_meta_contains_attention_check_rows(self, raw_df, prepared):
        modeling_df, y, groups, meta = prepared
        if "attention_check_passed" in raw_df.columns:
            n_attn = raw_df["attention_check_passed"].notna().sum()
            assert len(meta) == n_attn, (
                f"Expected {n_attn} attention-check rows in meta, got {len(meta)}"
            )

    def test_y_is_binary(self, prepared):
        _, y, _, _ = prepared
        assert set(np.unique(y)).issubset({0, 1}), "y contains values outside {0, 1}"

    def test_groups_length_matches_df(self, prepared):
        modeling_df, y, groups, meta = prepared
        assert len(groups) == len(modeling_df)
        assert len(y) == len(modeling_df)

    def test_log1p_applied_to_rt(self, prepared):
        modeling_df, _, _, _ = prepared
        rt = modeling_df["reaction_time_ms"]
        assert rt.min() > 0, "log1p(x) > 0 for x > 0; got non-positive value"
        assert rt.max() < 20, "RT values look raw (not log-transformed)"

    def test_only_completed_true_rows(self, raw_df, prepared):
        modeling_df, _, _, _ = prepared
        if "completed" in raw_df.columns:
            n_complete = (raw_df["completed"] == True).sum()
            # After attention-check rows removed, count should be <= n_complete
            assert len(modeling_df) <= n_complete

    def test_missing_column_raises(self, raw_df):
        bad = raw_df.drop(columns=["choice"])
        with pytest.raises(ValueError, match="Missing required columns"):
            load_and_prepare(bad)

    def test_all_feature_cols_present(self, prepared):
        modeling_df, _, _, _ = prepared
        for col in FEATURE_COLS:
            assert col in modeling_df.columns, f"Feature column {col} missing from modeling_df"


# ─────────────────────────────────────────────────────────────────────────────
# wilson_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestWilsonCI:
    def test_zero_n_returns_unit_interval(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_bounds_in_unit_interval(self):
        for n in [10, 50, 100, 500]:
            for s in [0, n // 4, n // 2, n]:
                lo, hi = wilson_ci(s, n)
                assert 0.0 <= lo <= hi <= 1.0, f"Bad CI for s={s}, n={n}: [{lo}, {hi}]"

    def test_all_successes_upper_bound_one(self):
        _, hi = wilson_ci(100, 100)
        assert math.isclose(hi, 1.0, abs_tol=1e-9)

    def test_no_successes_lower_bound_zero(self):
        lo, _ = wilson_ci(0, 100)
        assert lo == 0.0

    def test_half_successes_symmetric_around_0p5(self):
        lo, hi = wilson_ci(50, 100)
        assert abs((lo + hi) / 2 - 0.5) < 0.01

    def test_known_value(self):
        # n=100, 60 successes: Wilson CI approx [0.500, 0.693]
        lo, hi = wilson_ci(60, 100)
        assert 0.49 < lo < 0.52
        assert 0.68 < hi < 0.71

    def test_ci_narrows_with_larger_n(self):
        lo1, hi1 = wilson_ci(50, 100)
        lo2, hi2 = wilson_ci(500, 1000)
        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width2 < width1


# ─────────────────────────────────────────────────────────────────────────────
# cv_evaluate
# ─────────────────────────────────────────────────────────────────────────────

class TestCvEvaluate:
    def _small_df(self):
        """Create a minimal dataset (6 participants x 10 trials) for unit tests."""
        rng = np.random.default_rng(0)
        n_participants, n_trials = 6, 10
        rows = []
        for p in range(n_participants):
            for t in range(n_trials):
                rows.append({
                    "trial_index": t + 1,
                    "bias_type": rng.integers(1, 4),
                    "frame": rng.integers(0, 2),
                    "expected_value_ratio": rng.uniform(0.85, 1.15),
                    "anchor_value": 0.0,
                    "previous_choice": rng.integers(0, 2),
                    "is_first_trial": int(t == 0),
                    "reaction_time_ms": float(rng.integers(300, 5000)),
                    TARGET_COL: rng.integers(0, 2),
                    "participant_id": f"P{p:03d}",
                })
        return pd.DataFrame(rows)

    def test_returns_required_keys(self):
        df = self._small_df()
        df["reaction_time_ms"] = np.log1p(df["reaction_time_ms"])
        y = df[TARGET_COL].to_numpy(dtype=int)
        groups = df["participant_id"].to_numpy()
        result = cv_evaluate(df, y, groups, fit_logreg, k=CV_K_FALLBACK)
        for key in ["accuracy", "precision", "recall", "f1", "wilson_ci",
                    "confusion_matrix", "by_bias_type", "baselines"]:
            assert key in result, f"Missing key: {key}"

    def test_accuracy_is_valid_proportion(self):
        df = self._small_df()
        df["reaction_time_ms"] = np.log1p(df["reaction_time_ms"])
        y = df[TARGET_COL].to_numpy(dtype=int)
        groups = df["participant_id"].to_numpy()
        for fit_fn in [fit_logreg, fit_mlp]:
            result = cv_evaluate(df, y, groups, fit_fn, k=CV_K_FALLBACK)
            acc = result["accuracy"]["mean"]
            assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of [0, 1]"

    def test_confusion_matrix_shape(self):
        df = self._small_df()
        df["reaction_time_ms"] = np.log1p(df["reaction_time_ms"])
        y = df[TARGET_COL].to_numpy(dtype=int)
        groups = df["participant_id"].to_numpy()
        result = cv_evaluate(df, y, groups, fit_logreg, k=CV_K_FALLBACK)
        assert result["confusion_matrix"].shape == (2, 2)

    def test_baselines_have_chance_of_0p5(self):
        df = self._small_df()
        df["reaction_time_ms"] = np.log1p(df["reaction_time_ms"])
        y = df[TARGET_COL].to_numpy(dtype=int)
        groups = df["participant_id"].to_numpy()
        result = cv_evaluate(df, y, groups, fit_logreg, k=CV_K_FALLBACK)
        assert result["baselines"]["chance"] == 0.50

    def test_logreg_has_feature_names(self):
        df = self._small_df()
        df["reaction_time_ms"] = np.log1p(df["reaction_time_ms"])
        y = df[TARGET_COL].to_numpy(dtype=int)
        groups = df["participant_id"].to_numpy()
        result = cv_evaluate(df, y, groups, fit_logreg, k=CV_K_FALLBACK)
        assert "coeff_mean" in result
        assert "feature_names" in result
        assert len(result["feature_names"]) == len(result["coeff_mean"])

    def test_mlp_has_no_coeff(self):
        df = self._small_df()
        df["reaction_time_ms"] = np.log1p(df["reaction_time_ms"])
        y = df[TARGET_COL].to_numpy(dtype=int)
        groups = df["participant_id"].to_numpy()
        result = cv_evaluate(df, y, groups, fit_mlp, k=CV_K_FALLBACK)
        assert "coeff_mean" not in result


# ─────────────────────────────────────────────────────────────────────────────
# run_full (integration test against synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunFull:
    def test_returns_expected_keys(self, full_results):
        for key in ["lr", "nn", "k", "n_participants", "n_samples", "n_flagged",
                    "feature_names", "exclude_flagged",
                    "fig_cm_lr", "fig_cm_nn", "fig_bias_bar", "fig_lr_coeff"]:
            assert key in full_results, f"Missing key: {key}"

    def test_n_participants_is_50(self, full_results):
        assert full_results["n_participants"] == 50

    def test_n_flagged_nonzero(self, full_results):
        assert full_results["n_flagged"] >= 2, (
            f"Expected >=2 flagged participants (planted), got {full_results['n_flagged']}"
        )

    def test_lr_accuracy_is_valid(self, full_results):
        acc = full_results["lr"]["accuracy"]["mean"]
        assert acc > 0.55, f"LR accuracy {acc:.3f} should beat chance on synthetic data"

    def test_nn_accuracy_is_valid(self, full_results):
        acc = full_results["nn"]["accuracy"]["mean"]
        assert acc > 0.55, f"NN accuracy {acc:.3f} should beat chance on synthetic data"

    def test_wilson_ci_is_valid(self, full_results):
        for model in ["lr", "nn"]:
            lo, hi = full_results[model]["wilson_ci"]
            assert 0.0 <= lo <= hi <= 1.0, f"{model} Wilson CI invalid: [{lo}, {hi}]"

    def test_confusion_matrix_sums_to_n_samples(self, full_results):
        ns = full_results["n_samples"]
        for model in ["lr", "nn"]:
            cm_sum = full_results[model]["confusion_matrix"].sum()
            # Each sample appears in exactly one test fold across all folds
            assert abs(cm_sum - ns) / ns < 0.05, (
                f"{model} CM sum {cm_sum} inconsistent with n_samples={ns}"
            )

    def test_by_bias_type_has_three_entries(self, full_results):
        for model in ["lr", "nn"]:
            by_bias = full_results[model]["by_bias_type"]
            assert len(by_bias) == 3, f"{model} by_bias_type has {len(by_bias)} entries"

    def test_exclude_flagged_reduces_participants(self, raw_df):
        # Synthetic data sets all created_at to the same timestamp (script runs instantly),
        # so session_too_fast=True for every participant. To test the exclude_flagged path,
        # we supply a version of the data with realistic elapsed timestamps so that most
        # participants have session_duration >= SESSION_MIN_SECS and are not flagged.
        from datetime import datetime, timezone, timedelta
        df = raw_df.copy()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        pids = df["participant_id"].unique()
        ts_col = []
        pid_next = {pid: 0 for pid in pids}
        for pid in df["participant_id"]:
            idx = pid_next[pid]
            # 30s per trial → ~15-minute session, well above SESSION_MIN_SECS=90s
            ts_col.append((t0 + timedelta(seconds=30 * idx)).isoformat())
            pid_next[pid] += 1
        df["created_at"] = ts_col

        all_results  = run_full(df, exclude_flagged=False)
        excl_results = run_full(df, exclude_flagged=True)
        assert excl_results["n_participants"] < all_results["n_participants"], (
            "Excluding flagged participants should reduce n_participants"
        )

    def test_figures_are_matplotlib_figures(self, full_results):
        import matplotlib.pyplot as plt
        for key in ["fig_cm_lr", "fig_cm_nn", "fig_bias_bar", "fig_lr_coeff"]:
            fig = full_results[key]
            assert isinstance(fig, plt.Figure), f"{key} is not a matplotlib Figure"

    def test_feature_names_match_feature_cols(self, full_results):
        names = full_results["feature_names"]
        assert len(names) > len(FEATURE_COLS), (
            "OneHotEncoder should expand feature count beyond raw FEATURE_COLS"
        )
        assert any("bias_type" in n for n in names), (
            "Expected one-hot encoded bias_type features in feature_names"
        )
