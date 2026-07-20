"""test_scenarios.py — Unit tests for the parametric scenario generator.

Run from the cogbias/ directory:
    pytest tests/test_scenarios.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from scenarios import generate_participant_scenarios
from config import (
    BIAS_TYPE, FRAME, N_SCENARIOS_PER_BIAS, N_SCENARIOS_TOTAL,
    EV_RATIO_RANGE, N_FRAMING_GAIN, N_FRAMING_LOSS,
)


@pytest.fixture
def scenarios_seed42():
    return generate_participant_scenarios(seed=42)


@pytest.fixture
def scenarios_seed99():
    return generate_participant_scenarios(seed=99)


class TestDeterminism:
    def test_same_seed_identical(self):
        s1 = generate_participant_scenarios(seed=7)
        s2 = generate_participant_scenarios(seed=7)
        assert s1 == s2

    def test_different_seeds_differ(self, scenarios_seed42, scenarios_seed99):
        params_42 = [(s["expected_value_ratio"], s["anchor_value"]) for s in scenarios_seed42]
        params_99 = [(s["expected_value_ratio"], s["anchor_value"]) for s in scenarios_seed99]
        assert params_42 != params_99


class TestCounts:
    def test_exactly_30_scenarios(self, scenarios_seed42):
        assert len(scenarios_seed42) == N_SCENARIOS_TOTAL

    def test_exactly_10_per_bias_type(self, scenarios_seed42):
        for bt_name, bt_val in BIAS_TYPE.items():
            count = sum(1 for s in scenarios_seed42 if s["bias_type"] == bt_val)
            assert count == N_SCENARIOS_PER_BIAS, (
                f"Expected {N_SCENARIOS_PER_BIAS} {bt_name}, got {count}"
            )

    def test_counts_hold_across_many_seeds(self):
        for seed in [0, 1, 100, 999, 2**31 - 1]:
            scenarios = generate_participant_scenarios(seed)
            for bt_name, bt_val in BIAS_TYPE.items():
                count = sum(1 for s in scenarios if s["bias_type"] == bt_val)
                assert count == N_SCENARIOS_PER_BIAS, (
                    f"seed={seed}: {bt_name} count={count}"
                )


class TestFramingCounterbalance:
    def test_exactly_5_gain_5_loss(self, scenarios_seed42):
        framing = [s for s in scenarios_seed42 if s["bias_type"] == BIAS_TYPE["framing"]]
        gain = sum(1 for s in framing if s["frame"] == FRAME["gain"])
        loss = sum(1 for s in framing if s["frame"] == FRAME["loss"])
        assert gain == N_FRAMING_GAIN
        assert loss == N_FRAMING_LOSS

    def test_counterbalance_holds_across_50_seeds(self):
        for seed in range(50):
            scenarios = generate_participant_scenarios(seed)
            framing = [s for s in scenarios if s["bias_type"] == BIAS_TYPE["framing"]]
            gain = sum(1 for s in framing if s["frame"] == FRAME["gain"])
            loss = sum(1 for s in framing if s["frame"] == FRAME["loss"])
            assert gain == 5 and loss == 5, (
                f"seed={seed}: gain={gain}, loss={loss}"
            )


class TestAnchorValues:
    def test_anchor_nonzero_only_for_anchoring(self, scenarios_seed42):
        for s in scenarios_seed42:
            if s["bias_type"] == BIAS_TYPE["anchoring"]:
                assert s["anchor_value"] > 0
            else:
                assert s["anchor_value"] == 0.0

    def test_anchor_within_template_range(self, scenarios_seed42):
        from scenarios import _ANCHORING_TEMPLATES
        template_map = {t["template_id"]: t for t in _ANCHORING_TEMPLATES}
        for s in scenarios_seed42:
            if s["bias_type"] != BIAS_TYPE["anchoring"]:
                continue
            tpl = template_map[s["scenario_template_id"]]
            assert tpl["anchor_min"] <= s["anchor_value"] <= tpl["anchor_max"]


class TestEVRatio:
    def test_ev_ratio_positive_for_all(self, scenarios_seed42):
        for s in scenarios_seed42:
            assert s["expected_value_ratio"] > 0

    def test_ev_ratio_in_range_loss_aversion(self, scenarios_seed42):
        lo, hi = EV_RATIO_RANGE
        for s in scenarios_seed42:
            if s["bias_type"] == BIAS_TYPE["loss_aversion"]:
                ev = s["expected_value_ratio"]
                # ab_swap may invert, so either ev or 1/ev is in [lo, hi]
                assert (lo - 0.05 <= ev <= hi + 0.5) or (lo - 0.05 <= 1/ev <= hi + 0.5)


class TestTrialIndex:
    def test_trial_index_covers_1_to_30(self, scenarios_seed42):
        indices = sorted(s["trial_index"] for s in scenarios_seed42)
        assert indices == list(range(1, 31))

    def test_is_first_trial_exactly_once(self, scenarios_seed42):
        first_count = sum(1 for s in scenarios_seed42 if s["is_first_trial"] == 1)
        assert first_count == 1

    def test_is_first_trial_at_index_1(self, scenarios_seed42):
        first = next(s for s in scenarios_seed42 if s["trial_index"] == 1)
        assert first["is_first_trial"] == 1

    def test_all_30_templates_appear(self, scenarios_seed42):
        template_ids = {s["scenario_template_id"] for s in scenarios_seed42}
        expected = (
            {f"F{i:02d}" for i in range(1, 11)}
            | {f"A{i:02d}" for i in range(1, 11)}
            | {f"L{i:02d}" for i in range(1, 11)}
        )
        assert template_ids == expected, f"Missing: {expected - template_ids}"

    def test_order_differs_across_seeds(self, scenarios_seed42, scenarios_seed99):
        order_42 = [s["scenario_template_id"] for s in scenarios_seed42]
        order_99 = [s["scenario_template_id"] for s in scenarios_seed99]
        assert order_42 != order_99


class TestRequiredFields:
    REQUIRED = {
        "trial_index", "bias_type", "frame", "expected_value_ratio", "anchor_value",
        "previous_choice", "is_first_trial", "reaction_time_ms", "choice",
        "problem_text", "option_a_text", "option_b_text",
        "scenario_template_id", "canonical_biased_choice",
    }

    def test_all_required_fields_present(self, scenarios_seed42):
        for s in scenarios_seed42:
            missing = self.REQUIRED - set(s.keys())
            assert not missing, f"{s.get('scenario_template_id')} missing: {missing}"

    def test_canonical_biased_choice_is_0_or_1(self, scenarios_seed42):
        for s in scenarios_seed42:
            assert s["canonical_biased_choice"] in (0, 1)

    def test_choice_placeholder_is_none(self, scenarios_seed42):
        for s in scenarios_seed42:
            assert s["choice"] is None
