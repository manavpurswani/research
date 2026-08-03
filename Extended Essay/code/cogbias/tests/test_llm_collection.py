"""tests/test_llm_collection.py — Unit tests for collect_llm_data.py"""

from __future__ import annotations

import csv
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import collect_llm_data as cld
from scenarios import BIAS_TYPE, generate_participant_scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────────────────────────────────

def _scenario_of(bias_type_val: int) -> dict:
    for seed in range(1000, 1100):
        for sc in generate_participant_scenarios(seed):
            if sc["bias_type"] == bias_type_val:
                return sc
    raise RuntimeError(f"No scenario with bias_type={bias_type_val} in seeds 1000-1099")


class TestPromptRendering:
    def test_framing_prompt(self):
        sc = _scenario_of(BIAS_TYPE["framing"])
        p = cld.build_prompt(sc)
        assert f"Scenario {sc['trial_index']} of 30:" in p
        assert f"Option A: {sc['option_a_text']}" in p
        assert f"Option B: {sc['option_b_text']}" in p
        assert "Your choice (A or B):" in p

    def test_anchoring_prompt(self):
        sc = _scenario_of(BIAS_TYPE["anchoring"])
        p = cld.build_prompt(sc)
        assert "Option A:" in p
        assert "Option B:" in p
        assert "Your choice (A or B):" in p

    def test_loss_aversion_prompt(self):
        sc = _scenario_of(BIAS_TYPE["loss_aversion"])
        p = cld.build_prompt(sc)
        assert "Option A:" in p
        assert "Option B:" in p
        assert "Your choice (A or B):" in p

    def test_problem_text_in_prompt(self):
        sc = _scenario_of(BIAS_TYPE["framing"])
        p = cld.build_prompt(sc)
        assert sc["problem_text"] in p


# ─────────────────────────────────────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseParser:
    @pytest.mark.parametrize("raw,expected_choice", [
        ("A", 0),
        ("B", 1),
        ("a", 0),
        ("b", 1),
        (" A ", 0),
        (" B ", 1),
        ("Option A", 0),
        ("Option B", 1),
    ])
    def test_valid(self, raw, expected_choice):
        choice, invalid = cld.parse_response(raw)
        assert choice == expected_choice
        assert invalid is False

    @pytest.mark.parametrize("raw", ["", "maybe", "I don't know", "neither", "C"])
    def test_invalid(self, raw):
        choice, invalid = cld.parse_response(raw)
        assert choice is None
        assert invalid is True


# ─────────────────────────────────────────────────────────────────────────────
# Latency measurement uses time.monotonic()
# ─────────────────────────────────────────────────────────────────────────────

class TestLatency:
    def test_stub_returns_positive_int(self):
        _, latency_ms = cld._call_stub("any", "any", 0.5)
        assert isinstance(latency_ms, int)
        assert latency_ms > 0

    def test_openai_caller_uses_monotonic(self):
        assert "time.monotonic()" in inspect.getsource(cld._call_openai)

    def test_anthropic_caller_uses_monotonic(self):
        assert "time.monotonic()" in inspect.getsource(cld._call_anthropic)

    def test_google_caller_uses_monotonic(self):
        assert "time.monotonic()" in inspect.getsource(cld._call_google)

    def test_together_caller_uses_monotonic(self):
        assert "time.monotonic()" in inspect.getsource(cld._call_together)


# ─────────────────────────────────────────────────────────────────────────────
# CSV schema
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_REQUIRED = {
    "participant_id", "reaction_time_ms", "bias_type", "frame",
    "expected_value_ratio", "anchor_value", "previous_choice",
    "is_first_trial", "choice",
}


class TestCSVSchema:
    def _run_collect(self, tmp_path: Path, n_models: int = 1) -> list[dict]:
        out = tmp_path / "out.csv"
        models = [("openai", "gpt-4o")] * n_models
        with patch.object(cld, "_MODELS", models):
            with patch.object(cld, "_TEMPERATURES", [0.5]):
                cld.collect(out, dry_run=True)
        with open(out, newline="") as f:
            return list(csv.DictReader(f))

    def test_pipeline_columns_present(self, tmp_path):
        rows = self._run_collect(tmp_path)
        headers = set(rows[0].keys())
        assert _PIPELINE_REQUIRED <= headers

    def test_row_count(self, tmp_path):
        rows = self._run_collect(tmp_path, n_models=2)
        assert len(rows) == 60  # 2 responders × 30 trials

    def test_completed_true(self, tmp_path):
        rows = self._run_collect(tmp_path)
        assert all(r["completed"] == "True" for r in rows)

    def test_attention_check_empty(self, tmp_path):
        rows = self._run_collect(tmp_path)
        assert all(r["attention_check_passed"] == "" for r in rows)

    def test_choice_is_0_or_1_or_empty(self, tmp_path):
        rows = self._run_collect(tmp_path)
        for r in rows:
            assert r["choice"] in ("0", "1", "")
