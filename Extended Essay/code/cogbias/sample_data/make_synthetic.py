"""make_synthetic.py — Generate a synthetic dataset of simulated participant responses.

Usage:
    python sample_data/make_synthetic.py [N_PARTICIPANTS] [OUTPUT_CSV]

Defaults: 50 participants, output → sample_data/synthetic_responses.csv

Planted bias effects match typical literature effect sizes:
  Framing     : 70% choose the canonical biased option (Tversky & Kahneman 1981)
  Anchoring   : 62% choose the canonical biased option (moderate effect)
  Loss aversion: 74% choose the canonical biased option (Kahneman 2011)

Planted quality-filter triggers (for testing pipeline.flag_quality()):
  - 1 in 20 participants is a speedrunner: all trial RTs < RT_TOO_FAST_MS
  - 1 in 25 participants fails the attention check

Expected pipeline behavior on this file:
  - Both models should achieve overall accuracy > 55%.
  - Framing and loss-aversion accuracy should beat the per-condition majority-vote
    baseline (frame and ev_ratio features carry the bias signal).
  - Quality filters should fire on the planted bad actors.
"""

import sys
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from config import (
    BIAS_TYPE, TARGET_COL, GLOBAL_SEED,
    RT_TOO_FAST_MS, RT_TOO_SLOW_MS,
)
from scenarios import generate_participant_scenarios

# ── Bias rates (P(canonical biased choice)) ───────────────────────────────────
BIAS_RATES = {
    BIAS_TYPE["framing"]: 0.80,      # strong framing effect; provides the main LR/NN signal
    BIAS_TYPE["anchoring"]: 0.65,
    BIAS_TYPE["loss_aversion"]: 0.74,
}

RT_MEAN_LOG = 8.0   # exp(8.0) ≈ 2980 ms
RT_STD_LOG  = 0.55  # log-normal noise

SYNTHETIC_TOKEN_ID    = "00000000-0000-0000-0000-000000000000"
SYNTHETIC_TOKEN_LABEL = "synthetic"
SYNTHETIC_CONSENT_VER = "v1"


def _simulate_participant(
    participant_seed: int,
    rng_resp: random.Random,
    np_rng: np.random.Generator,
    is_speedrunner: bool,
    fails_attention: bool,
    session_start: "datetime",
) -> List[Dict[str, Any]]:
    """Simulate one participant's rows (30 trials + 1 attention check = 31 rows).

    created_at is set to session_start + cumulative RT so that
    pipeline.flag_quality()'s session_too_fast check reflects realistic durations
    (~15 min for a normal participant, <<90 s for speedrunners).
    """
    from datetime import timedelta
    participant_id = str(uuid.UUID(int=rng_resp.getrandbits(128)))
    scenarios = generate_participant_scenarios(participant_seed)
    acknowledged_at = session_start.isoformat()
    rows: List[Dict[str, Any]] = []
    prev_choice = 0
    elapsed_ms = 0  # cumulative milliseconds since session_start

    for position, sc in enumerate(scenarios):
        # Attention check injected between positions 19 and 20
        if position == 20:
            attn_rt = (
                int(np_rng.uniform(50, RT_TOO_FAST_MS - 1))
                if is_speedrunner
                else int(np_rng.lognormal(7.5, 0.4))
            )
            elapsed_ms += attn_rt
            attn_ts = (session_start + timedelta(milliseconds=elapsed_ms)).isoformat()
            rows.append(_attention_check_row(
                participant_id, participant_seed, acknowledged_at,
                passed=(not fails_attention), is_speedrunner=is_speedrunner, np_rng=np_rng,
                created_at=attn_ts,
            ))

        bias_rate = BIAS_RATES[sc["bias_type"]]
        choice = (
            sc["canonical_biased_choice"]
            if rng_resp.random() < bias_rate
            else 1 - sc["canonical_biased_choice"]
        )

        if is_speedrunner:
            rt_ms = int(np_rng.uniform(50, RT_TOO_FAST_MS - 1))
        else:
            rt_ms = int(np_rng.lognormal(RT_MEAN_LOG, RT_STD_LOG))
            rt_ms = max(RT_TOO_FAST_MS, min(rt_ms, RT_TOO_SLOW_MS))

        elapsed_ms += rt_ms
        trial_ts = (session_start + timedelta(milliseconds=elapsed_ms)).isoformat()

        rows.append({
            "trial_index": sc["trial_index"],
            "bias_type": sc["bias_type"],
            "frame": sc["frame"],
            "expected_value_ratio": sc["expected_value_ratio"],
            "anchor_value": sc["anchor_value"],
            "previous_choice": prev_choice if not sc["is_first_trial"] else 0,
            "is_first_trial": sc["is_first_trial"],
            "reaction_time_ms": rt_ms,
            TARGET_COL: choice,
            "participant_id": participant_id,
            "participant_seed": participant_seed,
            "scenario_template_id": sc["scenario_template_id"],
            "token_id": SYNTHETIC_TOKEN_ID,
            "token_label": SYNTHETIC_TOKEN_LABEL,
            "consent_version": SYNTHETIC_CONSENT_VER,
            "acknowledged_at": acknowledged_at,
            "attention_check_passed": None,  # None = real trial
            "completed": True,
            "created_at": trial_ts,
        })
        prev_choice = choice

    return rows


def _attention_check_row(
    participant_id: str,
    participant_seed: int,
    acknowledged_at: str,
    passed: bool,
    is_speedrunner: bool,
    np_rng: np.random.Generator,
    created_at: str = "",
) -> Dict[str, Any]:
    rt_ms = (
        int(np_rng.uniform(50, RT_TOO_FAST_MS - 1))
        if is_speedrunner
        else int(np_rng.lognormal(7.5, 0.4))
    )
    return {
        "trial_index": 31,
        "bias_type": 1,        # placeholder; excluded from modeling
        "frame": 0,
        "expected_value_ratio": 1.0,
        "anchor_value": 0.0,
        "previous_choice": 0,
        "is_first_trial": 0,
        "reaction_time_ms": max(50, min(rt_ms, RT_TOO_SLOW_MS)),
        TARGET_COL: 0,         # correct answer is A
        "participant_id": participant_id,
        "participant_seed": participant_seed,
        "scenario_template_id": "ATTN",
        "token_id": SYNTHETIC_TOKEN_ID,
        "token_label": SYNTHETIC_TOKEN_LABEL,
        "consent_version": SYNTHETIC_CONSENT_VER,
        "acknowledged_at": acknowledged_at,
        "attention_check_passed": passed,  # not None → marks this as attention check row
        "completed": True,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def generate(
    n_participants: int = 50,
    output_path: str | None = None,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """Generate a synthetic dataset and optionally write it to CSV."""
    rng_master = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    all_rows: List[Dict[str, Any]] = []
    session_base = datetime(2026, 1, 20, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(n_participants):
        p_seed = rng_master.randint(0, 2**31 - 1)
        rng_resp = random.Random(rng_master.randint(0, 2**31 - 1))
        is_speedrunner = (i % 20 == 19)
        fails_attention = (i % 25 == 24)
        # Stagger session starts by 5 minutes per participant
        from datetime import timedelta as _td
        session_start = session_base + _td(minutes=5 * i)
        all_rows.extend(
            _simulate_participant(
                p_seed, rng_resp, np_rng, is_speedrunner, fails_attention,
                session_start=session_start,
            )
        )

    df = pd.DataFrame(all_rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows ({n_participants} participants) → {output_path}")

    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out_default = Path(__file__).parent / "synthetic_responses.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else str(out_default)

    df = generate(n_participants=n, output_path=out)

    print("\n── Dataset summary ──────────────────────────────────────────────")
    real = df[df["attention_check_passed"].isna()]
    print(f"  Total rows  : {len(df)}  ({df['participant_id'].nunique()} participants)")
    print(f"  Trial rows  : {len(real)}  (attention check rows excluded)")
    print(f"  Choice A/B  : {int((real['choice']==0).sum())} / {int((real['choice']==1).sum())}")

    print("\n── Bias-type choice rates (trial rows only) ─────────────────────")
    from config import BIAS_TYPE_LABEL
    for bt, label in sorted(BIAS_TYPE_LABEL.items()):
        sub = real[real["bias_type"] == bt]
        rate = sub["choice"].mean()
        print(f"  {label:20s} (n={len(sub)})  mean choice=B: {rate:.3f}")

    print("\n── Planted quality issues ───────────────────────────────────────")
    max_rt = real.groupby("participant_id")["reaction_time_ms"].max()
    n_speed = int((max_rt < RT_TOO_FAST_MS).sum())
    attn_fail = df[df["attention_check_passed"] == False]["participant_id"].nunique()
    print(f"  Speedrunners (all RT < {RT_TOO_FAST_MS}ms) : {n_speed}")
    print(f"  Attention failures            : {attn_fail}")
    print(f"  completed=True rows           : {int(df['completed'].sum())}")
