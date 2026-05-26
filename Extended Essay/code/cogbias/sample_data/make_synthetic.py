"""
make_synthetic.py — Generate a synthetic dataset of simulated participant responses.

Usage:
    python sample_data/make_synthetic.py [N_PARTICIPANTS] [OUTPUT_CSV]

Defaults: 40 participants, output → sample_data/synthetic_responses.csv

The synthetic participants exhibit planted, realistic bias effects:
  Framing     : 70 % choose the canonical biased option  (strong, well-replicated effect)
  Anchoring   : 62 % choose the canonical biased option  (moderate effect)
  Loss aversion: 74 % choose the canonical biased option (strong effect)

These proportions match typical effect sizes reported in the literature and are
sufficient for the ML pipeline to beat both the chance (50 %) and majority-vote
baselines on at least two of the three bias types.

Expected behaviour when pipeline.py is run on this file:
  - Both models should achieve overall accuracy > 60 %.
  - Framing and loss-aversion accuracy should beat the per-condition majority-vote
    baseline (because the model can leverage the 'frame' and 'ev_ratio' features).
  - Anchoring accuracy may be close to the majority-vote baseline (the anchor_value
    feature carries the signal, but individual variation is modelled as noise here).
"""

import os
import sys
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Allow importing from the parent directory (cogbias/)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from config import (
    BIAS_TYPE, FEATURE_COLS, METADATA_COLS, TARGET_COL, GLOBAL_SEED
)
from scenarios import generate_participant_scenarios

# ── Bias rates (probability of choosing the canonical biased option) ──────────
BIAS_RATES = {
    BIAS_TYPE["framing"]: 0.70,
    BIAS_TYPE["anchoring"]: 0.62,
    BIAS_TYPE["loss_aversion"]: 0.74,
}

# ── Reaction-time distribution (log-normal, in milliseconds) ──────────────────
RT_MEAN_LOG = 8.0      # exp(8.0) ≈ 2980 ms
RT_STD_LOG = 0.55      # gives a realistic right-skewed distribution


def _simulate_participant(
    participant_seed: int,
    rng_resp: random.Random,
    np_rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Simulate one participant's 30 responses with realistic bias effects."""
    participant_id = str(uuid.UUID(int=rng_resp.getrandbits(128)))
    scenarios = generate_participant_scenarios(participant_seed)
    rows: List[Dict[str, Any]] = []
    prev_choice = 0

    for sc in scenarios:
        bias_t = sc["bias_type"]
        bias_rate = BIAS_RATES[bias_t]

        # Simulate choice: biased with probability bias_rate
        if rng_resp.random() < bias_rate:
            choice = sc["canonical_biased_choice"]
        else:
            choice = 1 - sc["canonical_biased_choice"]

        # Simulate reaction time (log-normal)
        rt_ms = int(np_rng.lognormal(RT_MEAN_LOG, RT_STD_LOG))
        rt_ms = max(300, min(rt_ms, 60_000))  # clamp to plausible range

        row: Dict[str, Any] = {
            # Feature columns
            "trial_index": sc["trial_index"],
            "bias_type": sc["bias_type"],
            "frame": sc["frame"],
            "expected_value_ratio": sc["expected_value_ratio"],
            "anchor_value": sc["anchor_value"],
            "previous_choice": prev_choice if not sc["is_first_trial"] else 0,
            "is_first_trial": sc["is_first_trial"],
            "reaction_time_ms": rt_ms,
            # Target
            TARGET_COL: choice,
            # Metadata
            "participant_id": participant_id,
            "participant_seed": participant_seed,
            "scenario_template_id": sc["scenario_template_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        prev_choice = choice

    return rows


def generate(
    n_participants: int = 40,
    output_path: str | None = None,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset and optionally write it to CSV.

    Parameters
    ----------
    n_participants : number of simulated participants
    output_path    : CSV path; if None, nothing is written to disk
    seed           : master seed for reproducibility

    Returns
    -------
    pd.DataFrame with all responses (n_participants × 30 rows)
    """
    rng_master = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    all_rows: List[Dict[str, Any]] = []
    for i in range(n_participants):
        p_seed = rng_master.randint(0, 2**31 - 1)
        rng_resp = random.Random(rng_master.randint(0, 2**31 - 1))
        all_rows.extend(_simulate_participant(p_seed, rng_resp, np_rng))

    df = pd.DataFrame(all_rows)
    column_order = (
        FEATURE_COLS + [TARGET_COL] + METADATA_COLS
    )
    # Keep only columns that exist (safety)
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows ({n_participants} participants) → {output_path}")

    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out_default = Path(__file__).parent / "synthetic_responses.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else str(out_default)

    df = generate(n_participants=n, output_path=out)

    print("\n── Dataset summary ──────────────────────────────────────────────")
    print(f"  Rows : {len(df)}  ({df['participant_id'].nunique()} participants × 30 trials)")
    print(f"  Choice balance: A={int((df['choice']==0).sum())}  B={int((df['choice']==1).sum())}")
    print("\n── Bias-type choice rates ───────────────────────────────────────")
    from config import BIAS_TYPE_LABEL
    for bt, label in sorted(BIAS_TYPE_LABEL.items()):
        sub = df[df["bias_type"] == bt]
        b_rate = sub["choice"].mean()
        print(f"  {label:20s} (n={len(sub)})  mean choice=B: {b_rate:.3f}")
    print("\n── First 3 rows ─────────────────────────────────────────────────")
    print(df.head(3).to_string())
