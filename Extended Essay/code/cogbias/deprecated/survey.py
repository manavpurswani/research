# SECURITY: survey.py MUST NOT import client_admin() from storage.py.
# This file uses only the anon Supabase key (INSERT-only via RLS).
# Any import of client_admin, fetch_all, export_csv, admin_set_collection,
# admin_log_audit, admin_generate_tokens, admin_get_tokens, admin_health_check,
# or admin_sanity_check is FORBIDDEN here. See storage.py for the security model.

"""survey.py — Streamlit survey flow for the cognitive-bias experiment.

Entry point: run_survey(token: str) called from app.py with the validated UUID
extracted from the ?token= query parameter.

Flow
----
  Screen "consent"       -> ACK_V1 mini-consent checkbox
  Screen "instructions"  -> task description
  Screen "trial"         -> 30 binary-choice trials (0-indexed positions 0-29)
  Screen "attention"     -> attention check (injected after position 19)
  Screen "finish"        -> mark_session_complete, show completion code

Data insertion
--------------
  Each trial/attention-check row is inserted immediately after the participant
  answers (completed=False).  After all 31 rows are inserted a single call to
  mark_session_complete(participant_id) flips every row to completed=True.

Reaction time
-------------
  time.monotonic() is used for all RT measurements (NOT time.time()).
"""

from __future__ import annotations

import random
import time
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

from config import CONSENT_VERSION
from consent_text import ACK_V1
from scenarios import generate_participant_scenarios
from storage import (
    consume_token,
    insert_trial,
    is_collection_open,
    mark_session_complete,
    verify_token,
)
from tokens import is_valid_uuid


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_ATTENTION_CHECK_POSITION = 20  # inserted BEFORE this 0-based position (after trial 19)
_ATTENTION_CORRECT = 0          # correct answer is Option A


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session(token: str) -> None:
    defaults: Dict[str, Any] = {
        "survey_screen":     "consent",
        "token":             token,
        "participant_id":    None,
        "participant_seed":  None,
        "scenarios":         None,
        "acknowledged_at":   None,
        "trial_position":    0,
        "prev_choice":       0,
        "_attn_done":        False,
        "_display_mono":     None,
        "_token_consumed":   False,
        "_all_inserted":     False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_trial_row(
    sc: Dict[str, Any],
    choice: int,
    rt_ms: int,
    attention_check_passed: Optional[bool] = None,
) -> None:
    row: Dict[str, Any] = {
        "trial_index":            sc["trial_index"],
        "bias_type":              sc["bias_type"],
        "frame":                  sc["frame"],
        "expected_value_ratio":   sc["expected_value_ratio"],
        "anchor_value":           sc["anchor_value"],
        "previous_choice":        st.session_state["prev_choice"],
        "is_first_trial":         sc["is_first_trial"],
        "reaction_time_ms":       rt_ms,
        "choice":                 choice,
        "participant_id":         st.session_state["participant_id"],
        "participant_seed":       st.session_state["participant_seed"],
        "scenario_template_id":   sc["scenario_template_id"],
        "token_id":               st.session_state["token"],
        "token_label":            "",
        "consent_version":        CONSENT_VERSION,
        "acknowledged_at":        st.session_state["acknowledged_at"],
        "attention_check_passed": attention_check_passed,
        "completed":              False,
    }
    try:
        insert_trial(row)
    except Exception as exc:
        st.error(
            f"A database error occurred on trial {sc['trial_index']}. "
            "Please message the researcher immediately with your completion code: "
            f"`{st.session_state['participant_id']}`\n\nError: {exc}"
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Attention-check pseudo-scenario
# ─────────────────────────────────────────────────────────────────────────────

_ATTN_SCENARIO: Dict[str, Any] = {
    "trial_index":          31,
    "bias_type":            1,
    "frame":                0,
    "expected_value_ratio": 1.0,
    "anchor_value":         0.0,
    "scenario_template_id": "ATTN",
    "problem_text": (
        "**Comprehension check** — To confirm you are reading carefully, "
        "please select **Option A** below."
    ),
    "option_a_text": "Select this option.",
    "option_b_text": "Do not select this option.",
    "is_first_trial": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Screen: gate error
# ─────────────────────────────────────────────────────────────────────────────

def _screen_gate(reason: str) -> None:
    st.error(reason)
    st.markdown(
        "If you believe this is a mistake, please contact the researcher "
        "and quote any error text shown above."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Screen: consent
# ─────────────────────────────────────────────────────────────────────────────

def _screen_consent() -> None:
    st.markdown(ACK_V1)
    checked = st.checkbox("I have read the above and agree to participate.", key="_ck")
    if st.button("Start Survey", type="primary", disabled=not checked):
        if not consume_token(st.session_state["token"]):
            st.error(
                "This invitation link has already been used or has expired. "
                "Each link is single-use. Please contact the researcher for a new one."
            )
            st.stop()
        pid  = str(_uuid_mod.uuid4())
        seed = random.randint(0, 2**31 - 1)
        st.session_state.update({
            "participant_id":   pid,
            "participant_seed": seed,
            "scenarios":        generate_participant_scenarios(seed),
            "acknowledged_at":  _now_iso(),
            "_token_consumed":  True,
            "survey_screen":    "instructions",
        })
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Screen: instructions
# ─────────────────────────────────────────────────────────────────────────────

def _screen_instructions() -> None:
    st.title("Instructions")
    st.markdown("""
You will be shown **30 short scenarios**, one at a time.  For each one, choose
**Option A** or **Option B** — there are no right or wrong answers.

- Once you click an option you move on immediately. **You cannot go back.**
- Please complete all questions in **one sitting** without interruptions.
- A progress bar at the top shows how far you are.

When you are ready, click **Begin** below.
""")
    if st.button("Begin", type="primary"):
        st.session_state["survey_screen"] = "trial"
        st.session_state["_display_mono"] = time.monotonic()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Screen: trial
# ─────────────────────────────────────────────────────────────────────────────

def _screen_trial() -> None:
    pos       = st.session_state["trial_position"]
    scenarios = st.session_state["scenarios"]
    total_q   = len(scenarios) + 1  # 31 questions total (30 + attention check)

    progress_steps = pos + (1 if st.session_state["_attn_done"] else 0)
    st.progress(progress_steps / total_q, text=f"Question {progress_steps + 1} of {total_q}")
    st.divider()

    sc = scenarios[pos]
    st.markdown(f"### {sc['problem_text']}")
    st.write("")

    def _on_choice(choice: int) -> None:
        rt_ms = max(0, round((time.monotonic() - st.session_state["_display_mono"]) * 1000))
        _insert_trial_row(sc, choice, rt_ms)
        st.session_state["prev_choice"] = choice
        next_pos = pos + 1
        if next_pos == _ATTENTION_CHECK_POSITION and not st.session_state["_attn_done"]:
            st.session_state["survey_screen"] = "attention"
        elif next_pos >= len(scenarios):
            st.session_state["survey_screen"] = "finish"
        else:
            st.session_state["survey_screen"] = "trial"
        st.session_state["trial_position"] = next_pos
        st.session_state["_display_mono"]  = time.monotonic()
        st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div style="border:2px solid #4C72B0;border-radius:10px;'
            f'padding:16px 20px;min-height:100px;">'
            f'<strong>Option A</strong><br><br>{sc["option_a_text"]}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Choose Option A", key=f"a_{pos}", use_container_width=True):
            _on_choice(0)

    with col_b:
        st.markdown(
            f'<div style="border:2px solid #DD8452;border-radius:10px;'
            f'padding:16px 20px;min-height:100px;">'
            f'<strong>Option B</strong><br><br>{sc["option_b_text"]}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Choose Option B", key=f"b_{pos}", use_container_width=True):
            _on_choice(1)


# ─────────────────────────────────────────────────────────────────────────────
# Screen: attention check
# ─────────────────────────────────────────────────────────────────────────────

def _screen_attention() -> None:
    pos = st.session_state["trial_position"]
    total_q = len(st.session_state["scenarios"]) + 1
    st.progress(pos / total_q, text=f"Question {pos + 1} of {total_q}")
    st.divider()

    sc = _ATTN_SCENARIO
    st.markdown(f"### {sc['problem_text']}")
    st.write("")

    def _on_attn(choice: int) -> None:
        rt_ms  = max(0, round((time.monotonic() - st.session_state["_display_mono"]) * 1000))
        passed = choice == _ATTENTION_CORRECT
        _insert_trial_row(sc, choice, rt_ms, attention_check_passed=passed)
        st.session_state["_attn_done"] = True
        remaining = len(st.session_state["scenarios"]) - st.session_state["trial_position"]
        next_screen = "finish" if remaining <= 0 else "trial"
        st.session_state["survey_screen"] = next_screen
        st.session_state["_display_mono"] = time.monotonic()
        st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div style="border:2px solid #4C72B0;border-radius:10px;'
            f'padding:16px 20px;min-height:80px;">'
            f'<strong>Option A</strong><br><br>{sc["option_a_text"]}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Choose Option A", key="attn_a", use_container_width=True):
            _on_attn(0)
    with col_b:
        st.markdown(
            f'<div style="border:2px solid #DD8452;border-radius:10px;'
            f'padding:16px 20px;min-height:80px;">'
            f'<strong>Option B</strong><br><br>{sc["option_b_text"]}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Choose Option B", key="attn_b", use_container_width=True):
            _on_attn(1)


# ─────────────────────────────────────────────────────────────────────────────
# Screen: finish
# ─────────────────────────────────────────────────────────────────────────────

def _screen_finish() -> None:
    if not st.session_state.get("_all_inserted"):
        pid = st.session_state["participant_id"]
        try:
            mark_session_complete(pid)
            st.session_state["_all_inserted"] = True
        except Exception as exc:
            st.error(
                "There was a problem finalising your session. "
                "Please message the researcher with your completion code below.\n"
                f"Error: {exc}"
            )
    pid = st.session_state["participant_id"]
    st.title("Thank You!")
    st.markdown(f"""
Your responses have been recorded anonymously. Thank you for participating!

**Your completion code:** `{pid}`

Please send this code to the researcher to confirm your participation.
You may now close this tab.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_survey(token: str) -> None:
    """Called by app.py after extracting the ?token= query parameter."""
    if not is_valid_uuid(token):
        _screen_gate("Invalid or missing invitation link. Each survey requires a unique URL.")
        return

    _init_session(token)
    screen = st.session_state["survey_screen"]

    if not st.session_state.get("_token_consumed"):
        try:
            if not verify_token(token):
                _screen_gate(
                    "This invitation link has already been used or is not valid. "
                    "Please contact the researcher if you believe this is an error."
                )
                return
        except Exception:
            _screen_gate(
                "Could not verify your invitation link (database unreachable). "
                "Please try again in a moment or contact the researcher."
            )
            return

    if screen == "consent":
        try:
            if not is_collection_open():
                _screen_gate(
                    "Data collection is not currently open. "
                    "Please check back later or contact the researcher."
                )
                return
        except Exception:
            pass  # fail open on transient DB errors
        _screen_consent()
    elif screen == "instructions":
        _screen_instructions()
    elif screen == "trial":
        _screen_trial()
    elif screen == "attention":
        _screen_attention()
    elif screen == "finish":
        _screen_finish()
    else:
        _screen_gate(f"Unknown survey state: {screen!r}. Please contact the researcher.")
