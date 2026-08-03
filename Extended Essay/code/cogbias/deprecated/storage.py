"""storage.py — Supabase persistence layer for the cognitive-bias survey.

# SECURITY — READ BEFORE EDITING
# ─────────────────────────────────────────────────────────────────────────────
# This module exposes TWO separate client factories:
#
#   client_anon()   — uses SUPABASE_ANON_KEY (survey path, INSERT-only via RLS)
#   client_admin()  — uses SUPABASE_SERVICE_ROLE_KEY (admin path, full access)
#
# survey.py MUST ONLY call the functions listed under "Survey-safe API" below.
# survey.py MUST NEVER import or call client_admin() or any admin_* function.
# ─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

try:
    from supabase import create_client, Client as _Client
    _HAS_SUPABASE = True
except ImportError:
    _HAS_SUPABASE = False

from config import (
    SUPABASE_TABLE, SUPABASE_TOKENS_TABLE, SUPABASE_CONFIG_TABLE,
    SUPABASE_AUDIT_TABLE, FEATURE_COLS, TARGET_COL,
)

# Columns the DB accepts for a trial/attention-check row
_DB_TRIAL_COLS: frozenset = frozenset(
    FEATURE_COLS + [TARGET_COL,
    "participant_id", "participant_seed", "scenario_template_id",
    "token_id", "token_label", "consent_version", "acknowledged_at",
    "attention_check_passed", "completed"]
)


# ─────────────────────────────────────────────────────────────────────────────
# Client factories
# ─────────────────────────────────────────────────────────────────────────────

def client_anon() -> "_Client":
    """Return a Supabase client authenticated with the anon (public) key.

    Raises RuntimeError if credentials are not in st.secrets or if the
    supabase-py package is not installed.  Never used for admin operations.
    """
    if not _HAS_SUPABASE:
        raise RuntimeError("supabase-py is not installed. Run: pip install supabase")
    if not _HAS_STREAMLIT:
        raise RuntimeError("Streamlit secrets not available outside Streamlit context")
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except KeyError as exc:
        raise RuntimeError(f"Missing Streamlit secret for anon client: {exc}") from exc
    return create_client(url, key)


def client_admin() -> "_Client":
    """Return a Supabase client authenticated with the service-role key.

    MUST NOT be imported or called from survey.py.
    Raises RuntimeError if credentials are not in st.secrets.
    """
    if not _HAS_SUPABASE:
        raise RuntimeError("supabase-py is not installed. Run: pip install supabase")
    if not _HAS_STREAMLIT:
        raise RuntimeError("Streamlit secrets not available outside Streamlit context")
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    except KeyError as exc:
        raise RuntimeError(f"Missing Streamlit secret for admin client: {exc}") from exc
    return create_client(url, key)


# ─────────────────────────────────────────────────────────────────────────────
# Survey-safe API  (anon key only — safe to call from survey.py)
# ─────────────────────────────────────────────────────────────────────────────

def is_collection_open() -> bool:
    """Return True if the admin has opened data collection.

    Calls the SECURITY DEFINER function so the anon key can read the toggle
    without direct access to the config table.
    """
    c = client_anon()
    result = c.rpc("is_collection_open", {}).execute()
    return bool(result.data)


def verify_token(token: str) -> bool:
    """Return True if the token UUID exists and has not been consumed."""
    c = client_anon()
    result = c.rpc("verify_token", {"p_token": token}).execute()
    return bool(result.data)


def consume_token(token: str) -> bool:
    """Atomically consume the token. Returns True if this call consumed it."""
    c = client_anon()
    result = c.rpc("consume_token", {"p_token": token}).execute()
    return bool(result.data)


def insert_trial(row: Dict[str, Any]) -> bool:
    """Insert a single trial (or attention-check) row via the anon client.

    Filters the dict to only DB-known columns.  Returns True on success.
    Raises on DB error so the caller can handle retry / abort logic.
    """
    c = client_anon()
    clean = {k: v for k, v in row.items() if k in _DB_TRIAL_COLS}
    c.table(SUPABASE_TABLE).insert(clean).execute()
    return True


def mark_session_complete(participant_id: str) -> None:
    """Flip completed=True for all rows belonging to this participant.

    Calls the SECURITY DEFINER function so the anon key can UPDATE rows
    without a direct UPDATE grant on the table.
    """
    c = client_anon()
    c.rpc("mark_session_complete", {"p_participant_id": participant_id}).execute()


# ─────────────────────────────────────────────────────────────────────────────
# Admin API  (service-role key — MUST NOT be called from survey.py)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all() -> pd.DataFrame:
    """Fetch all completed sessions from the responses table (admin only).

    Returns an empty DataFrame if there is no data.
    Raises on error.
    """
    c = client_admin()
    result = c.table(SUPABASE_TABLE).select("*").execute()
    if not result.data:
        return pd.DataFrame()
    return pd.DataFrame(result.data)


def export_csv(path: str | Path) -> bool:
    """Fetch all data and write to path as CSV. Returns True on success."""
    df = fetch_all()
    df.to_csv(path, index=False)
    return True


def admin_set_collection(open: bool) -> None:
    """Toggle data collection on/off (admin only)."""
    c = client_admin()
    c.table(SUPABASE_CONFIG_TABLE).upsert(
        {"key": "collection_open", "value": "true" if open else "false"}
    ).execute()


def admin_log_audit(action: str, detail: Optional[Dict[str, Any]] = None) -> None:
    """Append a row to admin_audit (admin only)."""
    c = client_admin()
    row: Dict[str, Any] = {"action": action}
    if detail:
        row["detail"] = detail
    c.table(SUPABASE_AUDIT_TABLE).insert(row).execute()


def admin_generate_tokens(n: int, label: str) -> List[Dict[str, Any]]:
    """Generate n new tokens with the given label. Returns the inserted rows."""
    if n < 1 or n > 200:
        raise ValueError(f"Token count must be 1-200; got {n}")
    c = client_admin()
    rows = [{"id": str(uuid.uuid4()), "label": label} for _ in range(n)]
    result = c.table(SUPABASE_TOKENS_TABLE).insert(rows).execute()
    return result.data or rows


def admin_get_tokens() -> pd.DataFrame:
    """Return all tokens as a DataFrame (admin only)."""
    c = client_admin()
    result = c.table(SUPABASE_TOKENS_TABLE).select("*").execute()
    if not result.data:
        return pd.DataFrame()
    return pd.DataFrame(result.data)


def admin_health_check() -> Dict[str, Any]:
    """Return basic counts for the health dashboard (admin only)."""
    c = client_admin()
    responses_all = c.table(SUPABASE_TABLE).select("participant_id", count="exact").execute()
    responses_done = (
        c.table(SUPABASE_TABLE)
        .select("participant_id", count="exact")
        .eq("completed", True)
        .execute()
    )
    tokens_all = c.table(SUPABASE_TOKENS_TABLE).select("id", count="exact").execute()
    tokens_used = (
        c.table(SUPABASE_TOKENS_TABLE)
        .select("id", count="exact")
        .not_.is_("used_at", "null")
        .execute()
    )
    return {
        "total_rows":        responses_all.count or 0,
        "completed_rows":    responses_done.count or 0,
        "tokens_total":      tokens_all.count or 0,
        "tokens_used":       tokens_used.count or 0,
        "tokens_available":  (tokens_all.count or 0) - (tokens_used.count or 0),
    }


def admin_sanity_check() -> List[str]:
    """Return a list of warning strings (empty = no problems) (admin only)."""
    c = client_admin()
    warnings: List[str] = []

    all_resp = c.table(SUPABASE_TABLE).select("participant_id, completed").execute()
    if not all_resp.data:
        warnings.append("No responses in database.")
        return warnings

    df = pd.DataFrame(all_resp.data)
    pid_counts = df.groupby("participant_id")["completed"].agg(["sum", "count"])

    incomplete = pid_counts[pid_counts["sum"] == 0]
    if not incomplete.empty:
        warnings.append(
            f"{len(incomplete)} participant(s) have rows but completed=False on all."
        )

    wrong_count = pid_counts[pid_counts["count"] != 31]
    if not wrong_count.empty:
        warnings.append(
            f"{len(wrong_count)} participant(s) have unexpected row count (expected 31)."
        )

    return warnings
