"""
storage.py — Supabase persistence layer.

Provides thin wrappers around supabase-py for inserting participant responses
and fetching / exporting the full dataset. All DB credentials are read from
Streamlit secrets so no secrets appear in source code.

Supabase DDL (run once in the Supabase SQL editor):
──────────────────────────────────────────────────────────────────────────────
CREATE TABLE responses (
    id                   BIGSERIAL PRIMARY KEY,
    participant_id       UUID        NOT NULL,
    participant_seed     INTEGER     NOT NULL,
    scenario_template_id TEXT        NOT NULL,
    trial_index          SMALLINT    NOT NULL,
    bias_type            SMALLINT    NOT NULL,
    frame                SMALLINT    NOT NULL,
    expected_value_ratio REAL        NOT NULL,
    anchor_value         REAL        NOT NULL,
    previous_choice      SMALLINT    NOT NULL,
    is_first_trial       SMALLINT    NOT NULL,
    reaction_time_ms     INTEGER     NOT NULL,
    choice               SMALLINT    NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_responses_participant ON responses (participant_id);
──────────────────────────────────────────────────────────────────────────────

Streamlit secrets (`.streamlit/secrets.toml`) must contain:
    SUPABASE_URL = "https://<project>.supabase.co"
    SUPABASE_KEY = "<anon-or-service-role-key>"
    ADMIN_PASSWORD = "<your-admin-password>"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

try:
    from supabase import create_client, Client as _SupabaseClient
    _HAS_SUPABASE = True
except ImportError:
    _HAS_SUPABASE = False

from config import SUPABASE_TABLE, FEATURE_COLS, TARGET_COL, METADATA_COLS

# Column order to use when inserting rows
_INSERT_COLS = FEATURE_COLS + [TARGET_COL] + [
    c for c in METADATA_COLS if c != "created_at"  # created_at defaults to NOW()
]

_client: Optional[_SupabaseClient] = None  # lazily initialised


def init_client() -> Optional[_SupabaseClient]:
    """
    Create and cache the Supabase client using Streamlit secrets.

    Returns None and shows a Streamlit error if credentials are missing or
    the supabase-py package is not installed.
    """
    global _client
    if _client is not None:
        return _client

    if not _HAS_SUPABASE:
        if _HAS_STREAMLIT:
            st.error(
                "The `supabase` package is not installed. "
                "Run `pip install supabase` and restart the app."
            )
        return None

    if not _HAS_STREAMLIT:
        return None  # Running outside Streamlit (e.g., tests); skip DB init

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except KeyError as exc:
        st.error(
            f"Missing Streamlit secret: {exc}. "
            "Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
            "and fill in your Supabase credentials."
        )
        return None

    try:
        _client = create_client(url, key)
        return _client
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not connect to Supabase: {exc}")
        return None


def insert_responses(rows: List[Dict[str, Any]]) -> bool:
    """
    Batch-insert one participant's 30 trial rows into Supabase.

    Parameters
    ----------
    rows : list of dicts — one dict per trial, matching the responses table schema

    Returns
    -------
    True on success, False on failure (error is shown via st.error).
    """
    client = init_client()
    if client is None:
        return False

    # Keep only columns the DB knows about
    clean_rows = [
        {k: v for k, v in row.items() if k in set(_INSERT_COLS)}
        for row in rows
    ]

    try:
        client.table(SUPABASE_TABLE).insert(clean_rows).execute()
        return True
    except Exception as exc:  # noqa: BLE001
        if _HAS_STREAMLIT:
            st.error(f"Database insert failed: {exc}")
        return False


def fetch_all() -> Optional[pd.DataFrame]:
    """
    Fetch every row from the responses table.

    Returns a DataFrame, or None if the fetch fails.
    """
    client = init_client()
    if client is None:
        return None

    try:
        result = client.table(SUPABASE_TABLE).select("*").execute()
        if not result.data:
            return pd.DataFrame()
        return pd.DataFrame(result.data)
    except Exception as exc:  # noqa: BLE001
        if _HAS_STREAMLIT:
            st.error(f"Database fetch failed: {exc}")
        return None


def export_csv(path: str | Path) -> bool:
    """
    Fetch all data and write it to a CSV file.

    Returns True on success, False on failure.
    """
    df = fetch_all()
    if df is None:
        return False
    df.to_csv(path, index=False)
    return True


def check_connection() -> bool:
    """Lightweight connection check — returns True if Supabase is reachable."""
    client = init_client()
    if client is None:
        return False
    try:
        client.table(SUPABASE_TABLE).select("id").limit(1).execute()
        return True
    except Exception:  # noqa: BLE001
        return False
