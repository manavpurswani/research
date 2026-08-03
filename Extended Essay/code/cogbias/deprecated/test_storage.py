"""test_storage.py — Unit tests for storage.py.

All tests mock the Supabase client so no live database connection is needed.
Run from the cogbias/ directory:
    pytest tests/test_storage.py -v
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import storage
from config import (
    FEATURE_COLS, SUPABASE_AUDIT_TABLE, SUPABASE_CONFIG_TABLE,
    SUPABASE_TABLE, SUPABASE_TOKENS_TABLE, TARGET_COL,
)
from storage import _DB_TRIAL_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: minimal fake Supabase client + chain builder
# ─────────────────────────────────────────────────────────────────────────────

def _fake_result(data=None, count=None):
    r = MagicMock()
    r.data = data if data is not None else []
    r.count = count
    return r


def _chain_mock(**kwargs) -> MagicMock:
    """Build a mock that returns _fake_result(**kwargs) at the end of any chain."""
    result = _fake_result(**kwargs)
    m = MagicMock()
    m.table.return_value = m
    m.select.return_value = m
    m.insert.return_value = m
    m.upsert.return_value = m
    m.update.return_value = m
    m.eq.return_value = m
    m.not_.is_.return_value = m
    m.limit.return_value = m
    m.rpc.return_value = m
    m.execute.return_value = result
    return m


# ─────────────────────────────────────────────────────────────────────────────
# _DB_TRIAL_COLS
# ─────────────────────────────────────────────────────────────────────────────

class TestDBTrialCols:
    def test_contains_all_feature_cols(self):
        for col in FEATURE_COLS:
            assert col in _DB_TRIAL_COLS, f"{col} missing from _DB_TRIAL_COLS"

    def test_contains_target_col(self):
        assert TARGET_COL in _DB_TRIAL_COLS

    def test_contains_metadata_cols(self):
        for col in ["participant_id", "token_id", "consent_version",
                    "acknowledged_at", "attention_check_passed", "completed"]:
            assert col in _DB_TRIAL_COLS, f"{col} missing from _DB_TRIAL_COLS"

    def test_does_not_contain_created_at(self):
        assert "created_at" not in _DB_TRIAL_COLS

    def test_is_frozenset(self):
        assert isinstance(_DB_TRIAL_COLS, frozenset)


# ─────────────────────────────────────────────────────────────────────────────
# client_anon / client_admin error paths (no secrets available)
# ─────────────────────────────────────────────────────────────────────────────

class TestClientFactories:
    def test_client_anon_raises_without_streamlit(self):
        with patch.object(storage, "_HAS_STREAMLIT", False):
            with pytest.raises(RuntimeError, match="Streamlit secrets"):
                storage.client_anon()

    def test_client_admin_raises_without_streamlit(self):
        with patch.object(storage, "_HAS_STREAMLIT", False):
            with pytest.raises(RuntimeError, match="Streamlit secrets"):
                storage.client_admin()

    def test_client_anon_raises_without_supabase(self):
        with patch.object(storage, "_HAS_SUPABASE", False):
            with pytest.raises(RuntimeError, match="supabase-py"):
                storage.client_anon()

    def test_client_admin_raises_without_supabase(self):
        with patch.object(storage, "_HAS_SUPABASE", False):
            with pytest.raises(RuntimeError, match="supabase-py"):
                storage.client_admin()

    def test_client_anon_uses_anon_key(self):
        fake_create = MagicMock(return_value=MagicMock())
        fake_secrets = {"SUPABASE_URL": "https://x.supabase.co", "SUPABASE_ANON_KEY": "anon-key"}
        with patch.object(storage, "_HAS_STREAMLIT", True), \
             patch.object(storage, "_HAS_SUPABASE", True), \
             patch("storage.st") as mock_st, \
             patch("storage.create_client", fake_create):
            mock_st.secrets.__getitem__.side_effect = fake_secrets.__getitem__
            storage.client_anon()
            fake_create.assert_called_once_with("https://x.supabase.co", "anon-key")

    def test_client_admin_uses_service_role_key(self):
        fake_create = MagicMock(return_value=MagicMock())
        fake_secrets = {
            "SUPABASE_URL": "https://x.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "svc-key",
        }
        with patch.object(storage, "_HAS_STREAMLIT", True), \
             patch.object(storage, "_HAS_SUPABASE", True), \
             patch("storage.st") as mock_st, \
             patch("storage.create_client", fake_create):
            mock_st.secrets.__getitem__.side_effect = fake_secrets.__getitem__
            storage.client_admin()
            fake_create.assert_called_once_with("https://x.supabase.co", "svc-key")

    def test_client_anon_missing_key_raises(self):
        with patch.object(storage, "_HAS_STREAMLIT", True), \
             patch.object(storage, "_HAS_SUPABASE", True), \
             patch("storage.st") as mock_st:
            mock_st.secrets.__getitem__.side_effect = KeyError("SUPABASE_ANON_KEY")
            with pytest.raises(RuntimeError, match="Missing Streamlit secret"):
                storage.client_anon()

    def test_client_admin_missing_key_raises(self):
        with patch.object(storage, "_HAS_STREAMLIT", True), \
             patch.object(storage, "_HAS_SUPABASE", True), \
             patch("storage.st") as mock_st:
            mock_st.secrets.__getitem__.side_effect = KeyError("SUPABASE_SERVICE_ROLE_KEY")
            with pytest.raises(RuntimeError, match="Missing Streamlit secret"):
                storage.client_admin()


# ─────────────────────────────────────────────────────────────────────────────
# insert_trial -- column filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestInsertTrial:
    def _mock_anon(self):
        return _chain_mock(data=[{"id": 1}])

    def _call_insert(self, row: dict, mock_client):
        with patch("storage.client_anon", return_value=mock_client):
            return storage.insert_trial(row)

    def test_returns_true_on_success(self):
        m = self._mock_anon()
        row = {col: 0 for col in _DB_TRIAL_COLS} | {"junk_col": "ignored"}
        assert self._call_insert(row, m) is True

    def test_unknown_columns_stripped(self):
        m = self._mock_anon()
        row = {col: 0 for col in _DB_TRIAL_COLS}
        row["__extra__"] = "should_be_stripped"
        self._call_insert(row, m)
        inserted = m.table.return_value.insert.call_args[0][0]
        assert "__extra__" not in inserted

    def test_all_db_cols_passed_through(self):
        m = self._mock_anon()
        row = {col: 0 for col in _DB_TRIAL_COLS}
        self._call_insert(row, m)
        inserted = m.table.return_value.insert.call_args[0][0]
        for col in _DB_TRIAL_COLS:
            assert col in inserted

    def test_uses_responses_table(self):
        m = self._mock_anon()
        row = {col: 0 for col in _DB_TRIAL_COLS}
        self._call_insert(row, m)
        m.table.assert_called_once_with(SUPABASE_TABLE)


# ─────────────────────────────────────────────────────────────────────────────
# Survey-safe RPC calls
# ─────────────────────────────────────────────────────────────────────────────

class TestSurveyRPCs:
    def test_verify_token_true(self):
        m = _chain_mock(data=True)
        with patch("storage.client_anon", return_value=m):
            assert storage.verify_token(str(uuid.uuid4())) is True

    def test_verify_token_false(self):
        m = _chain_mock(data=False)
        with patch("storage.client_anon", return_value=m):
            assert storage.verify_token(str(uuid.uuid4())) is False

    def test_consume_token_true(self):
        m = _chain_mock(data=True)
        with patch("storage.client_anon", return_value=m):
            assert storage.consume_token(str(uuid.uuid4())) is True

    def test_consume_token_false(self):
        m = _chain_mock(data=False)
        with patch("storage.client_anon", return_value=m):
            assert storage.consume_token(str(uuid.uuid4())) is False

    def test_is_collection_open_true(self):
        m = _chain_mock(data=True)
        with patch("storage.client_anon", return_value=m):
            assert storage.is_collection_open() is True

    def test_is_collection_open_false(self):
        m = _chain_mock(data=False)
        with patch("storage.client_anon", return_value=m):
            assert storage.is_collection_open() is False

    def test_mark_session_complete_calls_rpc(self):
        m = _chain_mock(data=None)
        pid = str(uuid.uuid4())
        with patch("storage.client_anon", return_value=m):
            storage.mark_session_complete(pid)
        m.rpc.assert_called_once_with(
            "mark_session_complete", {"p_participant_id": pid}
        )


# ─────────────────────────────────────────────────────────────────────────────
# Admin API
# ─────────────────────────────────────────────────────────────────────────────

class TestAdminFetchAll:
    def test_returns_dataframe(self):
        rows = [{"id": 1, "choice": 0}, {"id": 2, "choice": 1}]
        m = _chain_mock(data=rows)
        with patch("storage.client_admin", return_value=m):
            df = storage.fetch_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_returns_empty_df_when_no_data(self):
        m = _chain_mock(data=[])
        with patch("storage.client_admin", return_value=m):
            df = storage.fetch_all()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_uses_responses_table(self):
        m = _chain_mock(data=[])
        with patch("storage.client_admin", return_value=m):
            storage.fetch_all()
        m.table.assert_called_with(SUPABASE_TABLE)


class TestAdminSetCollection:
    def test_open_true_upserts_true(self):
        m = _chain_mock()
        with patch("storage.client_admin", return_value=m):
            storage.admin_set_collection(True)
        m.table.assert_called_with(SUPABASE_CONFIG_TABLE)
        upsert_arg = m.table.return_value.upsert.call_args[0][0]
        assert upsert_arg["value"] == "true"
        assert upsert_arg["key"] == "collection_open"

    def test_open_false_upserts_false(self):
        m = _chain_mock()
        with patch("storage.client_admin", return_value=m):
            storage.admin_set_collection(False)
        upsert_arg = m.table.return_value.upsert.call_args[0][0]
        assert upsert_arg["value"] == "false"


class TestAdminLogAudit:
    def test_inserts_into_audit_table(self):
        m = _chain_mock()
        with patch("storage.client_admin", return_value=m):
            storage.admin_log_audit("login_success", {"ip": "127.0.0.1"})
        m.table.assert_called_with(SUPABASE_AUDIT_TABLE)
        row = m.table.return_value.insert.call_args[0][0]
        assert row["action"] == "login_success"
        assert row["detail"] == {"ip": "127.0.0.1"}

    def test_detail_optional(self):
        m = _chain_mock()
        with patch("storage.client_admin", return_value=m):
            storage.admin_log_audit("export_csv")
        row = m.table.return_value.insert.call_args[0][0]
        assert "detail" not in row


class TestAdminGenerateTokens:
    def test_generates_correct_count(self):
        def fake_insert(rows):
            m2 = MagicMock()
            m2.execute.return_value = _fake_result(data=rows)
            return m2
        m = _chain_mock()
        m.table.return_value.insert.side_effect = fake_insert
        with patch("storage.client_admin", return_value=m):
            result = storage.admin_generate_tokens(5, "test-batch")
        assert len(result) == 5

    def test_all_ids_are_valid_uuids(self):
        def fake_insert(rows):
            m2 = MagicMock()
            m2.execute.return_value = _fake_result(data=rows)
            return m2
        m = _chain_mock()
        m.table.return_value.insert.side_effect = fake_insert
        with patch("storage.client_admin", return_value=m):
            result = storage.admin_generate_tokens(3, "batch")
        for row in result:
            uuid.UUID(row["id"])  # raises if invalid

    def test_raises_for_zero_tokens(self):
        m = _chain_mock()
        with patch("storage.client_admin", return_value=m):
            with pytest.raises(ValueError, match="1-200"):
                storage.admin_generate_tokens(0, "bad")

    def test_raises_for_too_many_tokens(self):
        m = _chain_mock()
        with patch("storage.client_admin", return_value=m):
            with pytest.raises(ValueError, match="1-200"):
                storage.admin_generate_tokens(201, "bad")

    def test_uses_tokens_table(self):
        m = _chain_mock()
        with patch("storage.client_admin", return_value=m):
            storage.admin_generate_tokens(1, "x")
        m.table.assert_called_with(SUPABASE_TOKENS_TABLE)


class TestAdminSanityCheck:
    def test_no_data_returns_warning(self):
        m = _chain_mock(data=[])
        with patch("storage.client_admin", return_value=m):
            warnings = storage.admin_sanity_check()
        assert len(warnings) == 1
        assert "No responses" in warnings[0]

    def test_clean_data_no_warnings(self):
        pid = str(uuid.uuid4())
        rows = [{"participant_id": pid, "completed": True}] * 31
        m = _chain_mock(data=rows)
        with patch("storage.client_admin", return_value=m):
            warnings = storage.admin_sanity_check()
        assert warnings == []

    def test_incomplete_session_flagged(self):
        pid = str(uuid.uuid4())
        rows = [{"participant_id": pid, "completed": False}] * 31
        m = _chain_mock(data=rows)
        with patch("storage.client_admin", return_value=m):
            warnings = storage.admin_sanity_check()
        assert any("completed=False" in w for w in warnings)

    def test_wrong_row_count_flagged(self):
        pid = str(uuid.uuid4())
        rows = [{"participant_id": pid, "completed": True}] * 20
        m = _chain_mock(data=rows)
        with patch("storage.client_admin", return_value=m):
            warnings = storage.admin_sanity_check()
        assert any("unexpected row count" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Security audit: survey path must not expose admin API
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurityBoundary:
    @staticmethod
    def _code_lines(path: Path) -> str:
        """Return survey.py with comment and blank lines stripped."""
        lines = path.read_text().splitlines()
        return "\n".join(
            ln for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        )

    def test_survey_module_does_not_import_client_admin(self):
        survey_path = Path(__file__).resolve().parent.parent / "survey.py"
        if not survey_path.exists():
            pytest.skip("survey.py not yet written")
        code = self._code_lines(survey_path)
        assert "client_admin" not in code, (
            "survey.py must NEVER import or call client_admin()"
        )

    def test_survey_module_does_not_call_admin_functions(self):
        survey_path = Path(__file__).resolve().parent.parent / "survey.py"
        if not survey_path.exists():
            pytest.skip("survey.py not yet written")
        code = self._code_lines(survey_path)
        for fn in ["fetch_all", "export_csv", "admin_set_collection",
                   "admin_log_audit", "admin_generate_tokens",
                   "admin_get_tokens", "admin_health_check", "admin_sanity_check"]:
            assert fn not in code, f"survey.py must not call {fn}()"
