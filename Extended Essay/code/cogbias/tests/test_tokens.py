"""test_tokens.py — Unit tests for tokens.py.

Run from the cogbias/ directory:
    pytest tests/test_tokens.py -v
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokens import (
    extract_token_from_url,
    format_survey_url,
    is_valid_uuid,
    new_token_id,
)

_VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"
_BASE_URL = "https://myapp.streamlit.app"


class TestIsValidUuid:
    def test_valid_uuid4(self):
        assert is_valid_uuid(str(uuid.uuid4())) is True

    def test_valid_uuid_known(self):
        assert is_valid_uuid(_VALID_UUID) is True

    def test_uppercase_accepted(self):
        assert is_valid_uuid(_VALID_UUID.upper()) is True

    def test_mixed_case_accepted(self):
        assert is_valid_uuid("550E8400-e29b-41D4-A716-446655440000") is True

    def test_all_zeros_accepted(self):
        assert is_valid_uuid("00000000-0000-0000-0000-000000000000") is True

    def test_empty_string_rejected(self):
        assert is_valid_uuid("") is False

    def test_non_string_rejected(self):
        assert is_valid_uuid(None) is False   # type: ignore
        assert is_valid_uuid(42) is False     # type: ignore

    def test_too_short_rejected(self):
        assert is_valid_uuid("550e8400-e29b-41d4-a716") is False

    def test_no_hyphens_rejected(self):
        assert is_valid_uuid("550e8400e29b41d4a716446655440000") is False

    def test_extra_chars_rejected(self):
        assert is_valid_uuid(_VALID_UUID + "x") is False

    def test_leading_trailing_whitespace_stripped(self):
        assert is_valid_uuid(f"  {_VALID_UUID}  ") is True

    def test_wrong_group_lengths_rejected(self):
        assert is_valid_uuid("550e8400-e29b-41d4-a716-44665544000") is False


class TestFormatSurveyUrl:
    def test_basic_url(self):
        url = format_survey_url(_BASE_URL, _VALID_UUID)
        assert url == f"{_BASE_URL}?token={_VALID_UUID}"

    def test_trailing_slash_stripped(self):
        url = format_survey_url(_BASE_URL + "/", _VALID_UUID)
        assert "?token=" in url
        assert "/?" not in url

    def test_multiple_trailing_slashes(self):
        url = format_survey_url(_BASE_URL + "///", _VALID_UUID)
        assert not url.startswith(_BASE_URL + "/")

    def test_invalid_token_raises(self):
        with pytest.raises(ValueError, match="Invalid token UUID"):
            format_survey_url(_BASE_URL, "not-a-uuid")

    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="Invalid token UUID"):
            format_survey_url(_BASE_URL, "")

    def test_token_appears_once_in_url(self):
        url = format_survey_url(_BASE_URL, _VALID_UUID)
        assert url.count(_VALID_UUID) == 1

    def test_roundtrip_with_extract(self):
        url = format_survey_url(_BASE_URL, _VALID_UUID)
        extracted = extract_token_from_url(url)
        assert extracted == _VALID_UUID.lower()


class TestExtractTokenFromUrl:
    def test_extracts_from_formatted_url(self):
        url = f"{_BASE_URL}?token={_VALID_UUID}"
        assert extract_token_from_url(url) is not None

    def test_returns_lowercase(self):
        url = f"{_BASE_URL}?token={_VALID_UUID.upper()}"
        result = extract_token_from_url(url)
        assert result == _VALID_UUID.lower()

    def test_strips_whitespace(self):
        url = f"{_BASE_URL}?token= {_VALID_UUID} "
        result = extract_token_from_url(url)
        assert result == _VALID_UUID.lower()

    def test_no_token_param_returns_none(self):
        assert extract_token_from_url(_BASE_URL) is None
        assert extract_token_from_url(f"{_BASE_URL}?mode=admin") is None

    def test_invalid_token_value_returns_none(self):
        assert extract_token_from_url(f"{_BASE_URL}?token=notauuid") is None
        assert extract_token_from_url(f"{_BASE_URL}?token=") is None

    def test_extra_params_ignored(self):
        url = f"{_BASE_URL}?foo=bar&token={_VALID_UUID}&baz=qux"
        result = extract_token_from_url(url)
        assert result == _VALID_UUID.lower()


class TestNewTokenId:
    def test_returns_string(self):
        assert isinstance(new_token_id(), str)

    def test_is_valid_uuid(self):
        assert is_valid_uuid(new_token_id())

    def test_unique_on_each_call(self):
        ids = {new_token_id() for _ in range(100)}
        assert len(ids) == 100, "new_token_id() produced duplicate UUIDs"

    def test_uuid4_format(self):
        token = new_token_id()
        parsed = uuid.UUID(token)
        assert parsed.version == 4
