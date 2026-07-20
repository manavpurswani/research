"""tokens.py — Token utilities for the survey invitation system.

Provides pure functions (no I/O) for token validation and URL formatting.
Database operations (generate, consume, list) live in storage.py.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_valid_uuid(value: str) -> bool:
    """Return True if value is a well-formed UUID string (any version)."""
    if not isinstance(value, str):
        return False
    return bool(_UUID_RE.match(value.strip()))


def format_survey_url(base_url: str, token: str) -> str:
    """Return the survey URL with the token as a query parameter.

    Parameters
    ----------
    base_url : str
        The root URL of the Streamlit app, e.g. "https://myapp.streamlit.app".
        Trailing slashes are stripped.
    token : str
        A UUID string that has already been validated.

    Returns
    -------
    str
        e.g. "https://myapp.streamlit.app?token=<uuid>"
    """
    if not is_valid_uuid(token):
        raise ValueError(f"Invalid token UUID: {token!r}")
    base = base_url.rstrip("/")
    return f"{base}?token={token}"


def extract_token_from_url(url: str) -> Optional[str]:
    """Parse the token query parameter out of a survey URL.

    Returns the token string (lowercased, stripped) if present and valid,
    otherwise None.
    """
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    token_list = params.get("token", [])
    if not token_list:
        return None
    token = token_list[0].strip().lower()
    return token if is_valid_uuid(token) else None


def new_token_id() -> str:
    """Return a new random UUID4 string suitable for use as a token ID."""
    return str(uuid.uuid4())
