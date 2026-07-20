"""verify_setup.py — Pre-launch readiness check.

Exits 0 only when every check passes. Run before opening collection:

    python verify_setup.py

Checks
------
  1. Python version >= 3.11
  2. Required packages importable (streamlit, supabase, sklearn, pandas, numpy, matplotlib)
  3. .streamlit/secrets.toml present and contains required keys
  4. Supabase connectivity: is_collection_open RPC reachable (read-only)
  5. Pre-commit hook installed
  6. No TensorFlow installed (spec requires scikit-learn only)
  7. scenarios.py produces exactly 30 scenarios with correct counterbalance
  8. pipeline.py imports without error
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

results: List[Tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    line = f"  [{tag}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    results.append((ok, label))
    return ok


def warn(label: str, detail: str = "") -> None:
    line = f"  [{WARN}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


# ── 1. Python version ─────────────────────────────────────────────────────────
print("\n── 1. Python version ───────────────────────────────────────────────────")
v = sys.version_info
check(v >= (3, 11), "Python >= 3.11", f"Found {v.major}.{v.minor}.{v.micro}")

# ── 2. Required packages ──────────────────────────────────────────────────────
print("\n── 2. Required packages ────────────────────────────────────────────────")
REQUIRED = [
    ("streamlit",  "streamlit"),
    ("supabase",   "supabase"),
    ("sklearn",    "scikit-learn"),
    ("pandas",     "pandas"),
    ("numpy",      "numpy"),
    ("matplotlib", "matplotlib"),
    ("pytest",     "pytest"),
]
for mod, pkg in REQUIRED:
    try:
        importlib.import_module(mod)
        check(True, f"{pkg} importable")
    except ImportError:
        check(False, f"{pkg} importable", f"Run: pip install {pkg}")

try:
    importlib.import_module("tensorflow")
    check(False, "TensorFlow absent", "TensorFlow is installed — remove it (spec: scikit-learn only)")
except ImportError:
    check(True, "TensorFlow absent")

# ── 3. Streamlit secrets ──────────────────────────────────────────────────────
print("\n── 3. Streamlit secrets ────────────────────────────────────────────────")
secrets_path = _HERE / ".streamlit" / "secrets.toml"
secrets: dict = {}
tomllib = None  # type: ignore

if not secrets_path.exists():
    check(False, "secrets.toml present", str(secrets_path))
else:
    check(True, "secrets.toml present")
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            pass

    if tomllib:
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        for key in ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY", "ADMIN_PASSWORD"]:
            present = key in secrets and bool(secrets[key])
            check(present, f"secrets.toml has {key}")
        if "ADMIN_PASSWORD" in secrets:
            pw = secrets["ADMIN_PASSWORD"]
            check(len(pw) >= 16, "ADMIN_PASSWORD length >= 16 chars",
                  f"Length is {len(pw)}")
    else:
        warn("Cannot parse secrets.toml (tomllib unavailable) — skipping key checks")

# ── 4. Supabase connectivity ──────────────────────────────────────────────────
print("\n── 4. Supabase connectivity ─────────────────────────────────────────────")
url = secrets.get("SUPABASE_URL", "")
anon_key = secrets.get("SUPABASE_ANON_KEY", "")
if url and anon_key:
    try:
        from supabase import create_client
        client = create_client(url, anon_key)
        result = client.rpc("is_collection_open", {}).execute()
        check(True, "Supabase RPC reachable (is_collection_open)")
        if result.data:
            warn("collection_open is currently TRUE — confirm this is intentional pre-launch")
        else:
            check(True, "collection_open = false (correct pre-launch state)")
    except Exception as exc:
        check(False, "Supabase RPC reachable", str(exc))
else:
    warn("Supabase credentials missing — skipping connectivity check")

# ── 5. Pre-commit hook ────────────────────────────────────────────────────────
print("\n── 5. Pre-commit hook ──────────────────────────────────────────────────")
precommit_cfg = _HERE / ".pre-commit-config.yaml"
git_hook = _HERE / ".git" / "hooks" / "pre-commit"
check(precommit_cfg.exists(), ".pre-commit-config.yaml present")
if git_hook.exists():
    check(True, "pre-commit hook installed")
else:
    check(False, "pre-commit hook installed", "Run: pre-commit install")

# ── 6. Scenario integrity ─────────────────────────────────────────────────────
print("\n── 6. Scenario integrity ────────────────────────────────────────────────")
try:
    sys.path.insert(0, str(_HERE))
    from scenarios import generate_participant_scenarios
    sc = generate_participant_scenarios(42)
    check(len(sc) == 30, "generate_participant_scenarios(42) returns 30 scenarios",
          f"Got {len(sc)}")
    bias_counts: dict = {}
    for s in sc:
        bias_counts[s["bias_type"]] = bias_counts.get(s["bias_type"], 0) + 1
    check(all(v == 10 for v in bias_counts.values()),
          "Exactly 10 scenarios per bias type", str(bias_counts))
    framing = [s for s in sc if s["bias_type"] == 1]
    gain = sum(1 for s in framing if s["frame"] == 0)
    loss = sum(1 for s in framing if s["frame"] == 1)
    check(gain == 5 and loss == 5, "Framing: exactly 5 gain + 5 loss frames",
          f"gain={gain}, loss={loss}")
except Exception as exc:
    check(False, "scenarios.py importable and correct", str(exc))

# ── 7. Pipeline importable ────────────────────────────────────────────────────
print("\n── 7. Pipeline importable ───────────────────────────────────────────────")
try:
    from pipeline import run_full, flag_quality, load_and_prepare, wilson_ci  # noqa: F401
    check(True, "pipeline.py imports cleanly")
except Exception as exc:
    check(False, "pipeline.py imports cleanly", str(exc))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ─────────────────────────────────────────────────────────────")
n_pass = sum(1 for ok, _ in results if ok)
n_fail = sum(1 for ok, _ in results if not ok)
print(f"  {n_pass} passed, {n_fail} failed")

if n_fail:
    print(f"\n  {FAIL} Fix the issues above before launching.\n")
    sys.exit(1)
else:
    print(f"\n  {PASS} All checks passed. Ready to launch.\n")
    sys.exit(0)
