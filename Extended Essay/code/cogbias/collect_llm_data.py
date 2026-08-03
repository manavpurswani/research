"""collect_llm_data.py — Drive LLM responders through cognitive-bias scenarios.

Usage:
    python collect_llm_data.py [--dry-run] [--output PATH]

--dry-run  : No API calls; stub responder returns random choices.
--output   : CSV output path (default: data/llm_responses.csv)

Responders: ~10 models × 5 temperatures = 50
Rows:        50 responders × 30 trials = 1,500
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scenarios import generate_participant_scenarios

load_dotenv(_HERE / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Responder catalogue
# ─────────────────────────────────────────────────────────────────────────────

_MODELS = [
    ("openai",    "gpt-4o"),
    ("openai",    "gpt-4o-mini"),
    ("anthropic", "claude-opus-4-6"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("google",    "gemini-3.5-flash-lite"),
    ("google",    "gemini-3.1-flash-lite"),
    ("together",  "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("together",  "deepseek-ai/DeepSeek-V4-Pro"),
    ("together",  "openai/gpt-oss-120b"),
]

_TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]

# Matches pipeline.py's expected column names (participant_id, reaction_time_ms)
_COLUMNS = [
    "participant_id",
    "responder_seed",
    "model_name",
    "temperature",
    "trial_index",
    "scenario_template_id",
    "bias_type",
    "frame",
    "expected_value_ratio",
    "anchor_value",
    "previous_choice",
    "is_first_trial",
    "reaction_time_ms",
    "choice",
    "invalid_response",
    "attention_check_passed",
    "consent_version",
    "completed",
    "created_at",
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are participating in a decision-making study. You will be presented with 30 scenarios, "
    "each requiring you to choose between two options: A or B.\n\n"
    "Please read the following scenario carefully and select the option you would choose. "
    'Respond with a single letter: "A" or "B". Do not explain your reasoning.'
)


def build_prompt(sc: dict) -> str:
    return (
        f"Scenario {sc['trial_index']} of 30:\n\n"
        f"{sc['problem_text']}\n\n"
        f"Option A: {sc['option_a_text']}\n"
        f"Option B: {sc['option_b_text']}\n\n"
        "Your choice (A or B):"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_response(raw: Optional[str]) -> tuple[Optional[int], bool]:
    """Return (choice, invalid_response). choice: 0=A, 1=B, None=invalid."""
    import re as _re
    if not raw:
        return None, True
    text = raw.strip()
    # Handle "Option A" / "option b" prefixes
    if text.lower().startswith("option "):
        text = text[7:].strip()
    # First pass: first alphabetic character (fast path for well-behaved models)
    for ch in text:
        if ch.isalpha():
            if ch.upper() == "A":
                return 0, False
            if ch.upper() == "B":
                return 1, False
            break  # First alpha is neither A nor B — fall through to last-letter search
    # Second pass: last standalone A or B (handles reasoning-model output like "I'll choose B")
    matches = _re.findall(r'\b[ABab]\b', text)
    if matches:
        last = matches[-1].upper()
        return (0 if last == "A" else 1), False
    return None, True


# ─────────────────────────────────────────────────────────────────────────────
# API callers
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(model: str, prompt: str, temperature: float) -> tuple[str, int]:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=5,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    return resp.choices[0].message.content or "", latency_ms


def _call_anthropic(model: str, prompt: str, temperature: float) -> tuple[str, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=5,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    return resp.content[0].text, latency_ms


def _call_google(model: str, prompt: str, temperature: float) -> tuple[str, int]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    t0 = time.monotonic()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=temperature,
            max_output_tokens=50,
        ),
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    # Thinking models separate thought parts from output parts; gather output only.
    text = resp.text
    if text is None and resp.candidates:
        parts = resp.candidates[0].content.parts or []
        text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
    return text or "", latency_ms


def _call_together(model: str, prompt: str, temperature: float) -> tuple[str, int]:
    from together import Together
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=max(temperature, 0.01),  # some models don't support temp=0.0
        max_tokens=1024,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    return resp.choices[0].message.content or "", latency_ms


def _call_stub(model: str, prompt: str, temperature: float) -> tuple[str, int]:
    return random.choice(["A", "B"]), random.randint(100, 500)


_DISPATCH = {
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
    "google":    _call_google,
    "together":  _call_together,
}

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between retries


def _call_with_retry(
    caller, model: str, prompt: str, temperature: float
) -> tuple[Optional[str], int]:
    for attempt in range(_MAX_RETRIES):
        try:
            return caller(model, prompt, temperature)
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                print(f"    retry {attempt+1}/{_MAX_RETRIES-1} after error: {type(e).__name__}", flush=True)
                time.sleep(_RETRY_DELAY)
            else:
                print(f"    giving up after {_MAX_RETRIES} attempts: {type(e).__name__}: {e}", flush=True)
                return None, 0


def _load_completed_seeds(output_path: Path) -> set[int]:
    """Return seeds of responders already fully written to the CSV."""
    if not output_path.exists():
        return set()
    completed: set[int] = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("completed") == "True":
                try:
                    completed.add(int(row["responder_seed"]))
                except (KeyError, ValueError):
                    pass
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# Collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect(output_path: Path, dry_run: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_seeds = _load_completed_seeds(output_path)
    if completed_seeds:
        print(f"Resuming — {len(completed_seeds)} responders already done, skipping them.\n", flush=True)

    seed_counter = 1000
    total = len(_MODELS) * len(_TEMPERATURES)
    done = len(completed_seeds)

    # Append to existing file if resuming, else write fresh with header
    file_mode = "a" if completed_seeds else "w"
    with open(output_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if file_mode == "w":
            writer.writeheader()

        for provider, model_name in _MODELS:
            caller = _call_stub if dry_run else _DISPATCH[provider]

            for temp in _TEMPERATURES:
                seed = seed_counter
                seed_counter += 1

                if seed in completed_seeds:
                    continue

                responder_id = str(uuid.uuid4())
                scenarios = generate_participant_scenarios(seed)
                rows: list[dict] = []

                print(
                    f"  [{done+1}/{total}] {model_name} temp={temp}"
                    f"  id={responder_id[:8]}…",
                    flush=True,
                )

                for sc in scenarios:
                    prompt = build_prompt(sc)
                    raw, latency_ms = _call_with_retry(caller, model_name, prompt, temp)
                    choice, invalid = parse_response(raw)

                    rows.append({
                        "participant_id":        responder_id,
                        "responder_seed":        seed,
                        "model_name":            model_name,
                        "temperature":           temp,
                        "trial_index":           sc["trial_index"],
                        "scenario_template_id":  sc["scenario_template_id"],
                        "bias_type":             sc["bias_type"],
                        "frame":                 sc["frame"],
                        "expected_value_ratio":  sc["expected_value_ratio"],
                        "anchor_value":          sc["anchor_value"],
                        "previous_choice":       sc["previous_choice"],
                        "is_first_trial":        int(sc["is_first_trial"]),
                        "reaction_time_ms":      latency_ms,
                        "choice":                "" if choice is None else choice,
                        "invalid_response":      invalid,
                        "attention_check_passed": "",
                        "consent_version":       "N/A",
                        "completed":             False,
                        "created_at":            datetime.now(timezone.utc).isoformat(),
                    })

                for row in rows:
                    row["completed"] = True
                writer.writerows(rows)
                f.flush()  # persist each responder immediately
                done += 1

    print(f"\nWrote {done * 30} rows total → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect LLM responses for cognitive-bias study."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stub API calls; validate prompt and schema only.",
    )
    parser.add_argument(
        "--output", default="data/llm_responses.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    mode = "DRY-RUN (stub)" if args.dry_run else "LIVE"
    n_responders = len(_MODELS) * len(_TEMPERATURES)
    print(f"collect_llm_data.py — {mode}")
    print(f"Responders : {len(_MODELS)} models × {len(_TEMPERATURES)} temps = {n_responders}")
    print(f"Trials/resp: 30  |  Total rows: {n_responders * 30}")
    print(f"Output     : {output_path}\n")

    collect(output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
