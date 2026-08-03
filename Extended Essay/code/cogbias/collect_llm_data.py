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
    ("anthropic", "claude-haiku-4-5"),
    ("google",    "gemini-1.5-pro"),
    ("google",    "gemini-1.5-flash"),
    ("together",  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    ("together",  "mistralai/Mixtral-8x22B-Instruct-v0.1"),
    ("together",  "Qwen/Qwen2-72B-Instruct"),
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

def parse_response(raw: str) -> tuple[Optional[int], bool]:
    """Return (choice, invalid_response). choice: 0=A, 1=B, None=invalid."""
    text = raw.strip()
    # Handle "Option A" / "option b" prefixes
    if text.lower().startswith("option "):
        text = text[7:].strip()
    for ch in text:
        if ch.isalpha():
            if ch.upper() == "A":
                return 0, False
            if ch.upper() == "B":
                return 1, False
            return None, True
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
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gmodel = genai.GenerativeModel(
        model_name=model,
        system_instruction=_SYSTEM,
    )
    t0 = time.monotonic()
    resp = gmodel.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": 5},
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    return resp.text, latency_ms


def _call_together(model: str, prompt: str, temperature: float) -> tuple[str, int]:
    import openai
    client = openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=max(temperature, 0.01),  # Together doesn't support temp=0.0
        max_tokens=5,
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


# ─────────────────────────────────────────────────────────────────────────────
# Collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect(output_path: Path, dry_run: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_counter = 1000
    total = len(_MODELS) * len(_TEMPERATURES)
    done = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()

        for provider, model_name in _MODELS:
            caller = _call_stub if dry_run else _DISPATCH[provider]

            for temp in _TEMPERATURES:
                responder_id = str(uuid.uuid4())
                seed = seed_counter
                seed_counter += 1

                scenarios = generate_participant_scenarios(seed)
                rows: list[dict] = []

                print(
                    f"  [{done+1}/{total}] {model_name} temp={temp}"
                    f"  id={responder_id[:8]}…",
                    flush=True,
                )

                for sc in scenarios:
                    prompt = build_prompt(sc)
                    raw, latency_ms = caller(model_name, prompt, temp)
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
                done += 1

    print(f"\nWrote {done * 30} rows → {output_path}")


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
