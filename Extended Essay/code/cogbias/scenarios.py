"""
scenarios.py — Scenario templates and per-participant parametric generator.

Thirty binary-choice scenarios are generated per participant from templates in three
cognitive-bias categories (10 each): framing effect (F01–F10), anchoring (A01–A10),
and loss aversion (L01–L10).

`generate_participant_scenarios(seed)` is fully deterministic given the integer seed,
so any participant's exact scenario set can be reproduced. The returned dicts include
`canonical_biased_choice` (what a biased person would choose), which is only used by
the synthetic data generator and is never a feature seen by the ML models.

────────────────────────────────────────────────────────────────────────────────
EV DESIGN SUMMARY
────────────────────────────────────────────────────────────────────────────────
Framing (F01–F10)
  total        = total units at risk (fixed per template)
  prob_save    = probability the risky option saves all units (fixed per template)
  ev_risky     = total × prob_save
  certain_num  = round(ev_ratio_target × ev_risky)    ← varies per participant
  EV(certain)  = certain_num
  EV(risky)    = ev_risky = total × prob_save
  ev_ratio     = certain_num / ev_risky  ≈ ev_ratio_target

Anchoring (A01–A10)
  low_val  = template default × uniform(0.90, 1.10)   ← varies per participant
  high_val = template default × uniform(0.90, 1.10)
  EV(A)    = value of the option displayed as A (no probability)
  EV(B)    = value of the option displayed as B
  ev_ratio = displayed_A_val / displayed_B_val
  anchor_value = sampled from [anchor_min, anchor_max]

Loss aversion (L01–L10)
  certain_amount = sampled from [c_min, c_max]
  prob_win       = fixed per template
  risky_amount   = certain_amount / (prob_win × ev_ratio_target)
  EV(certain)    = certain_amount
  EV(risky)      = prob_win × risky_amount  =  certain_amount / ev_ratio_target
  ev_ratio       = EV(certain) / EV(risky)  ≈ ev_ratio_target

For all bias types, ab_swap=True swaps which option is labelled A and which B on screen,
which inverts ev_ratio (EV_A/EV_B → EV_B/EV_A) and flips canonical_biased_choice.
"""

import math
import random as _random
from typing import Any, Dict, List

from config import BIAS_TYPE, CHOICE, EV_RATIO_RANGE, FRAME

# ─────────────────────────────────────────────────────────────────────────────
# FRAMING TEMPLATES  (10 templates, F01–F10)
# ─────────────────────────────────────────────────────────────────────────────
_FRAMING_TEMPLATES: List[Dict[str, Any]] = [
    {
        "template_id": "F01",
        "label": "Disease Outbreak",
        "total": 600,
        "unit": "people",
        "verb_gain": "saved",
        "verb_loss": "die",
        "prob_save": 1 / 3,
        "context": (
            "A rare infectious disease has broken out in a town. "
            "Without intervention, all {total} residents in the affected district will die. "
            "Two emergency programmes have been proposed."
        ),
    },
    {
        "template_id": "F02",
        "label": "Oil Spill",
        "total": 400,
        "unit": "seabirds",
        "verb_gain": "rescued",
        "verb_loss": "perish",
        "prob_save": 1 / 4,
        "context": (
            "An oil spill has contaminated a coastal nature reserve. "
            "Without a clean-up programme, all {total} seabirds in the area will perish. "
            "Two clean-up strategies have been evaluated."
        ),
    },
    {
        "template_id": "F03",
        "label": "School Fire",
        "total": 500,
        "unit": "students",
        "verb_gain": "evacuated safely",
        "verb_loss": "be trapped",
        "prob_save": 2 / 5,
        "context": (
            "A fire has broken out in a school. "
            "Without an emergency response, all {total} students in the building will be trapped. "
            "Two evacuation strategies are available."
        ),
    },
    {
        "template_id": "F04",
        "label": "Hospital Epidemic",
        "total": 200,
        "unit": "patients",
        "verb_gain": "recover",
        "verb_loss": "die",
        "prob_save": 1 / 2,
        "context": (
            "A fast-spreading virus has infected patients in a hospital ward. "
            "Without a new treatment, all {total} infected patients will die. "
            "Two treatment protocols have been proposed."
        ),
    },
    {
        "template_id": "F05",
        "label": "Earthquake Response",
        "total": 900,
        "unit": "residents",
        "verb_gain": "evacuated safely",
        "verb_loss": "be stranded",
        "prob_save": 1 / 3,
        "context": (
            "A major earthquake has struck a coastal city. "
            "Without emergency action, all {total} residents in the danger zone will be stranded. "
            "Two rescue plans have been prepared."
        ),
    },
    {
        "template_id": "F06",
        "label": "Factory Gas Leak",
        "total": 300,
        "unit": "workers",
        "verb_gain": "reach safety",
        "verb_loss": "be injured",
        "prob_save": 3 / 5,
        "context": (
            "A toxic gas leak has occurred at a manufacturing plant. "
            "Without immediate action, all {total} workers in the affected section will be injured. "
            "Two response plans have been developed."
        ),
    },
    {
        "template_id": "F07",
        "label": "Flood Warning",
        "total": 800,
        "unit": "households",
        "verb_gain": "protected from flooding",
        "verb_loss": "flood",
        "prob_save": 1 / 4,
        "context": (
            "A severe storm is forecast to flood a lowland area. "
            "Without intervention, all {total} households in the area will flood. "
            "Two mitigation schemes have been proposed."
        ),
    },
    {
        "template_id": "F08",
        "label": "Mine Rescue",
        "total": 90,
        "unit": "miners",
        "verb_gain": "rescued alive",
        "verb_loss": "die",
        "prob_save": 2 / 3,
        "context": (
            "A tunnel collapse has trapped workers underground. "
            "Without rescue, all {total} trapped miners will die. "
            "Two rescue strategies have been evaluated."
        ),
    },
    {
        "template_id": "F09",
        "label": "Wildfire Evacuation",
        "total": 120,
        "unit": "hikers",
        "verb_gain": "evacuated safely",
        "verb_loss": "be caught in the fire",
        "prob_save": 1 / 2,
        "context": (
            "A wildfire is approaching a national park where {total} hikers are located. "
            "Without evacuation, all hikers will be caught in the fire. "
            "Two evacuation routes are available."
        ),
    },
    {
        "template_id": "F10",
        "label": "Mountain Rescue",
        "total": 150,
        "unit": "passengers",
        "verb_gain": "safely evacuated",
        "verb_loss": "remain trapped",
        "prob_save": 1 / 3,
        "context": (
            "A coach has broken down in a dangerous mountain area. "
            "Without rescue, all {total} passengers will remain trapped indefinitely. "
            "Two rescue plans have been prepared."
        ),
    },
]


def _resolve_framing(
    tpl: Dict[str, Any],
    ev_ratio_target: float,
    frame: int,
) -> Dict[str, Any]:
    """
    Build a fully resolved framing scenario.

    EV calculation:
      ev_risky    = total × prob_save
      certain_num = round(ev_ratio_target × ev_risky)   [clipped to [1, total-1]]
      ev_ratio    = certain_num / ev_risky

    Biased response (framing effect, Tversky & Kahneman 1981):
      gain frame (frame=0) → prefer the CERTAIN option (risk-averse)
      loss frame (frame=1) → prefer the RISKY option  (risk-seeking)

    After ab_swap, the on-screen A label may point to the risky option instead.
    """
    total = tpl["total"]
    prob_save = tpl["prob_save"]
    unit = tpl["unit"]
    verb_gain = tpl["verb_gain"]
    verb_loss = tpl["verb_loss"]

    ev_risky = total * prob_save
    certain_num = int(round(ev_ratio_target * ev_risky))
    certain_num = max(1, min(certain_num, total - 1))
    ev_ratio_actual = certain_num / ev_risky

    prob_pct = int(round(prob_save * 100))
    fail_pct = 100 - prob_pct

    context_text = tpl["context"].format(total=total)

    if frame == FRAME["gain"]:
        certain_text = f"Exactly {certain_num} {unit} will be {verb_gain}."
        risky_text = (
            f"{prob_pct}% probability that all {total} {unit} will be {verb_gain}, "
            f"and {fail_pct}% probability that no {unit} will be {verb_gain}."
        )
    else:  # loss frame
        lost_certain = total - certain_num
        certain_text = f"Exactly {lost_certain} {unit} will {verb_loss}."
        risky_text = (
            f"{fail_pct}% probability that all {total} {unit} will {verb_loss}, "
            f"and {prob_pct}% probability that no {unit} will {verb_loss}."
        )

    # Option A is always the certain outcome; Option B is always the risky outcome.
    # Biased choice: gain frame → risk-averse → A (0); loss frame → risk-seeking → B (1).
    option_a_text = certain_text
    option_b_text = risky_text
    ev_ratio_displayed = ev_ratio_actual  # EV(certain) / EV(risky)
    canonical_biased_choice = CHOICE["A"] if frame == FRAME["gain"] else CHOICE["B"]

    return {
        "problem_text": context_text,
        "option_a_text": option_a_text,
        "option_b_text": option_b_text,
        "bias_type": BIAS_TYPE["framing"],
        "frame": frame,
        "expected_value_ratio": round(ev_ratio_displayed, 4),
        "anchor_value": 0.0,
        "canonical_biased_choice": canonical_biased_choice,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANCHORING TEMPLATES  (10 templates, A01–A10)
# ─────────────────────────────────────────────────────────────────────────────
_ANCHORING_TEMPLATES: List[Dict[str, Any]] = [
    {
        "template_id": "A01",
        "label": "Product Price",
        "unit": "dollars",
        "unit_symbol": "$",
        "low_default": 42,
        "high_default": 58,
        "anchor_min": 15.0,
        "anchor_max": 85.0,
        "context": (
            "Market research suggests that products in this category typically sell "
            "for around ${anchor:.0f} in this region."
        ),
        "question": "Which price do you believe more accurately reflects the fair market value of this product?",
        "option_template": "The fair market value of this product is ${val:.0f}.",
    },
    {
        "template_id": "A02",
        "label": "City Population",
        "unit": "million people",
        "unit_symbol": "M",
        "low_default": 1.8,
        "high_default": 2.4,
        "anchor_min": 0.5,
        "anchor_max": 5.0,
        "context": (
            "A demographic survey estimated the metropolitan population of a comparable "
            "city in this region to be approximately {anchor:.1f} million people."
        ),
        "question": "Which figure do you believe is the more accurate population estimate for this city?",
        "option_template": "The city population is approximately {val:.1f} million people.",
    },
    {
        "template_id": "A03",
        "label": "Annual Salary",
        "unit": "thousand dollars per year",
        "unit_symbol": "k$/yr",
        "low_default": 52.0,
        "high_default": 68.0,
        "anchor_min": 30.0,
        "anchor_max": 110.0,
        "context": (
            "Industry data suggests that professionals in this field earn "
            "approximately ${anchor:.0f},000 per year at the mid-career level."
        ),
        "question": "Which salary range better represents fair compensation for this role?",
        "option_template": "Fair compensation for this role is ${val:.0f},000 per year.",
    },
    {
        "template_id": "A04",
        "label": "Project Duration",
        "unit": "weeks",
        "unit_symbol": "wks",
        "low_default": 9.0,
        "high_default": 13.0,
        "anchor_min": 3.0,
        "anchor_max": 22.0,
        "context": (
            "A project manager familiar with similar work estimated that "
            "a project of this scope would take about {anchor:.0f} weeks to complete."
        ),
        "question": "Which timeline estimate is more realistic for completing this project?",
        "option_template": "This project will realistically take {val:.0f} weeks to complete.",
    },
    {
        "template_id": "A05",
        "label": "Exam Pass Rate",
        "unit": "percent",
        "unit_symbol": "%",
        "low_default": 60.0,
        "high_default": 74.0,
        "anchor_min": 25.0,
        "anchor_max": 95.0,
        "context": (
            "A national report cited an average pass rate of {anchor:.0f}% "
            "for students in this subject area."
        ),
        "question": "Which pass rate do you think is more achievable for a well-prepared cohort?",
        "option_template": "A well-prepared cohort of students can realistically achieve a {val:.0f}% pass rate.",
    },
    {
        "template_id": "A06",
        "label": "Daily Calorie Target",
        "unit": "kcal",
        "unit_symbol": "kcal",
        "low_default": 1850.0,
        "high_default": 2250.0,
        "anchor_min": 1000.0,
        "anchor_max": 3500.0,
        "context": (
            "A nutrition guide recommended a daily caloric intake of approximately "
            "{anchor:.0f} kcal for adults with a moderate activity level."
        ),
        "question": "Which daily calorie target is more appropriate for a healthy active adult?",
        "option_template": "The appropriate daily calorie target for a healthy active adult is {val:.0f} kcal.",
    },
    {
        "template_id": "A07",
        "label": "Loan Interest Rate",
        "unit": "percent per annum",
        "unit_symbol": "% p.a.",
        "low_default": 4.5,
        "high_default": 7.5,
        "anchor_min": 1.5,
        "anchor_max": 14.0,
        "context": (
            "A financial adviser mentioned that the standard interest rate on "
            "consumer loans in this market is currently around {anchor:.1f}% per year."
        ),
        "question": "Which interest rate better reflects a fair lending rate for this loan?",
        "option_template": "A fair interest rate for this loan is {val:.1f}% per year.",
    },
    {
        "template_id": "A08",
        "label": "Flight Duration",
        "unit": "hours",
        "unit_symbol": "hrs",
        "low_default": 6.0,
        "high_default": 9.0,
        "anchor_min": 2.0,
        "anchor_max": 16.0,
        "context": (
            "A travel guide mentioned that flights to this destination "
            "typically take around {anchor:.0f} hours from this departure city."
        ),
        "question": "Which flight duration is a more accurate estimate for this route?",
        "option_template": "The flight to this destination takes approximately {val:.0f} hours.",
    },
    {
        "template_id": "A09",
        "label": "Charitable Donation",
        "unit": "dollars",
        "unit_symbol": "$",
        "low_default": 35.0,
        "high_default": 65.0,
        "anchor_min": 5.0,
        "anchor_max": 200.0,
        "context": (
            "A fundraising report stated that the average donor to this charity "
            "contributes approximately ${anchor:.0f} per year."
        ),
        "question": "Which donation amount is the more appropriate annual contribution for a regular supporter?",
        "option_template": "An appropriate annual donation for a regular supporter is ${val:.0f}.",
    },
    {
        "template_id": "A10",
        "label": "Years of Experience",
        "unit": "years",
        "unit_symbol": "yrs",
        "low_default": 3.0,
        "high_default": 7.0,
        "anchor_min": 1.0,
        "anchor_max": 15.0,
        "context": (
            "A recruitment specialist noted that candidates with around "
            "{anchor:.0f} years of experience are typical for this type of role."
        ),
        "question": "Which experience level is more appropriate to require for this position?",
        "option_template": "Candidates for this position should have at least {val:.0f} years of experience.",
    },
]


def _resolve_anchoring(
    tpl: Dict[str, Any],
    rng: _random.Random,
    anchor_value: float,
) -> Dict[str, Any]:
    """
    Build a fully resolved anchoring scenario.

    EV calculation:
      low_val  = tpl["low_default"]  × uniform(0.90, 1.10)
      high_val = tpl["high_default"] × uniform(0.90, 1.10)
      EV       = the displayed option value (no probability; options are point estimates)
      ev_ratio = displayed_A_value / displayed_B_value

    Biased response (anchoring effect):
      anchor > midpoint(low_val, high_val) → biased toward the HIGH-value option
      anchor < midpoint                    → biased toward the LOW-value option
    """
    low_val = tpl["low_default"] * rng.uniform(0.90, 1.10)
    high_val = tpl["high_default"] * rng.uniform(0.90, 1.10)
    # Ensure ordering is maintained after jitter
    if low_val >= high_val:
        low_val, high_val = high_val, low_val

    midpoint = (low_val + high_val) / 2.0
    # Option A is always the low-value estimate; Option B is always the high-value estimate.
    # Biased choice: anchor ≥ midpoint → pulled toward high → B (1); else → A (0).
    canonical_biased_choice = CHOICE["B"] if anchor_value >= midpoint else CHOICE["A"]

    context_text = tpl["context"].format(anchor=anchor_value)
    low_option_text = tpl["option_template"].format(val=low_val)
    high_option_text = tpl["option_template"].format(val=high_val)
    question_text = tpl["question"]

    option_a_text = low_option_text
    option_b_text = high_option_text
    ev_ratio_displayed = low_val / high_val  # always < 1

    problem_text = f"{context_text}\n\n{question_text}"

    return {
        "problem_text": problem_text,
        "option_a_text": option_a_text,
        "option_b_text": option_b_text,
        "bias_type": BIAS_TYPE["anchoring"],
        "frame": FRAME["gain"],  # anchoring scenarios have no gain/loss framing
        "expected_value_ratio": round(ev_ratio_displayed, 4),
        "anchor_value": round(anchor_value, 2),
        "canonical_biased_choice": canonical_biased_choice,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSS AVERSION TEMPLATES  (10 templates, L01–L10)
# ─────────────────────────────────────────────────────────────────────────────
_LOSS_AVERSION_TEMPLATES: List[Dict[str, Any]] = [
    {
        "template_id": "L01",
        "label": "Debt Settlement",
        "c_min": 200,
        "c_max": 600,
        "prob_win": 0.50,
        "context_template": (
            "You owe a debt of ${total_owed:,}. "
            "A creditor offers two options to settle it."
        ),
        "certain_template": (
            "Pay a reduced guaranteed settlement of ${pay_certain:,} "
            "(saving you ${certain_amount:,} compared with the full debt)."
        ),
        "risky_template": (
            "{prob_pct}% chance the full debt is forgiven (you save ${risky_amount:,}); "
            "{fail_pct}% chance you must pay the full ${total_owed:,}."
        ),
    },
    {
        "template_id": "L02",
        "label": "Investment Recovery",
        "c_min": 500,
        "c_max": 1500,
        "prob_win": 0.40,
        "context_template": (
            "Your investment portfolio has declined by ${total_loss:,}. "
            "Your broker suggests two recovery strategies."
        ),
        "certain_template": (
            "Guaranteed partial recovery of ${certain_amount:,} "
            "(reducing your net loss to ${remaining:,})."
        ),
        "risky_template": (
            "{prob_pct}% chance of fully recovering ${risky_amount:,}; "
            "{fail_pct}% chance of recovering nothing."
        ),
    },
    {
        "template_id": "L03",
        "label": "Insurance Claim",
        "c_min": 300,
        "c_max": 900,
        "prob_win": 0.60,
        "context_template": (
            "You have filed an insurance claim for ${total_damage:,} in damages. "
            "The insurer presents two settlement options."
        ),
        "certain_template": (
            "Accept a guaranteed payout of ${certain_amount:,} now."
        ),
        "risky_template": (
            "{prob_pct}% chance of receiving the full ${risky_amount:,} payout; "
            "{fail_pct}% chance of receiving nothing."
        ),
    },
    {
        "template_id": "L04",
        "label": "Salary Negotiation",
        "c_min": 2000,
        "c_max": 8000,
        "prob_win": 0.50,
        "context_template": (
            "You are negotiating a pay rise of up to ${total_sought:,}. "
            "Your employer offers two options."
        ),
        "certain_template": (
            "Accept a guaranteed raise of ${certain_amount:,} per year."
        ),
        "risky_template": (
            "{prob_pct}% chance of receiving the full ${risky_amount:,} raise; "
            "{fail_pct}% chance of receiving no raise at all."
        ),
    },
    {
        "template_id": "L05",
        "label": "Late-Payment Penalty",
        "c_min": 100,
        "c_max": 400,
        "prob_win": 0.50,
        "context_template": (
            "You have incurred a late-payment penalty of ${total_penalty:,}. "
            "The organisation offers two ways to resolve it."
        ),
        "certain_template": (
            "Pay a guaranteed reduced penalty of ${pay_certain:,} "
            "(saving you ${certain_amount:,} off the total)."
        ),
        "risky_template": (
            "{prob_pct}% chance the entire penalty of ${risky_amount:,} is waived; "
            "{fail_pct}% chance you pay the full ${total_penalty:,}."
        ),
    },
    {
        "template_id": "L06",
        "label": "Scholarship Award",
        "c_min": 1000,
        "c_max": 4000,
        "prob_win": 0.40,
        "context_template": (
            "You have applied for a scholarship worth up to ${total_scholarship:,}. "
            "The committee presents two award options."
        ),
        "certain_template": (
            "Receive a guaranteed partial award of ${certain_amount:,}."
        ),
        "risky_template": (
            "{prob_pct}% chance of winning the full ${risky_amount:,} scholarship; "
            "{fail_pct}% chance of receiving nothing."
        ),
    },
    {
        "template_id": "L07",
        "label": "Business Contract",
        "c_min": 5000,
        "c_max": 15000,
        "prob_win": 0.50,
        "context_template": (
            "Your company is bidding for a contract worth up to ${total_contract:,}. "
            "The client offers two terms."
        ),
        "certain_template": (
            "Sign a guaranteed contract for ${certain_amount:,}."
        ),
        "risky_template": (
            "{prob_pct}% chance of winning the full ${risky_amount:,} contract; "
            "{fail_pct}% chance of losing the bid entirely."
        ),
    },
    {
        "template_id": "L08",
        "label": "Exam Mark Recovery",
        "c_min": 8,
        "c_max": 20,
        "prob_win": 0.50,
        "context_template": (
            "You lost {total_marks} marks on an examination due to a disputed question. "
            "The examiner offers two forms of appeal."
        ),
        "certain_template": (
            "Guaranteed reinstatement of {certain_amount} marks."
        ),
        "risky_template": (
            "{prob_pct}% chance all {risky_amount} marks are reinstated; "
            "{fail_pct}% chance no marks are reinstated."
        ),
    },
    {
        "template_id": "L09",
        "label": "Environmental Fine",
        "c_min": 3000,
        "c_max": 10000,
        "prob_win": 0.40,
        "context_template": (
            "Your organisation faces a regulatory fine of ${total_fine:,} for an environmental breach. "
            "The regulator offers two resolution paths."
        ),
        "certain_template": (
            "Accept a guaranteed fine reduction of ${certain_amount:,} "
            "(net fine: ${remaining:,})."
        ),
        "risky_template": (
            "{prob_pct}% chance the fine is completely waived (saving ${risky_amount:,}); "
            "{fail_pct}% chance you pay the full ${total_fine:,}."
        ),
    },
    {
        "template_id": "L10",
        "label": "Performance Bonus",
        "c_min": 500,
        "c_max": 2000,
        "prob_win": 0.50,
        "context_template": (
            "Your team is eligible for a performance bonus of up to ${total_bonus:,}. "
            "Management offers two payout structures."
        ),
        "certain_template": (
            "Receive a guaranteed bonus of ${certain_amount:,} now."
        ),
        "risky_template": (
            "{prob_pct}% chance of receiving the full ${risky_amount:,} bonus; "
            "{fail_pct}% chance of receiving no bonus."
        ),
    },
]


def _resolve_loss_aversion(
    tpl: Dict[str, Any],
    rng: _random.Random,
    ev_ratio_target: float,
    ab_swap: bool = False,
) -> Dict[str, Any]:
    """
    Build a fully resolved loss-aversion scenario.

    EV calculation:
      certain_amount = sampled from [c_min, c_max] (rounded to a round number)
      prob_win       = fixed per template
      risky_amount   = round(certain_amount / (prob_win × ev_ratio_target))
      EV(certain)    = certain_amount
      EV(risky)      = prob_win × risky_amount  ≈  certain_amount / ev_ratio_target
      ev_ratio       = certain_amount / (prob_win × risky_amount)  ≈  ev_ratio_target

    Biased response (loss aversion, Kahneman 2011 Prospect Theory):
      Participants prefer the CERTAIN option (framed as avoiding a loss)
      over the risky option even when EVs are similar.
    """
    c_min, c_max = tpl["c_min"], tpl["c_max"]
    prob_win = tpl["prob_win"]

    # Sample certain_amount and round to a "clean" number for readability
    certain_amount = rng.randint(c_min, c_max)
    # Round to nearest 50 if large, nearest 5 if small
    rounder = 50 if certain_amount >= 500 else 5
    certain_amount = round(certain_amount / rounder) * rounder
    certain_amount = max(c_min, min(certain_amount, c_max))

    risky_amount = round(certain_amount / (prob_win * ev_ratio_target))
    # Ensure risky_amount > certain_amount (risky option offers higher potential gain)
    risky_amount = max(risky_amount, certain_amount + 1)

    ev_ratio_actual = certain_amount / (prob_win * risky_amount)

    prob_pct = int(round(prob_win * 100))
    fail_pct = 100 - prob_pct

    # Build context-specific variables for text formatting
    multiplier = rng.uniform(2.5, 4.0)
    total_large = round(certain_amount * multiplier / rounder) * rounder

    params: Dict[str, Any] = {
        "certain_amount": certain_amount,
        "risky_amount": risky_amount,
        "prob_pct": prob_pct,
        "fail_pct": fail_pct,
        # Generic large-total fields used by various templates
        "total_owed": total_large,
        "total_loss": total_large,
        "total_damage": total_large,
        "total_sought": total_large,
        "total_penalty": total_large,
        "total_scholarship": total_large,
        "total_contract": total_large,
        "total_marks": certain_amount + rng.randint(1, 5),
        "total_fine": total_large,
        "total_bonus": total_large,
        # Derived fields
        "pay_certain": total_large - certain_amount,
        "remaining": total_large - certain_amount,
        "shortfall": total_large - certain_amount,
    }

    context_text = tpl["context_template"].format(**params)
    certain_text = tpl["certain_template"].format(**params)
    risky_text = tpl["risky_template"].format(**params)

    # ab_swap is retained for loss-aversion so that the class distribution stays balanced.
    # The ML model uses ev_ratio sign to determine which option is certain:
    #   ev_ratio < 1 → A is certain → loss-averse choice is A (0)
    #   ev_ratio > 1 → B is certain → loss-averse choice is B (1)
    canonical_biased_pre_swap = CHOICE["A"]

    if not ab_swap:
        option_a_text = certain_text
        option_b_text = risky_text
        ev_ratio_displayed = ev_ratio_actual  # EV(certain)/EV(risky) < 1 by design
        canonical_biased_choice = canonical_biased_pre_swap
    else:
        option_a_text = risky_text
        option_b_text = certain_text
        ev_ratio_displayed = 1.0 / ev_ratio_actual  # inverted → > 1
        canonical_biased_choice = 1 - canonical_biased_pre_swap  # prefer B (certain)

    return {
        "problem_text": context_text,
        "option_a_text": option_a_text,
        "option_b_text": option_b_text,
        "bias_type": BIAS_TYPE["loss_aversion"],
        "frame": FRAME["loss"],  # certain option is always loss-framed (avoidance)
        "expected_value_ratio": round(ev_ratio_displayed, 4),
        "anchor_value": 0.0,
        "canonical_biased_choice": canonical_biased_choice,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_participant_scenarios(seed: int) -> List[Dict[str, Any]]:
    """
    Return a list of 30 fully resolved scenario dicts for one participant.

    All numeric parameters are deterministically sampled from `seed`, so calling
    this function twice with the same seed returns identical results.

    Each dict contains:
      Static features  : trial_index, bias_type, frame, expected_value_ratio,
                         anchor_value, is_first_trial
      Display fields   : problem_text, option_a_text, option_b_text
      Metadata         : scenario_template_id, canonical_biased_choice
      Placeholders     : previous_choice=0, reaction_time_ms=0, choice=None
                         (filled in by the survey or the synthetic-data generator)

    NOTE: `trial_index` reflects the randomised presentation order, not the
    template order. `canonical_biased_choice` is for synthetic data only and
    must NOT be used as a model feature.
    """
    rng = _random.Random(seed)
    resolved: List[Dict[str, Any]] = []

    # ── Framing scenarios ────────────────────────────────────────────────────
    # Counterbalance: exactly 5 gain frames and 5 loss frames, in random order.
    _frames = [FRAME["gain"]] * 5 + [FRAME["loss"]] * 5
    rng.shuffle(_frames)
    for i, tpl in enumerate(_FRAMING_TEMPLATES):
        ev_ratio = rng.uniform(*EV_RATIO_RANGE)
        frame = _frames[i]
        sc = _resolve_framing(tpl, ev_ratio, frame)
        sc["scenario_template_id"] = tpl["template_id"]
        resolved.append(sc)

    # ── Anchoring scenarios ──────────────────────────────────────────────────
    for tpl in _ANCHORING_TEMPLATES:
        anchor = rng.uniform(tpl["anchor_min"], tpl["anchor_max"])
        sc = _resolve_anchoring(tpl, rng, anchor)
        sc["scenario_template_id"] = tpl["template_id"]
        resolved.append(sc)

    # ── Loss-aversion scenarios ──────────────────────────────────────────────
    for tpl in _LOSS_AVERSION_TEMPLATES:
        ev_ratio = rng.uniform(*EV_RATIO_RANGE)
        ab_swap = bool(rng.randint(0, 1))
        sc = _resolve_loss_aversion(tpl, rng, ev_ratio, ab_swap)
        sc["scenario_template_id"] = tpl["template_id"]
        resolved.append(sc)

    # ── Randomise trial order ────────────────────────────────────────────────
    rng.shuffle(resolved)

    # ── Assign trial_index and placeholder dynamic fields ────────────────────
    for idx, sc in enumerate(resolved):
        sc["trial_index"] = idx + 1
        sc["is_first_trial"] = int(idx == 0)
        # previous_choice and reaction_time_ms are filled in during the survey/synthesis
        sc["previous_choice"] = 0
        sc["reaction_time_ms"] = 0
        sc["choice"] = None  # target; filled in when the participant responds

    return resolved
