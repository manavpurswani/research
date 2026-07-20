"""config.py — Global constants, feature specifications, and reproducibility settings.

All numeric constants referenced across the platform are defined here.
Update this file to change experimental parameters without modifying other modules.
"""

# ── Reproducibility ────────────────────────────────────────────────────────────
GLOBAL_SEED: int = 42  # propagated to numpy.random, Python random, and sklearn random_state

# ── Scenario counts ───────────────────────────────────────────────────────────
N_SCENARIOS_PER_BIAS: int = 10
N_SCENARIOS_TOTAL: int = 30  # = 3 × N_SCENARIOS_PER_BIAS
N_FRAMING_GAIN: int = 5       # exactly 5 of the 10 framing trials must be gain-framed
N_FRAMING_LOSS: int = 5       # exactly 5 must be loss-framed (counterbalancing)

# ── Bias-type encodings ────────────────────────────────────────────────────────
BIAS_TYPE: dict = {
    "framing": 1,
    "anchoring": 2,
    "loss_aversion": 3,
}
BIAS_TYPE_LABEL: dict = {v: k for k, v in BIAS_TYPE.items()}

# ── Frame encodings ───────────────────────────────────────────────────────────
FRAME: dict = {
    "gain": 0,
    "loss": 1,
}

# ── Choice encodings ──────────────────────────────────────────────────────────
CHOICE: dict = {
    "A": 0,
    "B": 1,
}

# ── Feature columns fed to the ML models ─────────────────────────────────────
FEATURE_COLS: list = [
    "trial_index",           # int  1–30: position in this participant's randomised order
    "bias_type",             # int  1=framing, 2=anchoring, 3=loss_aversion
    "frame",                 # int  0=gain, 1=loss
    "expected_value_ratio",  # float EV(on-screen A) / EV(on-screen B)
    "anchor_value",          # float anchor shown; 0.0 for non-anchoring trials
    "previous_choice",       # int  0/1: choice on immediately prior trial
    "is_first_trial",        # int  1 if trial 1 (previous_choice undefined), else 0
    "reaction_time_ms",      # int  ms from scenario display to button click
]
TARGET_COL: str = "choice"  # int 0=A, 1=B

# ── Metadata columns (stored but not fed to models) ──────────────────────────
METADATA_COLS: list = [
    "participant_id",        # UUID string
    "participant_seed",      # int: seed used to generate this participant's scenarios
    "scenario_template_id",  # str: e.g. "F01", "A03", "L07"
    "created_at",            # ISO 8601 timestamp
]

# ── EV-ratio sampling range (framing + loss aversion scenarios) ───────────────
EV_RATIO_RANGE: tuple = (0.85, 1.15)

# ── Cross-validation settings ─────────────────────────────────────────────────
CV_K: int = 5           # GroupKFold splits
CV_K_FALLBACK: int = 3  # fallback when participants < CV_K
MIN_PARTICIPANTS_FOR_CV: int = CV_K_FALLBACK

# ── sklearn MLPClassifier hyper-parameters ────────────────────────────────────
# Spec: MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam',
#        max_iter=200, early_stopping=True, validation_fraction=0.2, random_state=SEED)
NN_HIDDEN_LAYER_SIZES: tuple = (16,)
NN_ACTIVATION: str = "relu"
NN_SOLVER: str = "adam"
NN_MAX_ITER: int = 200
NN_EARLY_STOPPING: bool = True
NN_VALIDATION_FRACTION: float = 0.20

# ── Data quality thresholds ───────────────────────────────────────────────────
RT_TOO_FAST_MS: int = 200         # per-trial: below this is flagged rt_too_fast
RT_TOO_SLOW_MS: int = 60_000      # per-trial: above this is flagged rt_too_slow
SESSION_MIN_SECS: float = 90.0    # total session < 90 s → flagged session_too_fast

# ── Supabase table names ──────────────────────────────────────────────────────
SUPABASE_TABLE: str = "responses"
SUPABASE_TOKENS_TABLE: str = "tokens"
SUPABASE_CONFIG_TABLE: str = "config"
SUPABASE_AUDIT_TABLE: str = "admin_audit"

# ── Consent version (bump when consent text changes) ─────────────────────────
CONSENT_VERSION: str = "v1"
