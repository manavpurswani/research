"""config.py — Global constants, feature specifications, and reproducibility settings.

All numeric constants referenced across the platform are defined here.
Update this file to change experimental parameters without modifying other modules.
"""

# ── Reproducibility ────────────────────────────────────────────────────────────
GLOBAL_SEED: int = 42  # seeds numpy, Python random, and TensorFlow

# ── Scenario counts ───────────────────────────────────────────────────────────
N_SCENARIOS_PER_BIAS: int = 10
N_SCENARIOS_TOTAL: int = 30  # = 3 × N_SCENARIOS_PER_BIAS

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
    "anchor_value",          # float anchor shown to participant; 0 for non-anchoring trials
    "previous_choice",       # int  0/1: participant's choice on the immediately prior trial
    "is_first_trial",        # int  1 if this is trial 1 (previous_choice is undefined), else 0
    "reaction_time_ms",      # int  ms from scenario display to button click
]
TARGET_COL: str = "choice"  # int 0=A, 1=B

# ── Metadata columns (stored but not fed to models) ──────────────────────────
METADATA_COLS: list = [
    "participant_id",        # UUID string
    "participant_seed",      # int: seed used to generate this participant's scenario set
    "scenario_template_id",  # str: e.g. "F01", "A03", "L07"
    "created_at",            # ISO 8601 timestamp
]

# ── EV-ratio sampling range (framing + loss aversion scenarios) ───────────────
EV_RATIO_RANGE: tuple = (0.85, 1.15)

# ── Cross-validation settings ─────────────────────────────────────────────────
CV_K: int = 5          # folds for GroupKFold
CV_K_FALLBACK: int = 3  # fall back when fewer than 5 participants
MIN_PARTICIPANTS_FOR_CV: int = CV_K_FALLBACK

# ── Neural-network hyper-parameters ───────────────────────────────────────────
NN_HIDDEN_NEURONS: int = 16
NN_ACTIVATION_HIDDEN: str = "relu"
NN_ACTIVATION_OUTPUT: str = "sigmoid"
NN_LOSS: str = "binary_crossentropy"
NN_OPTIMIZER: str = "adam"
NN_MAX_EPOCHS: int = 50
NN_BATCH_SIZE: int = 32
NN_EARLY_STOPPING_PATIENCE: int = 5
NN_VALIDATION_SPLIT: float = 0.20  # fraction of each training fold held out for early stopping

# ── Supabase table name ────────────────────────────────────────────────────────
SUPABASE_TABLE: str = "responses"
