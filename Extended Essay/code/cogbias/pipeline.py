"""
pipeline.py — Standalone ML evaluation pipeline for cognitive-bias prediction.

Can be imported by admin.py  OR  run directly:
    python pipeline.py path/to/responses.csv [output_dir]

When run as a script it prints all metric tables and saves four figures:
    confusion_matrix_lr.png
    confusion_matrix_nn.png
    accuracy_by_bias_type.png
    lr_coefficients.png

When imported, call `run_pipeline(df)` which returns a dict of results
including matplotlib Figure objects ready for st.pyplot().
────────────────────────────────────────────────────────────────────────────────
Evaluation design
  • Participant-grouped k-fold CV (GroupKFold, k=5; fallback k=3 for < 5 participants).
  • Every model is always tested on participants it has never seen during training.
  • Reported accuracy is mean ± std across folds.
  • Three baselines compared against model accuracy:
      1. Chance (0.50)
      2. Majority-class (globally most frequent choice in the training fold)
      3. Per-condition majority-vote (modal choice per bias_type in training fold)
  • Bias-stratified evaluation: metrics computed separately for framing, anchoring,
    and loss aversion within the same CV framework.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Suppress TF info/warning logs before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import logging as _logging
_logging.getLogger("tensorflow").setLevel(_logging.ERROR)
import tensorflow as tf

# Allow running as a script from any working directory
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (
    GLOBAL_SEED, CV_K, CV_K_FALLBACK, MIN_PARTICIPANTS_FOR_CV,
    FEATURE_COLS, TARGET_COL, BIAS_TYPE_LABEL,
    NN_HIDDEN_NEURONS, NN_ACTIVATION_HIDDEN, NN_ACTIVATION_OUTPUT,
    NN_LOSS, NN_OPTIMIZER, NN_MAX_EPOCHS, NN_BATCH_SIZE,
    NN_EARLY_STOPPING_PATIENCE, NN_VALIDATION_SPLIT,
)

# ── Global seeds ──────────────────────────────────────────────────────────────
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Feature split for the preprocessing pipeline ─────────────────────────────
# bias_type is one-hot encoded; all other features are scaled to zero mean / unit variance
_CATEGORICAL_FEATURES = ["bias_type"]
_NUMERIC_FEATURES = [f for f in FEATURE_COLS if f not in _CATEGORICAL_FEATURES]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Validate and encode the dataframe.

    Returns
    -------
    X           : float32 feature matrix (n_samples, n_features)
    y           : int array of binary labels (n_samples,)
    groups      : participant_id array for GroupKFold (n_samples,)
    feature_names : list of feature names matching columns of X
    """
    required = set(FEATURE_COLS + [TARGET_COL, "participant_id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    for col in _NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[FEATURE_COLS + [TARGET_COL]].isnull().any().any():
        raise ValueError("Dataset contains NaN values in required columns after coercion.")

    y = df[TARGET_COL].to_numpy(dtype=int)
    groups = df["participant_id"].to_numpy()
    X_raw = df[FEATURE_COLS]

    preprocessor = _build_preprocessor()
    X = preprocessor.fit_transform(X_raw).astype(np.float32)
    feature_names = _get_feature_names(preprocessor)

    return X, y, groups, feature_names


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_FEATURES),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
             _CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _get_feature_names(ct: ColumnTransformer) -> List[str]:
    """Return ordered feature names after ColumnTransformer."""
    num_names = _NUMERIC_FEATURES
    cat_names = ct.named_transformers_["cat"].get_feature_names_out(_CATEGORICAL_FEATURES).tolist()
    return num_names + cat_names


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_lr() -> LogisticRegression:
    """Logistic regression with L2 regularisation (sklearn default C=1.0)."""
    return LogisticRegression(
        max_iter=1000,
        random_state=GLOBAL_SEED,
        solver="lbfgs",
    )


def _build_nn(input_dim: int) -> tf.keras.Model:
    """
    Compact feedforward neural network:
      Input → Dense(16, ReLU) → Dense(1, sigmoid)

    Architecture is intentionally small to reduce overfitting on a modest dataset,
    while still providing the non-linear capacity that logistic regression lacks.
    """
    tf.random.set_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(
            NN_HIDDEN_NEURONS,
            activation=NN_ACTIVATION_HIDDEN,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
        ),
        tf.keras.layers.Dense(
            1,
            activation=NN_ACTIVATION_OUTPUT,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=GLOBAL_SEED),
        ),
    ])
    model.compile(optimizer=NN_OPTIMIZER, loss=NN_LOSS, metrics=["accuracy"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return accuracy, precision, recall, F1 (binary, pos_label=1)."""
    if len(np.unique(y_true)) < 2:
        # Only one class in this slice — accuracy is meaningful, others are undefined
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc, "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan")}
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline computation (within the CV framework)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fold_baselines(
    y_train: np.ndarray,
    bias_train: np.ndarray,
    y_test: np.ndarray,
    bias_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute the three baselines for a single fold.

    Baselines are derived from the TRAINING fold only (no leakage from test).

    1. Chance               : always predict 0.5 → accuracy = 0.50
    2. Majority-class       : always predict the most common class in y_train
    3. Per-condition mv     : for each bias_type, always predict the modal class
                              in the training instances of that bias_type
    """
    # Majority-class baseline
    majority_class = int(np.bincount(y_train).argmax())
    y_pred_majority = np.full(len(y_test), majority_class)
    majority_acc = accuracy_score(y_test, y_pred_majority)

    # Per-condition majority-vote baseline
    per_cond_preds = np.full(len(y_test), majority_class)  # fallback
    per_cond_by_bias: Dict[int, float] = {}
    for bt in np.unique(bias_train):
        train_mask = bias_train == bt
        if train_mask.sum() == 0:
            continue
        modal = int(np.bincount(y_train[train_mask]).argmax())
        test_mask = bias_test == bt
        per_cond_preds[test_mask] = modal
        if test_mask.sum() > 0:
            per_cond_by_bias[int(bt)] = accuracy_score(y_test[test_mask],
                                                        per_cond_preds[test_mask])

    per_cond_acc = accuracy_score(y_test, per_cond_preds)

    return {
        "chance": 0.50,
        "majority_class_acc": majority_acc,
        "per_cond_acc": per_cond_acc,
        "per_cond_by_bias": per_cond_by_bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    bias_type_col: np.ndarray,
    model_kind: str,
    feature_names: List[str],
    k: int,
) -> Dict[str, Any]:
    """
    Run GroupKFold cross-validation for either 'lr' or 'nn'.

    Returns a dict with per-fold metrics aggregated into mean ± std.
    """
    gkf = GroupKFold(n_splits=k)
    unique_bias_types = sorted(np.unique(bias_type_col).tolist())

    # Storage per fold
    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
    agg_cm = np.zeros((2, 2), dtype=int)
    by_bias: Dict[int, Dict[str, List[float]]] = {
        bt: {"acc": [], "prec": [], "rec": [], "f1": []} for bt in unique_bias_types
    }
    # LR-only
    coeff_list: List[np.ndarray] = []

    # Baseline storage across folds
    bl_maj_acc, bl_pc_acc = [], []
    bl_pc_by_bias: Dict[int, List[float]] = {bt: [] for bt in unique_bias_types}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        bt_tr = bias_type_col[train_idx]
        bt_te = bias_type_col[test_idx]

        if model_kind == "lr":
            preprocessor = _build_preprocessor()
            X_tr_p = preprocessor.fit_transform(
                _raw_from_indices(X, feature_names, train_idx)
            ).astype(np.float32)
            X_te_p = preprocessor.transform(
                _raw_from_indices(X, feature_names, test_idx)
            ).astype(np.float32)
            clf = _build_lr()
            clf.fit(X_tr_p, y_tr)
            y_pred = clf.predict(X_te_p)
            coeff_list.append(clf.coef_[0])
        else:  # nn
            preprocessor = _build_preprocessor()
            X_tr_p = preprocessor.fit_transform(
                _raw_from_indices(X, feature_names, train_idx)
            ).astype(np.float32)
            X_te_p = preprocessor.transform(
                _raw_from_indices(X, feature_names, test_idx)
            ).astype(np.float32)

            # Reserve 20% of training partition for early-stopping validation
            n_val = max(1, int(len(X_tr_p) * NN_VALIDATION_SPLIT))
            X_tr_m, X_val = X_tr_p[:-n_val], X_tr_p[-n_val:]
            y_tr_m, y_val = y_tr[:-n_val], y_tr[-n_val:]

            nn = _build_nn(X_tr_p.shape[1])
            es = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=NN_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
            nn.fit(
                X_tr_m, y_tr_m,
                validation_data=(X_val, y_val),
                epochs=NN_MAX_EPOCHS,
                batch_size=NN_BATCH_SIZE,
                callbacks=[es],
                verbose=0,
            )
            y_prob = nn.predict(X_te_p, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)

        # ── Overall fold metrics ─────────────────────────────────────────────
        m = _clf_metrics(y_te, y_pred)
        fold_acc.append(m["accuracy"])
        fold_prec.append(m["precision"])
        fold_rec.append(m["recall"])
        fold_f1.append(m["f1"])
        agg_cm += confusion_matrix(y_te, y_pred, labels=[0, 1])

        # ── Per-bias-type metrics ────────────────────────────────────────────
        for bt in unique_bias_types:
            mask = bt_te == bt
            if mask.sum() == 0:
                continue
            bm = _clf_metrics(y_te[mask], y_pred[mask])
            by_bias[bt]["acc"].append(bm["accuracy"])
            by_bias[bt]["prec"].append(bm["precision"])
            by_bias[bt]["rec"].append(bm["recall"])
            by_bias[bt]["f1"].append(bm["f1"])

        # ── Baselines for this fold ──────────────────────────────────────────
        bl = _compute_fold_baselines(y_tr, bt_tr, y_te, bt_te)
        bl_maj_acc.append(bl["majority_class_acc"])
        bl_pc_acc.append(bl["per_cond_acc"])
        for bt, acc in bl["per_cond_by_bias"].items():
            if bt in bl_pc_by_bias:
                bl_pc_by_bias[bt].append(acc)

    # ── Aggregate ────────────────────────────────────────────────────────────
    def _agg(vals: List[float]) -> Dict[str, float]:
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    by_bias_agg = {
        bt: {
            "accuracy":  _agg(by_bias[bt]["acc"]),
            "precision": _agg(by_bias[bt]["prec"]),
            "recall":    _agg(by_bias[bt]["rec"]),
            "f1":        _agg(by_bias[bt]["f1"]),
        }
        for bt in unique_bias_types
    }

    result: Dict[str, Any] = {
        "accuracy":  _agg(fold_acc),
        "precision": _agg(fold_prec),
        "recall":    _agg(fold_rec),
        "f1":        _agg(fold_f1),
        "confusion_matrix": agg_cm,
        "by_bias_type": by_bias_agg,
        "feature_names": feature_names,
        "baselines": {
            "chance": 0.50,
            "majority_class": {"mean": float(np.mean(bl_maj_acc)), "std": float(np.std(bl_maj_acc))},
            "per_condition":  {"mean": float(np.mean(bl_pc_acc)),  "std": float(np.std(bl_pc_acc))},
            "per_condition_by_bias": {
                bt: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for bt, v in bl_pc_by_bias.items() if v
            },
        },
    }
    if model_kind == "lr" and coeff_list:
        coeff_arr = np.array(coeff_list)
        result["coeff_mean"] = coeff_arr.mean(axis=0).tolist()
        result["coeff_std"] = coeff_arr.std(axis=0).tolist()

    return result


# ── Helper: reconstruct raw DataFrame slice for re-fitting ────────────────────

_raw_df_cache: Optional[pd.DataFrame] = None  # set once per pipeline run


def _raw_from_indices(
    X_encoded: np.ndarray,
    feature_names: List[str],
    indices: np.ndarray,
) -> pd.DataFrame:
    """
    Reconstruct a raw (un-preprocessed) DataFrame slice for the given row indices.
    We keep a module-level reference to the raw DataFrame so each CV fold can re-fit
    its own preprocessor on training data only (preventing data leakage).
    """
    if _raw_df_cache is None:
        raise RuntimeError("Call run_pipeline() before _raw_from_indices().")
    return _raw_df_cache.iloc[indices][FEATURE_COLS].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline on a responses DataFrame.

    Parameters
    ----------
    df : DataFrame containing at least FEATURE_COLS + [TARGET_COL, "participant_id"]

    Returns
    -------
    dict with keys "lr", "nn", "n_participants", "n_samples", plus Figure objects:
        "fig_cm_lr", "fig_cm_nn", "fig_bias_bar", "fig_lr_coeff"
    """
    global _raw_df_cache

    n_participants = df["participant_id"].nunique()
    k = CV_K if n_participants >= CV_K else (CV_K_FALLBACK if n_participants >= CV_K_FALLBACK else None)
    if k is None:
        raise ValueError(
            f"Need at least {MIN_PARTICIPANTS_FOR_CV} participants for cross-validation. "
            f"Got {n_participants}."
        )
    if n_participants < CV_K:
        print(f"[pipeline] Only {n_participants} participants — using k={k} folds.")

    # Cache the raw DataFrame for the per-fold preprocessors in _run_cv
    _raw_df_cache = df.copy()

    X, y, groups, feature_names = load_and_prepare(df)
    bias_type_col = df["bias_type"].to_numpy()

    print(f"[pipeline] Running logistic regression CV (k={k})...")
    lr_result = _run_cv(X, y, groups, bias_type_col, "lr", feature_names, k)

    print(f"[pipeline] Running neural-network CV (k={k})...")
    nn_result = _run_cv(X, y, groups, bias_type_col, "nn", feature_names, k)

    # ── Figures ──────────────────────────────────────────────────────────────
    fig_cm_lr   = _plot_confusion_matrix(lr_result["confusion_matrix"], "Logistic Regression")
    fig_cm_nn   = _plot_confusion_matrix(nn_result["confusion_matrix"], "Neural Network")
    fig_bias    = _plot_bias_bar(lr_result, nn_result)
    fig_lr_coef = _plot_lr_coefficients(lr_result)

    return {
        "lr": lr_result,
        "nn": nn_result,
        "k": k,
        "n_participants": n_participants,
        "n_samples": len(df),
        "feature_names": feature_names,
        "fig_cm_lr":   fig_cm_lr,
        "fig_cm_nn":   fig_cm_nn,
        "fig_bias_bar": fig_bias,
        "fig_lr_coeff": fig_lr_coef,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, title: str) -> plt.Figure:
    """Plot a 2×2 confusion matrix aggregated across all CV folds."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred A (0)", "Pred B (1)"])
    ax.set_yticklabels(["True A (0)", "True B (1)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title(f"Confusion Matrix — {title}\n(Aggregated across CV folds)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plot_bias_bar(lr: Dict, nn: Dict) -> plt.Figure:
    """Grouped bar chart of accuracy per bias type for LR, NN, and the three baselines."""
    bias_types_raw = sorted(lr["by_bias_type"].keys())
    labels = [BIAS_TYPE_LABEL.get(bt, str(bt)).replace("_", " ").title() for bt in bias_types_raw]

    lr_acc  = [lr["by_bias_type"][bt]["accuracy"]["mean"] for bt in bias_types_raw]
    nn_acc  = [nn["by_bias_type"][bt]["accuracy"]["mean"] for bt in bias_types_raw]

    # Per-condition baselines from LR result (both models share the same baselines)
    pc_bl   = [lr["baselines"]["per_condition_by_bias"].get(bt, {}).get("mean", 0.5)
               for bt in bias_types_raw]

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 1.5*width, lr_acc,  width, label="Logistic Regression", color="#4C72B0")
    ax.bar(x - 0.5*width, nn_acc,  width, label="Neural Network",      color="#DD8452")
    ax.bar(x + 0.5*width, pc_bl,   width, label="Per-cond. Majority",  color="#55A868", alpha=0.7)

    # Chance line and majority-class line
    chance_line = 0.50
    maj_mean = lr["baselines"]["majority_class"]["mean"]
    ax.axhline(chance_line, color="red",   linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax.axhline(maj_mean,    color="purple",linestyle=":",  linewidth=1.2,
               label=f"Majority-class ({maj_mean:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Accuracy (CV)")
    ax.set_title("Predictive Accuracy by Cognitive Bias Type")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_lr_coefficients(lr: Dict) -> plt.Figure:
    """Horizontal bar chart of mean LR feature coefficients across folds."""
    if "coeff_mean" not in lr:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No coefficients available", ha="center", va="center")
        return fig

    names = lr["feature_names"]
    means = np.array(lr["coeff_mean"])
    stds  = np.array(lr["coeff_std"])
    order = np.argsort(np.abs(means))[::-1]

    fig, ax = plt.subplots(figsize=(7, max(4, len(names) * 0.45)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means[order], xerr=stds[order],
            align="center", color=["#4C72B0" if v >= 0 else "#C44E52" for v in means[order]],
            alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (mean ± std across folds)")
    ax.set_title("Logistic Regression — Feature Coefficients\n"
                 "(positive → predicts Choice B; negative → predicts Choice A)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers for script mode
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(d: Dict[str, float], pct: bool = True) -> str:
    scale = 100 if pct else 1
    mean = d.get("mean", float("nan")) * scale
    std  = d.get("std",  float("nan")) * scale
    unit = "%" if pct else ""
    return f"{mean:.1f}{unit} ± {std:.1f}{unit}"


def print_results(results: Dict[str, Any]) -> None:
    print("\n═══ OVERALL MODEL PERFORMANCE ════════════════════════════════")
    header = f"{'Metric':<14} {'Logistic Reg':>18} {'Neural Network':>18}"
    print(header)
    print("─" * len(header))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        lr_v = results["lr"][metric]
        nn_v = results["nn"][metric]
        print(f"  {metric:<12} {_fmt(lr_v):>18} {_fmt(nn_v):>18}")

    blr = results["lr"]["baselines"]
    print(f"\n  Chance baseline             : 50.0%")
    print(f"  Majority-class baseline     : {_fmt(blr['majority_class'])}")
    print(f"  Per-condition majority-vote : {_fmt(blr['per_condition'])}")

    pc_mean = blr["per_condition"]["mean"]
    for model_name, res in [("LR", results["lr"]), ("NN", results["nn"])]:
        acc_mean = res["accuracy"]["mean"]
        delta = (acc_mean - pc_mean) * 100
        sign = "+" if delta >= 0 else ""
        verdict = "BEATS" if delta > 0 else "does NOT beat"
        print(f"\n  {model_name}: accuracy = {acc_mean*100:.1f}%  ({sign}{delta:.1f} pp vs majority-vote) → {verdict} majority-vote baseline")

    print("\n═══ PERFORMANCE BY BIAS TYPE ══════════════════════════════════")
    bias_types = sorted(results["lr"]["by_bias_type"].keys())
    hdr2 = f"  {'Bias Type':<20} {'Model':>6} {'Acc':>10} {'Prec':>10} {'Rec':>10} {'F1':>10} {'MV-BL':>10}"
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))
    from config import BIAS_TYPE_LABEL
    for bt in bias_types:
        label = BIAS_TYPE_LABEL.get(bt, str(bt)).replace("_", " ").title()
        mv_bl = blr["per_condition_by_bias"].get(bt, {}).get("mean", float("nan")) * 100
        for model_name, res in [("LR", results["lr"]), ("NN", results["nn"])]:
            bm = res["by_bias_type"][bt]
            print(
                f"  {label:<20} {model_name:>6} "
                f"{_fmt(bm['accuracy']):>10} {_fmt(bm['precision']):>10} "
                f"{_fmt(bm['recall']):>10} {_fmt(bm['f1']):>10} "
                f"{mv_bl:>9.1f}%"
            )

    if "coeff_mean" in results["lr"]:
        print("\n═══ LOGISTIC REGRESSION FEATURE COEFFICIENTS ══════════════════")
        names  = results["feature_names"]
        means  = results["lr"]["coeff_mean"]
        stds   = results["lr"]["coeff_std"]
        order  = sorted(range(len(means)), key=lambda i: abs(means[i]), reverse=True)
        print(f"  {'Feature':<32} {'Mean coeff':>12} {'Std':>8}")
        print("  " + "─" * 55)
        for i in order:
            print(f"  {names[i]:<32} {means[i]:>+12.4f} {stds[i]:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <responses.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(csv_path).parent

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['participant_id'].nunique()} participants from '{csv_path}'")

    results = run_pipeline(df)
    print_results(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    results["fig_cm_lr"].savefig(out_dir / "confusion_matrix_lr.png", dpi=150)
    results["fig_cm_nn"].savefig(out_dir / "confusion_matrix_nn.png", dpi=150)
    results["fig_bias_bar"].savefig(out_dir / "accuracy_by_bias_type.png", dpi=150)
    results["fig_lr_coeff"].savefig(out_dir / "lr_coefficients.png", dpi=150)
    print(f"\nFigures saved → {out_dir}")
