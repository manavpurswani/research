"""pipeline.py — Standalone ML evaluation pipeline for cognitive-bias prediction.

Can be imported by admin.py  OR  run directly:
    python pipeline.py path/to/responses.csv [output_dir]

When run as a script it prints all metric tables and saves four figures to output_dir:
    confusion_matrix_lr.png
    confusion_matrix_nn.png
    accuracy_by_bias_type.png
    lr_coefficients.png

When imported, call run_full(df) which returns a results dict including matplotlib
Figure objects ready for st.pyplot().

────────────────────────────────────────────────────────────────────────────────
Evaluation design
  • Participant-grouped k-fold CV (GroupKFold, k=5; fallback k=3 for < 5 participants).
  • Trials from one participant never appear in both train and test in the same fold.
  • Preprocessing (StandardScaler, OneHotEncoder) is fit per-fold on training data
    only — no leakage from the test fold.
  • log1p is applied to reaction_time_ms before any other scaling (data-independent).
  • Three baselines compared against model accuracy:
      1. Chance (0.50 constant)
      2. Majority-class (modal choice in the training fold)
      3. Per-condition majority-vote (modal choice per bias_type in training fold)
  • Bias-stratified evaluation: metrics computed separately for each bias type
    within the same CV framework.
  • Wilson 95% CI on accuracy computed over all held-out predictions combined.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (
    BIAS_TYPE_LABEL, CV_K, CV_K_FALLBACK, FEATURE_COLS, GLOBAL_SEED,
    MIN_PARTICIPANTS_FOR_CV, NN_ACTIVATION, NN_EARLY_STOPPING,
    NN_HIDDEN_LAYER_SIZES, NN_MAX_ITER, NN_SOLVER, NN_VALIDATION_FRACTION,
    RT_TOO_FAST_MS, RT_TOO_SLOW_MS, SESSION_MIN_SECS, TARGET_COL,
)

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
warnings.filterwarnings("ignore", category=UserWarning)

_NUMERIC_FEATURES = [f for f in FEATURE_COLS if f != "bias_type"]
_CAT_FEATURES = ["bias_type"]


# ─────────────────────────────────────────────────────────────────────────────
# Quality flags
# ─────────────────────────────────────────────────────────────────────────────

def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean quality-flag columns to a copy of df.

    Columns added
    -------------
    rt_too_fast      : trial RT < RT_TOO_FAST_MS (200 ms)
    rt_too_slow      : trial RT > RT_TOO_SLOW_MS (60 000 ms)
    session_too_fast : participant's total session time < SESSION_MIN_SECS (90 s)
    attention_failed : participant failed the embedded attention check
    any_flag         : True if any above flag applies to this participant's rows
    """
    df = df.copy()

    df["rt_too_fast"] = df["reaction_time_ms"] < RT_TOO_FAST_MS
    df["rt_too_slow"] = df["reaction_time_ms"] > RT_TOO_SLOW_MS

    if "created_at" in df.columns:
        ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df["_ts"] = ts
        session_duration = df.groupby("participant_id")["_ts"].transform(
            lambda s: (s.max() - s.min()).total_seconds()
        )
        df["session_too_fast"] = session_duration < SESSION_MIN_SECS
        df.drop(columns=["_ts"], inplace=True)
    else:
        df["session_too_fast"] = False

    if "attention_check_passed" in df.columns:
        failed_pids = set(df.loc[df["attention_check_passed"] == False, "participant_id"])
        df["attention_failed"] = df["participant_id"].isin(failed_pids)
    else:
        df["attention_failed"] = False

    per_pid_any = (
        df.groupby("participant_id")[["rt_too_fast", "rt_too_slow", "session_too_fast"]]
        .transform("any")
        .any(axis=1)
    )
    df["any_flag"] = per_pid_any | df["attention_failed"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Validate and filter the responses DataFrame for modeling.

    Steps
    -----
    1. Keep completed == True rows only.
    2. Separate attention-check rows (attention_check_passed IS NOT NULL) into meta.
    3. Apply log1p to reaction_time_ms (data-independent — no leakage risk).
    4. Validate that required columns are present and numeric.

    Returns
    -------
    modeling_df : DataFrame of real trial rows for modeling
    y           : int array of binary labels
    groups      : participant_id array for GroupKFold
    meta        : DataFrame of attention-check rows (excluded from modeling)
    """
    required = set(FEATURE_COLS + [TARGET_COL, "participant_id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    if "completed" in df.columns:
        df = df[df["completed"] == True].copy()

    if "invalid_response" in df.columns:
        before = len(df)
        df = df[df["invalid_response"].astype(str) != "True"].copy()
        dropped = before - len(df)
        if dropped:
            print(f"[pipeline] Dropped {dropped} invalid_response rows.", flush=True)

    if "attention_check_passed" in df.columns:
        attn_mask = df["attention_check_passed"].notna()
        meta = df[attn_mask].copy()
        df = df[~attn_mask].copy()
    else:
        meta = pd.DataFrame()

    if df.empty:
        raise ValueError(
            "No trial rows remain after filtering completed=True and attention checks."
        )

    df["reaction_time_ms"] = np.log1p(df["reaction_time_ms"].astype(float))

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[FEATURE_COLS + [TARGET_COL]].isnull().any().any():
        raise ValueError("NaN values in required columns after coercion.")

    y = df[TARGET_COL].to_numpy(dtype=int)
    groups = df["participant_id"].to_numpy()
    return df, y, groups, meta


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_FEATURES),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), _CAT_FEATURES),
        ],
        remainder="drop",
    )


def fit_logreg(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Train logistic regression with L2 regularisation (default C=1.0)."""
    clf = LogisticRegression(max_iter=1000, random_state=GLOBAL_SEED, solver="lbfgs")
    clf.fit(X, y)
    return clf


def fit_mlp(X: np.ndarray, y: np.ndarray) -> MLPClassifier:
    """Train a shallow MLP: hidden_layer_sizes=(16,), relu, adam, early stopping."""
    clf = MLPClassifier(
        hidden_layer_sizes=NN_HIDDEN_LAYER_SIZES,
        activation=NN_ACTIVATION,
        solver=NN_SOLVER,
        max_iter=NN_MAX_ITER,
        early_stopping=NN_EARLY_STOPPING,
        validation_fraction=NN_VALIDATION_FRACTION,
        random_state=GLOBAL_SEED,
    )
    clf.fit(X, y)
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Wilson confidence interval
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return (lo, hi) Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
        }
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _compute_fold_baselines(
    y_train: np.ndarray, bt_train: np.ndarray,
    y_test: np.ndarray,  bt_test: np.ndarray,
) -> Dict[str, Any]:
    majority = int(np.bincount(y_train).argmax())
    y_maj = np.full(len(y_test), majority)

    y_pc = y_maj.copy()
    pc_by_bias: Dict[int, float] = {}
    for bt in np.unique(bt_train):
        m_tr = bt_train == bt
        if m_tr.sum() == 0:
            continue
        modal = int(np.bincount(y_train[m_tr]).argmax())
        m_te = bt_test == bt
        y_pc[m_te] = modal
        if m_te.sum() > 0:
            pc_by_bias[int(bt)] = float(accuracy_score(y_test[m_te], y_pc[m_te]))

    return {
        "chance": 0.50,
        "majority_class_acc": float(accuracy_score(y_test, y_maj)),
        "per_cond_acc":       float(accuracy_score(y_test, y_pc)),
        "per_cond_by_bias":   pc_by_bias,
    }


def cv_evaluate(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    fit_fn: Callable,
    k: int,
    bias_types: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run GroupKFold cross-validation for either fit_logreg or fit_mlp.

    Preprocessing is fit per-fold on training data only (no leakage).
    Returns aggregated metrics, Wilson CI, baselines, and (for LR) coefficients.
    """
    gkf = GroupKFold(n_splits=k)
    X_raw = df[FEATURE_COLS]
    bt_col = bias_types if bias_types is not None else np.zeros(len(y), dtype=int)
    unique_bt = sorted(np.unique(bt_col).tolist())

    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
    agg_cm = np.zeros((2, 2), dtype=int)
    by_bias: Dict[int, Dict[str, List[float]]] = {
        bt: {"acc": [], "prec": [], "rec": [], "f1": []} for bt in unique_bt
    }
    bl_maj_acc, bl_pc_acc = [], []
    bl_pc_by_bias: Dict[int, List[float]] = {bt: [] for bt in unique_bt}
    coeff_list: List[np.ndarray] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    for train_idx, test_idx in gkf.split(X_raw, y, groups):
        X_tr_raw = X_raw.iloc[train_idx]
        X_te_raw = X_raw.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        bt_tr, bt_te = bt_col[train_idx], bt_col[test_idx]

        pre = _build_preprocessor()
        X_tr = pre.fit_transform(X_tr_raw).astype(np.float32)
        X_te = pre.transform(X_te_raw).astype(np.float32)

        if len(np.unique(y_tr)) < 2:
            warnings.warn("Training fold has only one class — skipping fold.")
            continue

        clf = fit_fn(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        m = _clf_metrics(y_te, y_pred)
        fold_acc.append(m["accuracy"])
        fold_prec.append(m["precision"])
        fold_rec.append(m["recall"])
        fold_f1.append(m["f1"])
        agg_cm += confusion_matrix(y_te, y_pred, labels=[0, 1])
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())

        for bt in unique_bt:
            mask = bt_te == bt
            if mask.sum() == 0:
                continue
            bm = _clf_metrics(y_te[mask], y_pred[mask])
            by_bias[bt]["acc"].append(bm["accuracy"])
            by_bias[bt]["prec"].append(bm["precision"])
            by_bias[bt]["rec"].append(bm["recall"])
            by_bias[bt]["f1"].append(bm["f1"])

        bl = _compute_fold_baselines(y_tr, bt_tr, y_te, bt_te)
        bl_maj_acc.append(bl["majority_class_acc"])
        bl_pc_acc.append(bl["per_cond_acc"])
        for bt, acc in bl["per_cond_by_bias"].items():
            if bt in bl_pc_by_bias:
                bl_pc_by_bias[bt].append(acc)

        if hasattr(clf, "coef_"):
            coeff_list.append(clf.coef_[0])

    def _agg(vals: List[float]) -> Dict[str, float]:
        arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        return {
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "std":  float(arr.std())  if len(arr) else float("nan"),
        }

    successes = int(sum(t == p for t, p in zip(all_y_true, all_y_pred)))
    ci_lo, ci_hi = wilson_ci(successes, len(all_y_true))

    result: Dict[str, Any] = {
        "accuracy":  _agg(fold_acc),
        "precision": _agg(fold_prec),
        "recall":    _agg(fold_rec),
        "f1":        _agg(fold_f1),
        "wilson_ci": (ci_lo, ci_hi),
        "confusion_matrix": agg_cm,
        "by_bias_type": {
            bt: {
                "accuracy":  _agg(by_bias[bt]["acc"]),
                "precision": _agg(by_bias[bt]["prec"]),
                "recall":    _agg(by_bias[bt]["rec"]),
                "f1":        _agg(by_bias[bt]["f1"]),
            }
            for bt in unique_bt
        },
        "baselines": {
            "chance": 0.50,
            "majority_class": _agg(bl_maj_acc),
            "per_condition":  _agg(bl_pc_acc),
            "per_condition_by_bias": {
                bt: _agg(v) for bt, v in bl_pc_by_bias.items() if v
            },
        },
    }

    if coeff_list:
        pre_ref = _build_preprocessor().fit(X_raw)
        result["feature_names"] = list(pre_ref.get_feature_names_out())
        coeff_arr = np.array(coeff_list)
        result["coeff_mean"] = coeff_arr.mean(axis=0).tolist()
        result["coeff_std"]  = coeff_arr.std(axis=0).tolist()

    return result


def confusion_matrices(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    fit_fn: Callable,
    k: int,
    bias_types: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Return aggregated confusion matrix aggregated across all CV folds."""
    res = cv_evaluate(df, y, groups, fit_fn, k, bias_types)
    return {"global": res["confusion_matrix"]}


def logreg_coefficients(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fit LR on all data and return feature coefficients sorted by |coef|."""
    X_raw = df[FEATURE_COLS]
    pre = _build_preprocessor()
    X = pre.fit_transform(X_raw).astype(np.float32)
    names = feature_names or list(pre.get_feature_names_out())
    clf = fit_logreg(X, y)
    df_coef = pd.DataFrame({"feature": names, "coefficient": clf.coef_[0]})
    df_coef["abs_coef"] = df_coef["coefficient"].abs()
    return (
        df_coef.sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_full(df: pd.DataFrame, exclude_flagged: bool = False) -> Dict[str, Any]:
    """
    Orchestrate the full evaluation pipeline.

    Runs analysis twice if called separately (once with all data, once excluding
    flagged participants). Call this function twice with exclude_flagged=False
    and exclude_flagged=True to produce both reports.

    Returns
    -------
    dict with "lr", "nn", "k", "n_participants", "n_samples", "n_flagged",
    "feature_names", "exclude_flagged", plus matplotlib Figure objects.
    """
    df_q = flag_quality(df)
    n_flagged = int(df_q["any_flag"].sum())

    if exclude_flagged:
        flagged_pids = set(df_q.loc[df_q["any_flag"], "participant_id"])
        df_use = df[~df["participant_id"].isin(flagged_pids)].copy()
    else:
        df_use = df.copy()

    n_participants = df_use["participant_id"].nunique()
    k = (
        CV_K if n_participants >= CV_K
        else CV_K_FALLBACK if n_participants >= CV_K_FALLBACK
        else None
    )
    if k is None:
        raise ValueError(
            f"Need at least {MIN_PARTICIPANTS_FOR_CV} participants for CV; got {n_participants}."
        )
    if n_participants < CV_K:
        print(f"[pipeline] Only {n_participants} participants — using k={k} folds.")

    modeling_df, y, groups, meta = load_and_prepare(df_use)
    bias_types = modeling_df["bias_type"].to_numpy()

    pre_ref = _build_preprocessor().fit(modeling_df[FEATURE_COLS])
    feature_names = list(pre_ref.get_feature_names_out())

    print(f"[pipeline] Logistic Regression CV (k={k}, n={n_participants})...")
    lr_result = cv_evaluate(modeling_df, y, groups, fit_logreg, k, bias_types)

    print(f"[pipeline] MLP CV (k={k})...")
    nn_result = cv_evaluate(modeling_df, y, groups, fit_mlp, k, bias_types)

    if "feature_names" in lr_result:
        feature_names = lr_result["feature_names"]

    fig_cm_lr   = _plot_confusion_matrix(lr_result["confusion_matrix"], "Logistic Regression")
    fig_cm_nn   = _plot_confusion_matrix(nn_result["confusion_matrix"], "Neural Network (MLP)")
    fig_bias    = _plot_bias_bar(lr_result, nn_result)
    fig_lr_coef = _plot_lr_coefficients(lr_result, feature_names)

    return {
        "lr": lr_result,
        "nn": nn_result,
        "k": k,
        "n_participants": n_participants,
        "n_samples": len(modeling_df),
        "n_flagged": n_flagged,
        "feature_names": feature_names,
        "exclude_flagged": exclude_flagged,
        "fig_cm_lr":    fig_cm_lr,
        "fig_cm_nn":    fig_cm_nn,
        "fig_bias_bar": fig_bias,
        "fig_lr_coeff": fig_lr_coef,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, title: str) -> plt.Figure:
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
    bias_types = sorted(lr["by_bias_type"].keys())
    labels = [BIAS_TYPE_LABEL.get(bt, str(bt)).replace("_", " ").title() for bt in bias_types]
    lr_acc = [lr["by_bias_type"][bt]["accuracy"]["mean"] for bt in bias_types]
    nn_acc = [nn["by_bias_type"][bt]["accuracy"]["mean"] for bt in bias_types]
    pc_bl  = [
        lr["baselines"]["per_condition_by_bias"].get(bt, {}).get("mean", 0.5)
        for bt in bias_types
    ]
    x = np.arange(len(labels)); w = 0.22
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 1.5*w, lr_acc, w, label="Logistic Regression", color="#4C72B0")
    ax.bar(x - 0.5*w, nn_acc, w, label="MLP (Neural Network)", color="#DD8452")
    ax.bar(x + 0.5*w, pc_bl,  w, label="Per-cond. Majority",  color="#55A868", alpha=0.7)
    bl = lr["baselines"]
    ax.axhline(0.50, color="red",    linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax.axhline(bl["majority_class"]["mean"], color="purple", linestyle=":", linewidth=1.2,
               label=f"Majority-class ({bl['majority_class']['mean']:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Accuracy (CV)")
    ax.set_title("Predictive Accuracy by Cognitive Bias Type")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_lr_coefficients(lr: Dict, feature_names: List[str]) -> plt.Figure:
    if "coeff_mean" not in lr:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No coefficients available", ha="center", va="center")
        return fig
    names = feature_names
    means = np.array(lr["coeff_mean"])
    stds  = np.array(lr.get("coeff_std", [0.0] * len(means)))
    order = np.argsort(np.abs(means))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(4, len(names) * 0.45)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means[order], xerr=stds[order], align="center",
            color=["#4C72B0" if v >= 0 else "#C44E52" for v in means[order]],
            alpha=0.85, capsize=3)
    ax.set_yticks(y_pos); ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (mean ± std across folds)")
    ax.set_title(
        "Logistic Regression — Feature Coefficients\n"
        "(positive → predicts Choice B; negative → predicts Choice A)",
        fontsize=11,
    )
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
    k, np_, ns = results["k"], results["n_participants"], results["n_samples"]
    excl = results.get("exclude_flagged", False)
    label = "EXCLUDING FLAGGED" if excl else "ALL COMPLETE SESSIONS"
    sep = "═" * 66
    print(f"\n{sep}")
    print(f"  RESULTS — {label}   (k={k}, {np_} participants, {ns} trials)")
    print(sep)
    print(f"\n  {'Metric':<14} {'Logistic Reg':>20} {'Neural Network':>20}")
    print("  " + "─" * 56)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {metric:<14} {_fmt(results['lr'][metric]):>20} {_fmt(results['nn'][metric]):>20}")

    bl   = results["lr"]["baselines"]
    lr_ci = results["lr"]["wilson_ci"]
    nn_ci = results["nn"]["wilson_ci"]
    print(f"\n  Wilson 95% CI (LR accuracy) : [{lr_ci[0]*100:.1f}%, {lr_ci[1]*100:.1f}%]")
    print(f"  Wilson 95% CI (NN accuracy) : [{nn_ci[0]*100:.1f}%, {nn_ci[1]*100:.1f}%]")
    print(f"\n  Chance baseline             : 50.0%")
    print(f"  Majority-class baseline     : {_fmt(bl['majority_class'])}")
    print(f"  Per-condition majority-vote : {_fmt(bl['per_condition'])}")

    pc_mean = bl["per_condition"]["mean"]
    for model_name, res in [("LR", results["lr"]), ("NN", results["nn"])]:
        acc = res["accuracy"]["mean"]
        ci  = res["wilson_ci"]
        delta = (acc - pc_mean) * 100
        sign  = "+" if delta >= 0 else ""
        verdict = "BEATS" if delta > 0 else "does NOT beat"
        print(
            f"\n  {model_name}: {acc*100:.1f}% (95% CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%])  "
            f"({sign}{delta:.1f} pp vs majority-vote) → {verdict} majority-vote baseline"
        )

    bias_types = sorted(results["lr"]["by_bias_type"].keys())
    print(f"\n  {'Bias Type':<20} {'Model':>6} {'Acc':>9} {'Prec':>9} {'Rec':>9} {'F1':>9} {'MV-BL':>9}")
    print("  " + "─" * 66)
    for bt in bias_types:
        label2 = BIAS_TYPE_LABEL.get(bt, str(bt)).replace("_", " ").title()
        mv = bl["per_condition_by_bias"].get(bt, {}).get("mean", float("nan")) * 100
        for mn, res in [("LR", results["lr"]), ("NN", results["nn"])]:
            bm = res["by_bias_type"][bt]
            print(
                f"  {label2:<20} {mn:>6} "
                f"{_fmt(bm['accuracy']):>9} {_fmt(bm['precision']):>9} "
                f"{_fmt(bm['recall']):>9} {_fmt(bm['f1']):>9} {mv:>8.1f}%"
            )

    if "coeff_mean" in results["lr"]:
        names = results["feature_names"]
        means = results["lr"]["coeff_mean"]
        stds  = results["lr"].get("coeff_std", [0.0] * len(means))
        order = sorted(range(len(means)), key=lambda i: abs(means[i]), reverse=True)
        print(f"\n  {'Feature':<30} {'Mean coeff':>12} {'Std':>8}")
        print("  " + "─" * 53)
        for i in order:
            print(f"  {names[i]:<30} {means[i]:>+12.4f} {stds[i]:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <responses.csv> [output_dir]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    out_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent

    df_in = pd.read_csv(csv_path)
    print(f"Loaded {len(df_in)} rows, {df_in['participant_id'].nunique()} participants")

    results_all = None
    for excl in (False, True):
        results = run_full(df_in, exclude_flagged=excl)
        print_results(results)
        if not excl:
            results_all = results

    out_dir.mkdir(parents=True, exist_ok=True)
    results_all["fig_cm_lr"].savefig(out_dir / "confusion_matrix_lr.png", dpi=150)
    results_all["fig_cm_nn"].savefig(out_dir / "confusion_matrix_nn.png", dpi=150)
    results_all["fig_bias_bar"].savefig(out_dir / "accuracy_by_bias_type.png", dpi=150)
    results_all["fig_lr_coeff"].savefig(out_dir / "lr_coefficients.png", dpi=150)
    print(f"\nFigures saved → {out_dir}")
