"""
admin.py — Password-gated analysis and export dashboard.

Accessible at the Admin page in the sidebar. The admin password is read from
st.secrets["ADMIN_PASSWORD"]. All tables and figures are essay-ready:
  §4.1  overall model performance table
  §4.2  per-bias-type table + grouped bar chart
  §5.2  LR coefficient table + figure
  Appendix C  downloadable CSV
  Appendix D  downloadable confusion-matrix PNGs
"""

import io
import os
import tempfile

import pandas as pd
import streamlit as st

from storage import fetch_all, check_connection
from pipeline import run_pipeline, print_results
from config import BIAS_TYPE_LABEL


# ─────────────────────────────────────────────────────────────────────────────
# Auth gate
# ─────────────────────────────────────────────────────────────────────────────

def _check_password() -> bool:
    """Return True once the correct admin password has been entered."""
    if st.session_state.get("admin_authenticated"):
        return True

    st.title("Admin Access")
    pwd = st.text_input("Enter admin password:", type="password", key="admin_pwd_input")
    if st.button("Login"):
        try:
            correct = st.secrets.get("ADMIN_PASSWORD", "")
        except Exception:
            correct = os.environ.get("ADMIN_PASSWORD", "")
        if pwd == correct and correct:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(d: dict, pct: bool = True) -> str:
    scale = 100 if pct else 1
    mean = d.get("mean", float("nan")) * scale
    std  = d.get("std",  float("nan")) * scale
    unit = "%" if pct else ""
    return f"{mean:.1f}{unit} ± {std:.1f}{unit}"


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard sections
# ─────────────────────────────────────────────────────────────────────────────

def _section_data(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    n_part = df["participant_id"].nunique()
    n_rows = len(df)
    n_A = int((df["choice"] == 0).sum())
    n_B = int((df["choice"] == 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows",    n_rows)
    c2.metric("Participants",  n_part)
    c3.metric("Choice A (0)", n_A)
    c4.metric("Choice B (1)", n_B)

    st.write("**Class balance:**", f"A={n_A/n_rows*100:.1f}%  B={n_B/n_rows*100:.1f}%")

    with st.expander("Preview first 10 rows"):
        st.dataframe(df.head(10), use_container_width=True)


def _section_41(results: dict) -> None:
    """§4.1 — Overall model performance table."""
    st.subheader("§ 4.1  Overall Model Performance")
    st.caption(
        f"Mean ± standard deviation across {results['k']}-fold grouped cross-validation "
        f"({results['n_participants']} participants, {results['n_samples']} trials)."
    )

    rows = []
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for metric, label in zip(metrics, labels):
        rows.append({
            "Metric":           label,
            "Logistic Regression": _fmt(results["lr"][metric]),
            "Neural Network":      _fmt(results["nn"][metric]),
        })

    bl = results["lr"]["baselines"]
    rows.append({"Metric": "— Chance baseline",           "Logistic Regression": "50.0%", "Neural Network": "50.0%"})
    rows.append({"Metric": "— Majority-class baseline",   "Logistic Regression": _fmt(bl["majority_class"]), "Neural Network": _fmt(bl["majority_class"])})
    rows.append({"Metric": "— Per-condition majority-vote","Logistic Regression": _fmt(bl["per_condition"]), "Neural Network": _fmt(bl["per_condition"])})
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

    # Verdict
    pc_mean = bl["per_condition"]["mean"]
    for model_name, res in [("Logistic Regression", results["lr"]), ("Neural Network", results["nn"])]:
        acc = res["accuracy"]["mean"]
        delta = (acc - pc_mean) * 100
        sign = "+" if delta >= 0 else ""
        verb = "**beats**" if delta > 0 else "does **not** beat"
        st.write(
            f"→ **{model_name}**: accuracy = {acc*100:.1f}%  "
            f"({sign}{delta:.1f} pp vs majority-vote)  —  {verb} the majority-vote baseline."
        )


def _section_42(results: dict) -> None:
    """§4.2 — Performance by bias type, table + chart."""
    st.subheader("§ 4.2  Predictive Performance by Bias Type")

    bias_types = sorted(results["lr"]["by_bias_type"].keys())
    bl_pc = results["lr"]["baselines"]["per_condition_by_bias"]

    rows = []
    for bt in bias_types:
        label = BIAS_TYPE_LABEL.get(bt, str(bt)).replace("_", " ").title()
        mv_bl = bl_pc.get(bt, {}).get("mean", float("nan")) * 100
        for model_name, res in [("Logistic Regression", results["lr"]), ("Neural Network", results["nn"])]:
            bm = res["by_bias_type"][bt]
            rows.append({
                "Bias Type": label,
                "Model": model_name,
                "Accuracy":  _fmt(bm["accuracy"]),
                "Precision": _fmt(bm["precision"]),
                "Recall":    _fmt(bm["recall"]),
                "F1-Score":  _fmt(bm["f1"]),
                "MV Baseline": f"{mv_bl:.1f}%",
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Bar chart (figures are already computed by pipeline)
    st.pyplot(results["fig_bias_bar"])
    st.download_button(
        "Download chart (PNG)",
        data=_fig_to_bytes(results["fig_bias_bar"]),
        file_name="accuracy_by_bias_type.png",
        mime="image/png",
    )


def _section_confusion(results: dict) -> None:
    """Confusion matrices for §4.1 / Appendix D."""
    st.subheader("Confusion Matrices (Appendix D)")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Logistic Regression")
        st.pyplot(results["fig_cm_lr"])
        st.download_button(
            "Download LR confusion matrix",
            data=_fig_to_bytes(results["fig_cm_lr"]),
            file_name="confusion_matrix_lr.png",
            mime="image/png",
        )
    with col2:
        st.caption("Neural Network")
        st.pyplot(results["fig_cm_nn"])
        st.download_button(
            "Download NN confusion matrix",
            data=_fig_to_bytes(results["fig_cm_nn"]),
            file_name="confusion_matrix_nn.png",
            mime="image/png",
        )


def _section_52(results: dict) -> None:
    """§5.2 — LR coefficients."""
    st.subheader("§ 5.2  Logistic Regression Feature Coefficients")
    st.caption(
        "Mean ± std of fitted coefficients across CV folds. "
        "Positive coefficient → feature increases probability of choosing B (1). "
        "Negative → increases probability of choosing A (0)."
    )

    if "coeff_mean" in results["lr"]:
        names  = results["feature_names"]
        means  = results["lr"]["coeff_mean"]
        stds   = results["lr"]["coeff_std"]
        import numpy as np
        order  = np.argsort(np.abs(means))[::-1]
        coeff_rows = [
            {
                "Feature":         names[i],
                "Mean Coefficient": f"{means[i]:+.4f}",
                "Std":             f"{stds[i]:.4f}",
                "Direction":       "→ B" if means[i] > 0 else "→ A",
            }
            for i in order
        ]
        st.dataframe(pd.DataFrame(coeff_rows), use_container_width=True, hide_index=True)

    st.pyplot(results["fig_lr_coeff"])
    st.download_button(
        "Download coefficient chart (PNG)",
        data=_fig_to_bytes(results["fig_lr_coeff"]),
        file_name="lr_coefficients.png",
        mime="image/png",
    )


def _section_export(df: pd.DataFrame) -> None:
    """Appendix C — CSV export."""
    st.subheader("Appendix C  — Export Dataset (CSV)")
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label=f"Download full dataset  ({len(df)} rows)",
        data=csv_bytes,
        file_name="cogbias_responses.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_admin() -> None:
    """Main entry point called from app.py."""
    if not _check_password():
        return

    st.title("Admin Dashboard — Cognitive Bias ML Analysis")

    # ── Data ──────────────────────────────────────────────────────────────────
    st.header("1.  Data")
    if st.button("Pull latest data from Supabase"):
        df = fetch_all()
        if df is not None and not df.empty:
            st.session_state["admin_df"] = df
            st.success(f"Loaded {len(df)} rows from {df['participant_id'].nunique()} participants.")
        elif df is not None:
            st.warning("The database is empty — no participant data yet.")
        # errors shown by storage.py

    # Also allow uploading a local CSV (useful for testing with synthetic data)
    uploaded = st.file_uploader("Or upload a local CSV file:", type="csv")
    if uploaded is not None:
        st.session_state["admin_df"] = pd.read_csv(uploaded)
        st.success(f"Loaded {len(st.session_state['admin_df'])} rows from uploaded file.")

    df: pd.DataFrame | None = st.session_state.get("admin_df")
    if df is None or df.empty:
        st.info("Load data above to proceed.")
        return

    _section_data(df)
    _section_export(df)

    st.divider()

    # ── Analysis ──────────────────────────────────────────────────────────────
    st.header("2.  Run Analysis")
    n_part = df["participant_id"].nunique()
    if n_part < 3:
        st.warning(f"Need at least 3 participants for cross-validation. Got {n_part}.")
        return

    if st.button("Run full analysis (may take 1–3 minutes)", type="primary"):
        with st.spinner("Training models with grouped cross-validation..."):
            try:
                results = run_pipeline(df)
                st.session_state["admin_results"] = results
                st.success("Analysis complete.")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

    results: dict | None = st.session_state.get("admin_results")
    if results is None:
        st.info("Click the button above to run the ML analysis.")
        return

    st.divider()
    st.header("3.  Results")

    tab1, tab2, tab3, tab4 = st.tabs(["§4.1 Overall", "§4.2 By Bias Type", "§5.2 Coefficients", "Confusion Matrices"])
    with tab1:
        _section_41(results)
    with tab2:
        _section_42(results)
    with tab3:
        _section_52(results)
    with tab4:
        _section_confusion(results)
