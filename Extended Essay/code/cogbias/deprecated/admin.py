"""admin.py — Password-gated administration and analysis dashboard.

Accessible at ?mode=admin in the app URL. The admin password is read from
st.secrets["ADMIN_PASSWORD"] and compared with hmac.compare_digest to prevent
timing attacks.  After 5 failed attempts in one session the admin panel is
locked for 5 minutes.

All admin actions are logged to the admin_audit table in Supabase.

Dashboard sections
------------------
  1. Auth gate (password + lockout)
  2. Health check  — row counts, token counts, sanity warnings
  3. Collection toggle  — open / close data collection
  4. Run analysis  — run_full(df, exclude_flagged=False) and run_full(df, True)
     §4.1  Overall model performance table + Wilson CI
     §4.2  Per-bias-type table + grouped bar chart
     §5.2  LR coefficient table + figure
     Confusion matrices (Appendix D)
  5. Export  — CSV + manifest.json (Appendix C)
  6. Token management  — generate tokens, display URLs
"""

from __future__ import annotations

import hmac
import io
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import BIAS_TYPE_LABEL
from pipeline import run_full
from storage import (
    admin_generate_tokens,
    admin_get_tokens,
    admin_health_check,
    admin_log_audit,
    admin_sanity_check,
    admin_set_collection,
    fetch_all,
)
from tokens import format_survey_url


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MAX_FAILURES = 5
_LOCKOUT_SECS = 300  # 5 minutes


# ─────────────────────────────────────────────────────────────────────────────
# Auth gate
# ─────────────────────────────────────────────────────────────────────────────

def _check_password() -> bool:
    """Return True once the correct password has been entered for this session.

    Uses hmac.compare_digest to prevent timing attacks.
    Locks the admin panel for _LOCKOUT_SECS after _MAX_FAILURES failures.
    """
    if st.session_state.get("_admin_auth"):
        return True

    failures: int = st.session_state.get("_admin_failures", 0)
    lock_until: float = st.session_state.get("_admin_lock_until", 0.0)

    if failures >= _MAX_FAILURES:
        remaining = lock_until - time.monotonic()
        if remaining > 0:
            st.error(
                f"Too many failed attempts. Admin access locked for "
                f"{int(remaining) // 60}m {int(remaining) % 60}s."
            )
            return False
        else:
            st.session_state["_admin_failures"] = 0

    st.title("Admin Access")
    pwd = st.text_input("Password:", type="password", key="_admin_pwd")

    if st.button("Login", key="_admin_login"):
        try:
            correct = st.secrets["ADMIN_PASSWORD"]
        except (KeyError, Exception):
            st.error("ADMIN_PASSWORD is not set in Streamlit secrets.")
            return False

        match = bool(correct) and hmac.compare_digest(pwd.encode(), correct.encode())
        if match:
            st.session_state["_admin_auth"]     = True
            st.session_state["_admin_failures"] = 0
            try:
                admin_log_audit("login_success")
            except Exception:
                pass
            st.rerun()
        else:
            new_failures = st.session_state.get("_admin_failures", 0) + 1
            st.session_state["_admin_failures"] = new_failures
            if new_failures >= _MAX_FAILURES:
                st.session_state["_admin_lock_until"] = time.monotonic() + _LOCKOUT_SECS
                st.error(f"Too many failed attempts. Locked for {_LOCKOUT_SECS // 60} minutes.")
            else:
                st.error(f"Incorrect password. ({new_failures}/{_MAX_FAILURES} attempts)")
            try:
                admin_log_audit("login_failure", {"attempt": new_failures})
            except Exception:
                pass

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(d: Dict[str, float], pct: bool = True) -> str:
    scale = 100 if pct else 1
    mean = d.get("mean", float("nan")) * scale
    std  = d.get("std",  float("nan")) * scale
    unit = "%" if pct else ""
    return f"{mean:.1f}{unit} ± {std:.1f}{unit}"


def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Health check
# ─────────────────────────────────────────────────────────────────────────────

def _section_health() -> None:
    st.subheader("Health Check")
    if st.button("Refresh health stats"):
        try:
            stats = admin_health_check()
            st.session_state["_health"] = stats
        except Exception as exc:
            st.error(f"Health check failed: {exc}")
            return

    stats = st.session_state.get("_health")
    if stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total rows",       stats["total_rows"])
        c2.metric("Completed rows",   stats["completed_rows"])
        c3.metric("Tokens total",     stats["tokens_total"])
        c4.metric("Tokens used",      stats["tokens_used"])
        c5.metric("Tokens available", stats["tokens_available"])

    if st.button("Run sanity check"):
        try:
            warnings = admin_sanity_check()
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("No sanity issues found.")
        except Exception as exc:
            st.error(f"Sanity check failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Collection toggle
# ─────────────────────────────────────────────────────────────────────────────

def _section_collection() -> None:
    st.subheader("Data Collection Toggle")
    col_on, col_off = st.columns(2)
    with col_on:
        if st.button("Open collection", type="primary"):
            try:
                admin_set_collection(True)
                admin_log_audit("collection_open")
                st.success("Collection is now OPEN. Participants can submit responses.")
            except Exception as exc:
                st.error(f"Failed to open collection: {exc}")
    with col_off:
        if st.button("Close collection"):
            try:
                admin_set_collection(False)
                admin_log_audit("collection_close")
                st.warning("Collection is now CLOSED. New participants will be blocked.")
            except Exception as exc:
                st.error(f"Failed to close collection: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Data load
# ─────────────────────────────────────────────────────────────────────────────

def _section_data() -> Optional[pd.DataFrame]:
    st.subheader("Dataset")
    if st.button("Pull latest data from Supabase"):
        try:
            df = fetch_all()
            st.session_state["_admin_df"] = df
            admin_log_audit("fetch_data", {"rows": len(df)})
            st.success(
                f"Loaded {len(df)} rows from "
                f"{df['participant_id'].nunique() if not df.empty else 0} participants."
            )
        except Exception as exc:
            st.error(f"Fetch failed: {exc}")

    uploaded = st.file_uploader("Or upload a CSV:", type="csv", key="_admin_csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["_admin_df"] = df
        st.success(f"Loaded {len(df)} rows from uploaded file.")

    df = st.session_state.get("_admin_df")
    if df is None or df.empty:
        st.info("Load data above to proceed.")
        return None

    n_p = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
    n_r = len(df)
    st.caption(f"{n_r} rows · {n_p} participants")
    with st.expander("Preview first 10 rows"):
        st.dataframe(df.head(10), use_container_width=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _section_41(results: Dict[str, Any], label: str) -> None:
    st.subheader(f"§ 4.1  Overall Model Performance  ({label})")
    ci_lr = results["lr"]["wilson_ci"]
    ci_nn = results["nn"]["wilson_ci"]
    st.caption(
        f"k={results['k']}-fold grouped CV · "
        f"{results['n_participants']} participants · {results['n_samples']} trials · "
        f"LR 95% CI [{ci_lr[0]*100:.1f}%, {ci_lr[1]*100:.1f}%] · "
        f"NN 95% CI [{ci_nn[0]*100:.1f}%, {ci_nn[1]*100:.1f}%]"
    )
    rows = []
    for metric, mlabel in [("accuracy","Accuracy"),("precision","Precision"),
                            ("recall","Recall"),("f1","F1-Score")]:
        rows.append({
            "Metric": mlabel,
            "Logistic Regression": _fmt(results["lr"][metric]),
            "Neural Network":      _fmt(results["nn"][metric]),
        })
    bl = results["lr"]["baselines"]
    rows += [
        {"Metric": "Chance baseline",             "Logistic Regression": "50.0%",                    "Neural Network": "50.0%"},
        {"Metric": "Majority-class baseline",     "Logistic Regression": _fmt(bl["majority_class"]), "Neural Network": _fmt(bl["majority_class"])},
        {"Metric": "Per-condition majority-vote", "Logistic Regression": _fmt(bl["per_condition"]),  "Neural Network": _fmt(bl["per_condition"])},
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

    pc = bl["per_condition"]["mean"]
    for mn, res in [("Logistic Regression", results["lr"]), ("Neural Network", results["nn"])]:
        acc   = res["accuracy"]["mean"]
        delta = (acc - pc) * 100
        sign  = "+" if delta >= 0 else ""
        verb  = "**beats**" if delta > 0 else "does **not** beat"
        st.write(f"**{mn}**: {acc*100:.1f}%  ({sign}{delta:.1f} pp)  — {verb} majority-vote baseline.")


def _section_42(results: Dict[str, Any], label: str) -> None:
    st.subheader(f"§ 4.2  Performance by Bias Type  ({label})")
    bias_types = sorted(results["lr"]["by_bias_type"].keys())
    bl_pc = results["lr"]["baselines"].get("per_condition_by_bias", {})
    rows = []
    for bt in bias_types:
        bt_label = BIAS_TYPE_LABEL.get(bt, str(bt)).replace("_", " ").title()
        mv = bl_pc.get(bt, {}).get("mean", float("nan")) * 100
        for mn, res in [("Logistic Regression", results["lr"]), ("Neural Network", results["nn"])]:
            bm = res["by_bias_type"][bt]
            rows.append({
                "Bias Type": bt_label, "Model": mn,
                "Accuracy":  _fmt(bm["accuracy"]), "Precision": _fmt(bm["precision"]),
                "Recall":    _fmt(bm["recall"]),   "F1":        _fmt(bm["f1"]),
                "MV-BL": f"{mv:.1f}%",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.pyplot(results["fig_bias_bar"])
    st.download_button("Download chart (PNG)", _fig_bytes(results["fig_bias_bar"]),
                       "accuracy_by_bias_type.png", "image/png", key=f"dl_bias_{label}")


def _section_52(results: Dict[str, Any], label: str) -> None:
    st.subheader(f"§ 5.2  LR Feature Coefficients  ({label})")
    if "coeff_mean" in results["lr"]:
        names = results["feature_names"]
        means = results["lr"]["coeff_mean"]
        stds  = results["lr"]["coeff_std"]
        order = np.argsort(np.abs(means))[::-1]
        rows  = [{"Feature": names[i], "Mean Coeff": f"{means[i]:+.4f}",
                  "Std": f"{stds[i]:.4f}", "Direction": "→ B" if means[i] > 0 else "→ A"}
                 for i in order]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.pyplot(results["fig_lr_coeff"])
    st.download_button("Download chart (PNG)", _fig_bytes(results["fig_lr_coeff"]),
                       "lr_coefficients.png", "image/png", key=f"dl_coef_{label}")


def _section_confusion(results: Dict[str, Any], label: str) -> None:
    st.subheader(f"Confusion Matrices — {label}  (Appendix D)")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Logistic Regression")
        st.pyplot(results["fig_cm_lr"])
        st.download_button("Download LR", _fig_bytes(results["fig_cm_lr"]),
                           "confusion_matrix_lr.png", "image/png", key=f"dl_cm_lr_{label}")
    with c2:
        st.caption("Neural Network")
        st.pyplot(results["fig_cm_nn"])
        st.download_button("Download NN", _fig_bytes(results["fig_cm_nn"]),
                           "confusion_matrix_nn.png", "image/png", key=f"dl_cm_nn_{label}")


def _section_analysis(df: pd.DataFrame) -> None:
    st.subheader("Run Analysis")
    n_p = df["participant_id"].nunique() if "participant_id" in df.columns else 0
    if n_p < 3:
        st.warning(f"Need at least 3 participants for CV. Got {n_p}.")
        return

    if st.button("Run full analysis (both all-data and cleaned)", type="primary"):
        with st.spinner("Running grouped CV for both models (may take 1-3 min)..."):
            try:
                r_all  = run_full(df, exclude_flagged=False)
                r_excl = run_full(df, exclude_flagged=True)
                st.session_state["_results_all"]  = r_all
                st.session_state["_results_excl"] = r_excl
                admin_log_audit("run_analysis", {
                    "n_participants": n_p,
                    "n_samples": r_all["n_samples"],
                    "n_flagged": r_all["n_flagged"],
                })
                st.success("Analysis complete.")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

    for r_key, rlabel in [("_results_all", "All Sessions"), ("_results_excl", "Excluding Flagged")]:
        results = st.session_state.get(r_key)
        if results is None:
            continue
        st.divider()
        with st.expander(f"Results — {rlabel}", expanded=(r_key == "_results_all")):
            tab1, tab2, tab3, tab4 = st.tabs(["§4.1 Overall", "§4.2 By Bias", "§5.2 Coeffs", "Confusion"])
            with tab1: _section_41(results, rlabel)
            with tab2: _section_42(results, rlabel)
            with tab3: _section_52(results, rlabel)
            with tab4: _section_confusion(results, rlabel)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Export
# ─────────────────────────────────────────────────────────────────────────────

def _section_export(df: pd.DataFrame) -> None:
    st.subheader("Export Dataset  (Appendix C)")
    csv_bytes = df.to_csv(index=False).encode()

    manifest = {
        "exported_at":    datetime.now(timezone.utc).isoformat(),
        "n_rows":         len(df),
        "n_participants": int(df["participant_id"].nunique()) if "participant_id" in df.columns else None,
        "columns":        list(df.columns),
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode()

    c1, c2 = st.columns(2)
    with c1:
        if st.download_button(
            f"Download CSV ({len(df)} rows)",
            data=csv_bytes,
            file_name="cogbias_responses.csv",
            mime="text/csv",
            key="dl_csv",
        ):
            try:
                admin_log_audit("export_csv", {"rows": len(df)})
            except Exception:
                pass
    with c2:
        st.download_button(
            "Download manifest.json",
            data=manifest_bytes,
            file_name="manifest.json",
            mime="application/json",
            key="dl_manifest",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Token management
# ─────────────────────────────────────────────────────────────────────────────

def _section_tokens() -> None:
    st.subheader("Token Management")

    with st.form("gen_tokens"):
        n_tokens = st.number_input("Number of tokens to generate", 1, 100, 30, step=1)
        label    = st.text_input("Batch label", "class-2026")
        base_url = st.text_input(
            "App base URL (for survey links)",
            "https://your-app.streamlit.app",
        )
        submitted = st.form_submit_button("Generate tokens")

    if submitted:
        try:
            tokens = admin_generate_tokens(int(n_tokens), label)
            st.success(f"Generated {len(tokens)} tokens.")
            admin_log_audit("token_generate", {"n": len(tokens), "label": label})

            urls = []
            for t in tokens:
                try:
                    url = format_survey_url(base_url, t["id"])
                except Exception:
                    url = f"{base_url.rstrip('/')}?token={t['id']}"
                urls.append({"token_id": t["id"], "survey_url": url})

            st.dataframe(pd.DataFrame(urls), use_container_width=True)
            url_text = "\n".join(r["survey_url"] for r in urls)
            st.download_button(
                "Download token URLs (TXT)",
                data=url_text.encode(),
                file_name="survey_links.txt",
                mime="text/plain",
                key="dl_tokens",
            )
        except Exception as exc:
            st.error(f"Token generation failed: {exc}")

    if st.button("View all tokens"):
        try:
            df_tok = admin_get_tokens()
            if df_tok.empty:
                st.info("No tokens found.")
            else:
                st.dataframe(df_tok, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not fetch tokens: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_admin() -> None:
    """Called by app.py when ?mode=admin is in the URL."""
    if not _check_password():
        return

    st.title("Admin Dashboard — Cognitive Bias Study")

    with st.expander("Health & Sanity", expanded=True):
        _section_health()

    st.divider()

    with st.expander("Collection Toggle"):
        _section_collection()

    st.divider()

    df = _section_data()
    if df is None:
        return

    st.divider()

    with st.expander("Run Analysis"):
        _section_analysis(df)

    st.divider()

    with st.expander("Export (Appendix C)"):
        _section_export(df)

    st.divider()

    with st.expander("Token Management"):
        _section_tokens()
