"""
app.py — Streamlit entrypoint.

Routes between the participant survey and the password-gated admin dashboard
using the sidebar selector.

Run locally:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Cognitive Bias Study",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Sidebar routing ────────────────────────────────────────────────────────────
# The sidebar is collapsed by default so participants see only the survey.
# Researchers open the sidebar and select Admin to access the dashboard.

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select page",
        options=["Survey", "Admin"],
        index=0,
        label_visibility="collapsed",
    )

# ── Page dispatch ──────────────────────────────────────────────────────────────
if page == "Survey":
    from survey import run_survey
    run_survey()
else:
    from admin import run_admin
    run_admin()
