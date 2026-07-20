"""app.py — Streamlit entrypoint. Query-parameter routing.

Routes:
  ?token=<uuid>   → participant survey (run_survey)
  ?mode=admin     → password-gated admin dashboard (run_admin)
  (no params)     → "invitation only" gate page

Run locally:
    streamlit run app.py
"""

import streamlit as st

from tokens import extract_token_from_url

st.set_page_config(
    page_title="Cognitive Bias Study",
    page_icon="\U0001f9e0",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Extract query parameters ───────────────────────────────────────────────────
params = st.query_params

mode  = params.get("mode", "")
token = params.get("token", "")

# ── Route ─────────────────────────────────────────────────────────────────────
if mode == "admin":
    from admin import run_admin
    run_admin()

elif token:
    validated = extract_token_from_url(f"http://x?token={token}")
    if validated is None:
        st.error(
            "The invitation link in your URL is malformed. "
            "Please check the link you received and try again."
        )
    else:
        from survey import run_survey
        run_survey(validated)

else:
    st.title("Cognitive Bias Study")
    st.markdown(
        """
This survey is **by invitation only**.

If you received a personal survey link from the researcher, please use
that link to access the study. Each link is single-use and tied to your
participation record.

If you believe you should have access, please contact the researcher.
        """
    )
