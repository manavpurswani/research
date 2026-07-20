# CLAUDE.md — Cognitive Bias EE Platform

## What this is
IB Extended Essay ML platform. Streamlit survey + Supabase backend + scikit-learn pipeline.
Research question: "To what extent can ML models predict human decision-making in cognitive bias tasks?"

## Entry points
- `streamlit run app.py` — starts the app
- `python verify_setup.py` — pre-launch readiness check (exits 0 = ready)
- `pytest tests/ -v` — run all 131 tests
- `python pipeline.py sample_data/synthetic_responses.csv` — run pipeline on CSV

## Key constraints (DO NOT CHANGE without updating EE)
- scikit-learn ONLY — no TensorFlow, no PyTorch
- Models: LogisticRegression(max_iter=1000, solver="lbfgs") and MLPClassifier(hidden_layer_sizes=(16,), early_stopping=True)
- GroupKFold k=5, grouped by participant_id, fallback k=3
- Per-fold StandardScaler+OneHotEncoder (no data leakage)
- Wilson 95% CI over all held-out predictions combined
- time.monotonic() for reaction time — never time.time()
- Attention check at position 20 (between trials 19 and 20, 0-indexed)

## Security invariants (NEVER break)
- survey.py MUST NOT import client_admin() or any admin_* function
- Admin uses SUPABASE_SERVICE_ROLE_KEY; survey uses SUPABASE_ANON_KEY
- Password check uses hmac.compare_digest — never ==
- Never commit .streamlit/secrets.toml or any eyJ... JWT string

## File map
- `app.py` — query-param router (?token= → survey, ?mode=admin → admin)
- `survey.py` — participant-facing survey flow
- `admin.py` — password-gated analysis dashboard
- `pipeline.py` — ML pipeline (flag_quality, load_and_prepare, cv_evaluate, run_full)
- `scenarios.py` — 30-scenario generator (deterministic from integer seed)
- `storage.py` — Supabase persistence (dual-client: anon + service_role)
- `tokens.py` — token UUID utilities (pure functions)
- `config.py` — all constants; change here only
- `supabase_setup.sql` — run once in Supabase SQL Editor

## Tests
131 tests across 4 files. All must pass before any commit touching pipeline, scenarios, storage, or tokens.
