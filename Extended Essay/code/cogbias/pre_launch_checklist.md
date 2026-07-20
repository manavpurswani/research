# Pre-Launch Checklist

Work through this list top-to-bottom before opening collection.

## Infrastructure
- [ ] Supabase project created and `supabase_setup.sql` run in SQL Editor
- [ ] RLS enabled on all four tables (verify in Supabase → Table Editor → RLS)
- [ ] All four SECURITY DEFINER functions present (verify_token, consume_token, is_collection_open, mark_session_complete)
- [ ] `.streamlit/secrets.toml` filled in with SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY, ADMIN_PASSWORD
- [ ] ADMIN_PASSWORD is at least 16 characters and not guessable

## Code
- [ ] `python verify_setup.py` exits 0 (all checks green)
- [ ] `pytest tests/ -v` → 131/131 passed
- [ ] `pre-commit install` run; `pre-commit run --all-files` passes

## Tokens
- [ ] Tokens generated in admin → Token Management (one per expected participant)
- [ ] Survey URLs copied and ready to distribute (download survey_links.txt)
- [ ] Test token: paste one URL in a private browser, complete the survey end-to-end
- [ ] After test: delete test participant rows from Supabase, reset token used_at = NULL

## Deployment
- [ ] App deployed to Streamlit Cloud (or equivalent)
- [ ] Deployed URL verified: bare URL shows "invitation only" page
- [ ] `?mode=admin` shows admin login
- [ ] One survey token URL opens the consent screen

## Final
- [ ] collection_open = false (admin shows "closed" state before you're ready)
- [ ] Open collection only when you're ready for participants: admin → Collection Toggle → Open
- [ ] Note start date/time for your records
