# EMERGENCY RUNBOOK

## Participant can't access survey
1. Verify URL contains `?token=<uuid>` — send exact link from admin → Token Management.
2. If token `used_at` is not null → token consumed. Generate a new one in admin.
3. If `collection_open = false` → open it in admin → Collection Toggle.

## Participant got a DB error mid-survey
1. Ask for their completion code (participant_id shown on finish screen).
2. In Supabase SQL Editor: `SELECT * FROM responses WHERE participant_id='<uuid>';`
3. If rows exist with `completed=false`: `SELECT mark_session_complete('<uuid>');`
4. Their data is usable — `pipeline.load_and_prepare()` filters `completed==True`.

## Admin panel locked out
Wait 5 minutes from the last failed attempt. The lockout is session-scoped and timestamp-based.

## Supabase down during survey
Participants see an error on screen. Ask them to note their completion code.
Incomplete rows (`completed=false`) are excluded from analysis automatically.

## Accidentally closed collection mid-survey
Re-open immediately in admin. In-flight sessions are unaffected (they passed the gate at consent time).

## JWT / secret accidentally committed
1. Rotate in Supabase dashboard → Settings → API → Regenerate keys.
2. Update `.streamlit/secrets.toml`.
3. Redeploy app.
4. Check `.pre-commit-config.yaml` hook is installed (`pre-commit install`).

## Reset a participant (test run or duplicate)
```sql
DELETE FROM responses WHERE participant_id = '<uuid>';
UPDATE tokens SET used_at = NULL WHERE id = '<token-uuid>';
```
