-- supabase_setup.sql
-- Run this ONCE in the Supabase SQL editor (Database → SQL Editor → New query).
-- Idempotent: uses IF NOT EXISTS / CREATE OR REPLACE throughout.
-- After running, verify in Table Editor that all four tables exist and RLS is ON.
--
-- Security model
-- ──────────────
--   anon key  (survey.py)   → INSERT only on responses; no direct table access for
--                              tokens / config / admin_audit; may call SECURITY
--                              DEFINER functions verify_token(), consume_token(),
--                              and mark_session_complete().
--   service_role key (admin.py) → full access; bypasses RLS by default in Supabase.
--
-- NEVER put the service_role key in survey.py or expose it client-side.

-- ═══════════════════════════════════════════════════════════════════════════════
-- 1. TABLES
-- ═══════════════════════════════════════════════════════════════════════════════

-- 1a. tokens — one row per survey invitation link
CREATE TABLE IF NOT EXISTS tokens (
    id         uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    label      text        NOT NULL,          -- human-readable batch name, e.g. "class-2026"
    created_at timestamptz NOT NULL DEFAULT now(),
    used_at    timestamptz                    -- NULL = unused; NOT NULL = already consumed
);

-- 1b. responses — one row per trial (real trial or attention check)
--     Inserted immediately after participant answers each question (completed=false).
--     All rows for a participant updated to completed=true at session end.
CREATE TABLE IF NOT EXISTS responses (
    id                     bigserial   PRIMARY KEY,
    participant_id         uuid        NOT NULL,
    trial_index            smallint    NOT NULL,
    bias_type              smallint    NOT NULL CHECK (bias_type IN (1, 2, 3)),
    frame                  smallint    NOT NULL CHECK (frame    IN (0, 1)),
    expected_value_ratio   float       NOT NULL,
    anchor_value           float       NOT NULL DEFAULT 0.0,
    previous_choice        smallint    NOT NULL DEFAULT 0 CHECK (previous_choice IN (0, 1)),
    is_first_trial         smallint    NOT NULL DEFAULT 0 CHECK (is_first_trial  IN (0, 1)),
    reaction_time_ms       integer     NOT NULL,
    choice                 smallint                CHECK (choice IN (0, 1)),
    participant_seed       integer,
    scenario_template_id   text,
    token_id               uuid        REFERENCES tokens(id) ON DELETE SET NULL,
    token_label            text,
    consent_version        text        NOT NULL,
    acknowledged_at        timestamptz,
    -- NULL  -> real trial row
    -- TRUE  -> participant passed the embedded attention check
    -- FALSE -> participant failed the embedded attention check
    attention_check_passed boolean,
    completed              boolean     NOT NULL DEFAULT false,
    created_at             timestamptz NOT NULL DEFAULT now()
);

-- 1c. config — key/value store for admin toggles
--     Key: 'collection_open'  Value: 'true' | 'false'
CREATE TABLE IF NOT EXISTS config (
    key        text        PRIMARY KEY,
    value      text        NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Seed the collection toggle to closed by default
INSERT INTO config (key, value)
VALUES ('collection_open', 'false')
ON CONFLICT (key) DO NOTHING;

-- 1d. admin_audit — append-only log of admin actions
CREATE TABLE IF NOT EXISTS admin_audit (
    id         bigserial   PRIMARY KEY,
    action     text        NOT NULL,   -- e.g. 'login_success', 'export_csv', 'token_generate'
    detail     jsonb,                  -- optional structured payload
    created_at timestamptz NOT NULL DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- 2. INDEXES (performance for 25+ concurrent survey users)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS responses_participant_id_idx
    ON responses (participant_id);

CREATE INDEX IF NOT EXISTS responses_created_at_idx
    ON responses (created_at DESC);

CREATE INDEX IF NOT EXISTS responses_completed_idx
    ON responses (completed) WHERE completed = true;

CREATE INDEX IF NOT EXISTS tokens_used_at_idx
    ON tokens (used_at) WHERE used_at IS NULL;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 3. ROW LEVEL SECURITY
-- ═══════════════════════════════════════════════════════════════════════════════

ALTER TABLE tokens      ENABLE ROW LEVEL SECURITY;
ALTER TABLE responses   ENABLE ROW LEVEL SECURITY;
ALTER TABLE config      ENABLE ROW LEVEL SECURITY;
ALTER TABLE admin_audit ENABLE ROW LEVEL SECURITY;

-- Drop any stale policies before re-creating (idempotent)
DO $$
DECLARE pol RECORD;
BEGIN
    FOR pol IN
        SELECT policyname, tablename
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename IN ('tokens', 'responses', 'config', 'admin_audit')
    LOOP
        EXECUTE format(
            'DROP POLICY IF EXISTS %I ON %I',
            pol.policyname, pol.tablename
        );
    END LOOP;
END $$;

-- responses: anon may INSERT only (no SELECT / UPDATE / DELETE)
CREATE POLICY anon_insert_responses ON responses
    FOR INSERT
    TO anon
    WITH CHECK (true);

-- tokens, config, admin_audit: no anon access at all
-- (access is via SECURITY DEFINER functions only)

-- ═══════════════════════════════════════════════════════════════════════════════
-- 4. SECURITY DEFINER FUNCTIONS
--    Run as the postgres owner -- bypass RLS so the anon client can check/consume
--    a token and mark a session complete without any direct table SELECT/UPDATE.
-- ═══════════════════════════════════════════════════════════════════════════════

-- 4a. verify_token(token_id) -> true if token exists and has not been consumed
CREATE OR REPLACE FUNCTION verify_token(p_token uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM tokens
        WHERE id = p_token
          AND used_at IS NULL
    );
END;
$$;

-- 4b. consume_token(token_id) -> true if token was successfully consumed now
--     Returns false if the token was already consumed or does not exist.
--     Atomic: the UPDATE is a single statement; no race condition.
CREATE OR REPLACE FUNCTION consume_token(p_token uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    UPDATE tokens
    SET used_at = now()
    WHERE id = p_token
      AND used_at IS NULL;
    RETURN FOUND;
END;
$$;

-- 4c. is_collection_open() -> boolean
--     Lets the anon (survey) client check the toggle without SELECT on config.
CREATE OR REPLACE FUNCTION is_collection_open()
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_val text;
BEGIN
    SELECT value INTO v_val FROM config WHERE key = 'collection_open';
    RETURN v_val = 'true';
END;
$$;

GRANT EXECUTE ON FUNCTION is_collection_open() TO anon;

-- 4d. mark_session_complete(participant_id) -> void
--     Called by the survey client after all 31 rows are inserted.
--     Flips completed = true for all rows belonging to this participant.
CREATE OR REPLACE FUNCTION mark_session_complete(p_participant_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    UPDATE responses
    SET completed = true
    WHERE participant_id = p_participant_id
      AND completed = false;
END;
$$;

-- Grant execute on SECURITY DEFINER functions to the anon role
GRANT EXECUTE ON FUNCTION verify_token(uuid)           TO anon;
GRANT EXECUTE ON FUNCTION consume_token(uuid)          TO anon;
GRANT EXECUTE ON FUNCTION mark_session_complete(uuid)  TO anon;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 5. GRANT TABLE PRIVILEGES
--    anon: INSERT on responses only (SELECT/UPDATE/DELETE are blocked by RLS
--    policy absence; explicit GRANT is still required for INSERT to work).
--    authenticated / service_role: Supabase grants full access by default.
-- ═══════════════════════════════════════════════════════════════════════════════

GRANT INSERT ON TABLE responses TO anon;
GRANT USAGE, SELECT ON SEQUENCE responses_id_seq TO anon;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 6. VERIFICATION QUERIES
--    Run these manually after setup to confirm everything is in place.
-- ═══════════════════════════════════════════════════════════════════════════════

-- Check tables exist:
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'public'
--   AND table_name IN ('tokens', 'responses', 'config', 'admin_audit');

-- Check RLS is enabled:
-- SELECT tablename, rowsecurity FROM pg_tables
-- WHERE schemaname = 'public'
--   AND tablename IN ('tokens', 'responses', 'config', 'admin_audit');

-- Check policies:
-- SELECT tablename, policyname, cmd, roles
-- FROM pg_policies WHERE schemaname = 'public';

-- Check functions:
-- SELECT routine_name FROM information_schema.routines
-- WHERE routine_schema = 'public'
--   AND routine_name IN ('verify_token', 'consume_token', 'mark_session_complete');

-- Check config seed:
-- SELECT * FROM config;

-- ═══════════════════════════════════════════════════════════════════════════════
-- END OF SETUP
-- ═══════════════════════════════════════════════════════════════════════════════
