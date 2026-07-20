# Data Dictionary — cogbias_responses.csv

Each row is one trial response from one participant. 31 rows per participant (30 trials + 1 attention check).

| Column | Type | Description |
|--------|------|-------------|
| `trial_index` | int 1–31 | Position in participant's randomised order (31 = attention check) |
| `bias_type` | int | 1=framing, 2=anchoring, 3=loss_aversion |
| `frame` | int | 0=gain, 1=loss (framing only; 1 for LA, 0 for anchoring) |
| `expected_value_ratio` | float | EV(on-screen A) / EV(on-screen B) |
| `anchor_value` | float | Anchor shown to participant; 0.0 for non-anchoring trials |
| `previous_choice` | int | 0/1 choice on preceding trial; 0 for first trial |
| `is_first_trial` | int | 1 if first trial, else 0 |
| `reaction_time_ms` | int | ms from display to click (time.monotonic()) |
| `choice` | int | **Target.** 0=Option A, 1=Option B |
| `participant_id` | uuid str | Unique per participant; groups CV folds |
| `participant_seed` | int | Seed used to generate this participant's scenarios |
| `scenario_template_id` | str | e.g. "F01", "A03", "L07", "ATTN" |
| `token_id` | uuid str | Invitation token UUID |
| `token_label` | str | Batch label when token was generated |
| `consent_version` | str | Consent text version; bump if consent changes |
| `acknowledged_at` | ISO 8601 | Timestamp when participant clicked Start Survey |
| `attention_check_passed` | bool / NULL | NULL=real trial; True/False=attention check row |
| `completed` | bool | True=full session submitted; False=abandoned |
| `created_at` | ISO 8601 | Server timestamp of row insertion (Supabase) |

## Exclusion rules (pipeline.py)

- `completed = False` → excluded from modeling
- `attention_check_passed IS NOT NULL` → excluded from modeling (quality flag only)
- `rt_too_fast`: reaction_time_ms < 200
- `rt_too_slow`: reaction_time_ms > 60000
- `session_too_fast`: total session duration < 90 s
- `attention_failed`: participant's attention check row has attention_check_passed = False
- `any_flag`: union of all above

## Features fed to models (FEATURE_COLS)

`trial_index`, `bias_type`, `frame`, `expected_value_ratio`, `anchor_value`, `previous_choice`, `is_first_trial`, `reaction_time_ms`

`reaction_time_ms` is log1p-transformed. `bias_type` is one-hot encoded. All numeric features StandardScaler-normalised per CV fold.
