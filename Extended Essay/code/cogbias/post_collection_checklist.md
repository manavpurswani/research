# Post-Collection Checklist

Run after closing collection and before beginning EE write-up analysis.

## Data
- [ ] Close collection in admin → Collection Toggle → Close
- [ ] Admin → Health Check → Run sanity check: no warnings
- [ ] Export CSV (admin → Export → Download CSV) → save as `cogbias_responses_YYYY-MM-DD.csv`
- [ ] Export manifest.json → save alongside CSV
- [ ] Verify CSV row count = (n_participants × 31)

## Quality Flags
- [ ] Admin → Run Analysis → check n_flagged count
- [ ] Decide exclude_flagged policy (document decision in EE §3.4)
- [ ] Attention check pass rate ≥ 80% (if lower, investigate)

## Analysis
- [ ] Run full analysis in admin dashboard (both exclude_flagged=False and True)
- [ ] Download all four figures (confusion matrices, bias bar chart, LR coefficients)
- [ ] Record final LR accuracy and 95% CI for §4.1
- [ ] Record final NN accuracy and 95% CI for §4.1
- [ ] Record per-bias-type breakdown for §4.2
- [ ] Record LR feature coefficients for §5.2

## EE Write-Up
- [ ] §3.3: report n_participants, n_flagged, exclusion decision
- [ ] §4.1: report Wilson 95% CI — if lower bound > 0.50, reject H₁ null
- [ ] §4.2: compare each bias type against majority-vote baseline
- [ ] §5.2: interpret top LR coefficients
- [ ] Appendix C: attach exported CSV and manifest.json
- [ ] Appendix D: attach confusion matrix PNGs
