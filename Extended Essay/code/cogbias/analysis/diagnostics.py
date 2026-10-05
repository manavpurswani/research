"""diagnostics.py: post-hoc diagnostics for the LLM-responder results.

Reports: previous_choice / is_first_trial, P(B) by frame, per-bias ROC AUC and
balanced accuracy, grouping by model name, quality-flag breakdown, majority
baseline per fold, and anchoring/loss-aversion stimulus checks.

Usage (from anywhere):
    python analysis/diagnostics.py

Reads data/llm_responses.csv. Writes nothing; prints to stdout.
Needs the pinned versions in requirements.txt.
"""
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1])
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import pipeline as P
from config import FEATURE_COLS, BIAS_TYPE_LABEL

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
raw = pd.read_csv("data/llm_responses.csv")
df, y, groups, meta = P.load_and_prepare(raw)
bt = df["bias_type"].to_numpy()
print(f"modeling rows={len(df)} participants={df.participant_id.nunique()} P(B)={y.mean():.4f}")

# ---------------------------------------------------------------- (1)
print("\n=== (1) previous_choice / is_first_trial ===")
print("FEATURE_COLS:", FEATURE_COLS)
print("is_first_trial in FEATURE_COLS:", "is_first_trial" in FEATURE_COLS)
print("previous_choice value_counts (raw csv):\n", raw["previous_choice"].value_counts(dropna=False))
print("previous_choice value_counts (modeling df):\n", df["previous_choice"].value_counts(dropna=False))
print("is_first_trial value_counts:\n", df["is_first_trial"].value_counts())
# what previous_choice SHOULD have been
d2 = raw.sort_values(["participant_id", "trial_index"]).copy()
d2["true_prev"] = d2.groupby("participant_id")["choice"].shift(1)
nz = d2.dropna(subset=["true_prev"])
print("rows where true previous choice == 1:", int((nz.true_prev == 1).sum()), "of", len(nz),
      "| stored previous_choice==1:", int((nz.previous_choice == 1).sum()))

# ---------------------------------------------------------------- (2)
print("\n=== (2) P(choice=B) on framing trials by frame ===")
fr = df[df.bias_type == 1]
print("overall:\n", fr.groupby("frame")["choice"].agg(["mean", "count"]).round(3))
t = fr.pivot_table(index="model_name", columns="frame", values="choice", aggfunc="mean").round(3)
t["gain-loss_gap"] = (t[1] - t[0]).round(3)
n = fr.pivot_table(index="model_name", columns="frame", values="choice", aggfunc="count")
t["n_gain"] = n[0]; t["n_loss"] = n[1]
t.columns = ["P(B|gain)", "P(B|loss)", "loss-gain", "n_gain", "n_loss"]
print(t.to_string())
# per-participant frame effect, paired
pp = fr.pivot_table(index="participant_id", columns="frame", values="choice", aggfunc="mean")
print("participants with P(B|loss)>P(B|gain):", int((pp[1] > pp[0]).sum()), " <:", int((pp[1] < pp[0]).sum()),
      " =:", int((pp[1] == pp[0]).sum()))

# ---------------------------------------------------------------- (3)/(4) CV machinery
def oof(df, y, groups, k, fit_fn):
    gkf = GroupKFold(n_splits=k)
    X_raw = df[FEATURE_COLS]
    prob = np.full(len(y), np.nan); pred = np.full(len(y), -1); fold = np.full(len(y), -1)
    for i, (tr, te) in enumerate(gkf.split(X_raw, y, groups)):
        pre = P._build_preprocessor()
        Xtr = pre.fit_transform(X_raw.iloc[tr]).astype(np.float32)
        Xte = pre.transform(X_raw.iloc[te]).astype(np.float32)
        clf = fit_fn(Xtr, y[tr])
        prob[te] = clf.predict_proba(Xte)[:, 1]; pred[te] = clf.predict(Xte); fold[te] = i
    return prob, pred, fold

def safe_auc(yt, p):
    return roc_auc_score(yt, p) if len(np.unique(yt)) == 2 else np.nan

def report(groups, k, title):
    print(f"\n--- {title} (k={k}, n_groups={len(np.unique(groups))}) ---")
    rows = []
    for mname, fn in [("LR", P.fit_logreg), ("MLP", P.fit_mlp)]:
        prob, pred, fold = oof(df, y, groups, k, fn)
        for label, mask in [("ALL", np.ones(len(y), bool))] + [(BIAS_TYPE_LABEL[b], bt == b) for b in (1, 2, 3)]:
            yt, pp_, pr = y[mask], prob[mask], pred[mask]
            fa = [safe_auc(y[mask & (fold == f)], prob[mask & (fold == f)]) for f in range(k)]
            fb = [balanced_accuracy_score(y[mask & (fold == f)], pred[mask & (fold == f)])
                  for f in range(k) if len(np.unique(y[mask & (fold == f)])) == 2]
            rows.append(dict(model=mname, bias=label, n=int(mask.sum()),
                             acc=accuracy_score(yt, pr), bal_acc=balanced_accuracy_score(yt, pr),
                             auc=safe_auc(yt, pp_), auc_foldmean=np.nanmean(fa), auc_foldsd=np.nanstd(fa),
                             balacc_foldmean=np.mean(fb), pred_B_rate=pr.mean(), true_B_rate=yt.mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== (3) ROC AUC / balanced accuracy, GroupKFold by participant ===")
report(groups, 5, "groups = participant_id")

print("\n=== (4) groups = model_name ===")
mg = df["model_name"].to_numpy()
report(mg, 5, "groups = model_name, k=5")
report(mg, 10, "groups = model_name, leave-one-model-out k=10")
for k in (5, 10):
    for mname, fn in [("LR", P.fit_logreg), ("MLP", P.fit_mlp)]:
        r = P.cv_evaluate(df, y, mg, fn, k, bt)
        print(f"\ncv_evaluate by model_name k={k} {mname}: acc {r['accuracy']['mean']:.3f}±{r['accuracy']['std']:.3f}"
              f"  majority {r['baselines']['majority_class']['mean']:.3f}  per-cond {r['baselines']['per_condition']['mean']:.3f}")
        for b in (1, 2, 3):
            print(f"   {BIAS_TYPE_LABEL[b]:<14} acc {r['by_bias_type'][b]['accuracy']['mean']:.3f}"
                  f"  per-cond BL {r['baselines']['per_condition_by_bias'].get(b, {}).get('mean', float('nan')):.3f}")

# ---------------------------------------------------------------- (5)
print("\n=== (5) flagged ===")
q = P.flag_quality(raw)
g = q.groupby("participant_id").agg(model=("model_name", "first"), temp=("temperature", "first"),
                                    rt_fast=("rt_too_fast", "any"), rt_slow=("rt_too_slow", "any"),
                                    sess_fast=("session_too_fast", "any"), attn=("attention_failed", "any"),
                                    any_flag=("any_flag", "any"), n_fast_trials=("rt_too_fast", "sum"),
                                    min_rt=("reaction_time_ms", "min"))
print("flagged participants:", int(g.any_flag.sum()), "of", len(g))
print("by reason (not exclusive):", {c: int(g[c].sum()) for c in ["rt_fast", "rt_slow", "sess_fast", "attn"]})
print("flagged ONLY by rt_fast:", int((g.rt_fast & ~g.sess_fast & ~g.attn & ~g.rt_slow).sum()),
      "| ONLY by sess_fast:", int((g.sess_fast & ~g.rt_fast & ~g.attn & ~g.rt_slow).sum()))
print("attention rows present:", int(raw["attention_check_passed"].notna().sum()))
print("flagged per model:\n", g.groupby("model").any_flag.agg(["sum", "count"]).T.to_string())
print("reaction_time_ms describe (raw):\n", raw["reaction_time_ms"].describe().round(0).to_string())
sd = q.assign(ts=pd.to_datetime(q.created_at, utc=True)).groupby("participant_id").ts.agg(lambda s: (s.max() - s.min()).total_seconds())
print("session duration secs describe:\n", sd.describe().round(1).to_string())
print("n trials with RT<200:", int((raw.reaction_time_ms < 200).sum()), "of", len(raw))
# model-level view of surviving
surv = g[~g.any_flag]
print("surviving participants per model:\n", surv.groupby("model").size().to_string())

# ---------------------------------------------------------------- (6)
print("\n=== (6) majority-class baseline, per fold ===")
gkf = GroupKFold(n_splits=5)
for i, (tr, te) in enumerate(gkf.split(df[FEATURE_COLS], y, groups)):
    maj = int(np.bincount(y[tr]).argmax())
    print(f"fold {i}: train P(B)={y[tr].mean():.3f} -> predict {maj}; test P(B)={y[te].mean():.3f}; acc={np.mean(y[te]==maj):.3f}")

# ---------------------------------------------------------------- (7)/(8)
print("\n=== (8) anchoring ranges and what anchor_value identifies ===")
an = df[df.bias_type == 2].copy()
rng = an.groupby("scenario_template_id").anchor_value.agg(["min", "max", "mean", "count"]).round(2)
print(rng.to_string())
# eta^2 of template on anchor_value
tot = ((an.anchor_value - an.anchor_value.mean()) ** 2).sum()
wit = an.groupby("scenario_template_id").anchor_value.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
print(f"eta^2 (a) anchoring trials only, n={len(an)}: {1 - wit/tot:.3f}")
_tot_all = ((df.anchor_value - df.anchor_value.mean()) ** 2).sum()
_bet_all = df.groupby("scenario_template_id").anchor_value.apply(
    lambda s: len(s) * (s.mean() - df.anchor_value.mean()) ** 2).sum()
print(f"eta^2 (b) all rows, n={len(df)}: {_bet_all/_tot_all:.3f}  (anchor_value is 0.0 off anchoring trials)")
from config import *
import scenarios as S
tpl = {t["template_id"]: t for t in S._ANCHORING_TEMPLATES}
an["rel"] = an.apply(lambda r: (r.anchor_value - tpl[r.scenario_template_id]["anchor_min"]) /
                               (tpl[r.scenario_template_id]["anchor_max"] - tpl[r.scenario_template_id]["anchor_min"]), axis=1)
print("corr(raw anchor_value, choice)      :", round(np.corrcoef(an.anchor_value, an.choice)[0, 1], 3))
print("corr(within-template anchor pos, choice):", round(np.corrcoef(an.rel, an.choice)[0, 1], 3))
an["rel_hi"] = an.rel >= 0.5
print("P(B | anchor in upper half of its template range):\n", an.groupby("rel_hi").choice.agg(["mean", "count"]).round(3).to_string())
print("P(B) by anchoring template:\n", an.groupby("scenario_template_id").choice.mean().round(3).to_string())

print("\n=== (7) loss aversion ===")
# ev_ratio does NOT identify which option is the certain one (range 0.85-1.18 and a
# random A/B swap), so rebuild each scenario from its responder_seed and read the text.
from scenarios import generate_participant_scenarios as _gen
_rows = []
for seed, g_ in raw[raw.invalid_response.astype(str) != "True"].groupby("responder_seed"):
    _sc = {x["trial_index"]: x for x in _gen(int(seed))}
    for _, r in g_.iterrows():
        x = _sc[int(r.trial_index)]
        _rows.append(dict(tpl=x["scenario_template_id"], bt=x["bias_type"], ev_gen=x["expected_value_ratio"],
                          ev_csv=r.expected_value_ratio, anch_gen=x["anchor_value"], anch_csv=r.anchor_value,
                          tpl_csv=r.scenario_template_id, choice=r.choice, canon=x["canonical_biased_choice"],
                          a_txt=x["option_a_text"], b_txt=x["option_b_text"], model=r.model_name))
d = pd.DataFrame(_rows)
print("regenerated scenarios match CSV: template", (d.tpl == d.tpl_csv).mean(),
      "| ev_ratio", np.isclose(d.ev_csv, d.ev_gen).mean(), "| anchor", np.isclose(d.anch_csv, d.anch_gen).mean())
la = d[d.bt == 3].copy()
la["a_is_certain"] = la.a_txt.str.contains("guaranteed|Guaranteed|Exactly|Accept|Pay a", regex=True) & ~la.a_txt.str.contains("chance")
la["certain"] = np.where(la.a_is_certain, 1 - la.choice, la.choice)
print("frame on loss-aversion / anchoring / framing rows:",
      df[df.bias_type == 3].frame.value_counts().to_dict(), df[df.bias_type == 2].frame.value_counts().to_dict(),
      df[df.bias_type == 1].frame.value_counts().to_dict())
print(f"ev_ratio range on loss-aversion rows: {la.ev_gen.min()} to {la.ev_gen.max()}; "
      f"agreement of (ev_ratio<1) with 'A is certain': {((la.ev_gen < 1) == la.a_is_certain).mean():.3f}")
print(f"P(choose certain option) overall: {la.certain.mean():.3f}  (n={len(la)});"
      f" equals P(choice == canonical): {(la.choice == la.canon).mean():.3f}")
print("by template:\n", la.groupby("tpl").certain.mean().round(3).to_string())
print("by model:\n", la.groupby("model").certain.mean().round(3).to_string())
la["ev_c_over_r"] = np.where(la.a_is_certain, la.ev_gen, 1 / la.ev_gen)
la["band"] = pd.cut(la.ev_c_over_r, [0, 0.95, 1.05, 2])
print("P(certain) by EV(certain)/EV(risky) band:\n", la.groupby("band", observed=True).certain.agg(["mean", "count"]).round(3).to_string())
