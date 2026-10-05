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
print(f"eta^2 (share of anchor_value variance explained by template id): {1 - wit/tot:.3f}")
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
la = df[df.bias_type == 3]
print("frame values on loss-aversion rows:", la.frame.value_counts().to_dict(),
      "| frame on anchoring rows:", df[df.bias_type == 2].frame.value_counts().to_dict(),
      "| framing rows:", df[df.bias_type == 1].frame.value_counts().to_dict())
print("ev_ratio<1 vs >1 on loss-aversion rows (A is certain iff <1):", (la.expected_value_ratio < 1).value_counts().to_dict())
print("P(B|ev_ratio>1) [B=certain]:", round(la[la.expected_value_ratio > 1].choice.mean(), 3),
      " P(B|ev_ratio<1) [B=risky]:", round(la[la.expected_value_ratio < 1].choice.mean(), 3))
print("P(choose CERTAIN option) overall:",
      round(np.where(la.expected_value_ratio < 1, 1 - la.choice, la.choice).mean(), 3))
print("by template, P(choose certain):")
la = la.assign(certain=np.where(la.expected_value_ratio < 1, 1 - la.choice, la.choice))
print(la.groupby("scenario_template_id").certain.mean().round(3).to_string())
