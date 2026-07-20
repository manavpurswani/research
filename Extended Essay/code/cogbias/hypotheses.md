# Research Hypotheses

**Research Question:** To what extent can machine learning models predict human decision-making in cognitive bias tasks?

---

## H1 — Overall Predictive Accuracy

**Null (H₀):** ML model accuracy does not differ from chance (50%).
**Alternative (H₁):** At least one ML model achieves accuracy significantly above chance.

*Test:* Wilson 95% CI on held-out predictions. If the lower bound exceeds 0.50, H₀ is rejected.

---

## H2 — Framing Effect Detectability

**Null (H₀):** Frame (gain vs. loss) does not predict participant choice above the majority-vote baseline.
**Alternative (H₂):** Models trained with `frame` as a feature achieve higher accuracy on framing trials than the per-condition majority-vote baseline.

---

## H3 — Loss Aversion Detectability

**Null (H₀):** EV ratio does not predict choice for loss-aversion trials above baseline.
**Alternative (H₃):** Models detect the preference for the certain option in loss-aversion scenarios above baseline.

---

## H4 — Anchoring Detectability

**Null (H₀):** Anchor value does not predict participant choice above baseline.
**Alternative (H₄):** Higher anchor values increase probability of choosing the high-value option above baseline.

---

## H5 — Model Comparison

**Null (H₀):** Logistic Regression and MLP Neural Network do not differ significantly in accuracy.
**Alternative (H₅):** One model outperforms the other (non-overlapping Wilson 95% CIs).

---

## Expected Ranges (from synthetic data validation)

| Metric | Expected Range |
|--------|---------------|
| LR accuracy | 55–70% |
| NN accuracy | 55–65% |
| Framing Δ (gain vs loss) | 25–40 pp |
| Loss-aversion P(certain) | 65–80% |
| Anchoring P(canonical) | 60–70% |

*Near-chance accuracy is a valid and publishable EE finding.*
