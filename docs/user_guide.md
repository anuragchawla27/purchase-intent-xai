# User Guide

For a growth/CRO team member using the dashboard or API day to day — not a
setup guide (see the README's Quick Start for that).

---

## Using the dashboard

Open the dashboard (`http://localhost:8501` once running) and use the
sidebar to navigate between six sections.

### Overview
Headline numbers: total sessions, overall conversion rate, the current
decision threshold, and which model is in production. Start here to
orient yourself.

### Live Session Scoring
Score a single session by hand. Fill in the form (page-visit counts,
durations, bounce/exit rates, month, visitor type, etc.) and click
**Score Session**. You'll get:
- **Prediction** — Purchase or No Purchase
- **Purchase Probability** — the model's raw confidence (0–100%)
- **Top Contributing Behaviours** — the five features that most drove
  *this specific* prediction, with a positive or negative value showing
  which direction each one pushed

Use this to sanity-check the model on a hypothetical session, or to
understand why a particular real session was flagged.

### Funnel & Cohort Analytics
How sessions move from browsing to converting, broken down by month and
visitor type, plus a scatter view of how page value and exit rate relate to
outcome. Use this to spot seasonal patterns (e.g. which months convert best
relative to their traffic) or to compare new vs. returning visitors.

### Model Comparison
How the final model (CatBoost) stacks up against five alternatives that
were tried and set aside. This reflects a one-time offline comparison — it
does not retrain live.

### Explainability
The global and per-session SHAP/LIME analysis behind the model's decisions,
including a comparison of the two methods on three example sessions. Read
this if you need to explain to a stakeholder *why* the model behaves the
way it does, beyond a single prediction.

### Performance Metrics
The final model's precision, recall, and PR-AUC at the chosen decision
threshold, a confusion matrix, and the estimated net business value of
operating at that threshold (see the model card for the assumptions behind
the dollar figures).

---

## Using the API

For integrating predictions into another system (e.g. a CRM or marketing
tool) rather than viewing them in the dashboard.

1. Send a `POST` request to `/predict` with a session's data as JSON (see
   `docs/api_reference.md` for the full field list and an example).
2. You'll get back a prediction, a probability, a confidence level, and the
   top contributing behaviours for that specific session.
3. Use `GET /health` to check the service is up before sending real traffic
   to it (e.g. in a monitoring script).

The interactive docs at `/docs` let you try requests directly from your
browser without writing any code — useful for a quick manual check.

---

## Interpreting a prediction

- **High confidence + Purchase** — the session shows strong buying signals
  (commonly high `PageValues`, low `ExitRates`, more product-page
  engagement). Safe to act on.
- **Low confidence, near the 0.45 threshold** — a genuinely uncertain
  session. The model's explanation is less reliable here (SHAP and LIME can
  disagree on borderline cases — see the model card), so treat these as a
  judgment call rather than a firm signal.
- **`PageValues` dominating the explanation** — expected and consistent
  across the whole project, but remember this feature carries a documented
  leakage caveat (it partly reflects behaviour close to checkout). Don't
  treat it as something you can influence directly with an intervention.

---

## Where to look for more detail

| Question | See |
|---|---|
| How was the model chosen and tuned? | `docs/model_card.md`, `notebooks/03_modeling.ipynb` |
| What do the SHAP/LIME visuals mean? | `docs/model_card.md`, `notebooks/04_xai.ipynb` |
| Full API request/response spec | `docs/api_reference.md` |
| How is the system deployed? | `docs/architecture.md`, `docs/aws_deployment.md` |
| Known limitations | `docs/model_card.md` |
