Model Card — E-Commerce Purchase Intent Prediction
Model Details
Algorithm: CatBoost Classifier, hyperparameter-tuned via RandomizedSearchCV (20 candidates × 5-fold stratified CV, scored on PR-AUC)
Best hyperparameters: depth=4, learning_rate=0.01, iterations=400, l2_leaf_reg=1, auto_class_weights='Balanced'
Training data: dataset/ecommerce_sessions.csv — 12,000 sessions, 17 input features (+5 engineered, 2 of which were retained), 15.72% positive (purchase) class
Imbalance strategy: class_weight='balanced' — benchmarked against SMOTE and threshold-only tuning; class weighting outperformed SMOTE for this dataset (see notebooks/03_modeling.ipynb, Step 4b)
Preprocessing: scikit-learn ColumnTransformer — numeric features scaled, categorical features (including the four integer-coded ID columns: OperatingSystems, Browser, Region, TrafficType) one-hot encoded with handle_unknown='ignore'
Final feature set: all 18 original features plus TotalPages and ProductPageRatio (three other engineered features — TotalDuration, AvgTimePerProductPage, IsReturningVisitor — were tested and dropped after importance analysis showed them redundant or overrated; see notebooks/02_features.ipynb)
Artifact: models/final_catboost_pipeline.joblib (full pipeline: preprocessing + model, single file)
Intended Use

This model is intended to support a growth or CRO (conversion-rate-optimization) team in identifying e-commerce sessions likely to end in a purchase, so the team can prioritize targeted interventions (e.g., a discount nudge, a free-shipping banner, a remarketing audience) toward sessions where that intervention is most likely to matter. The model outputs a purchase probability per session; sessions with a probability at or above the chosen decision threshold (0.45) are flagged for potential intervention. The threshold was chosen via a net-business-value sweep using illustrative assumptions ($50 value per captured conversion, $5 cost per intervention) — see notebooks/03_modeling.ipynb for the full sweep and reasoning. A real deployment should replace these illustrative figures with the retailer's actual average order value, margin, and intervention cost.

Out-of-Scope Uses
Not for individual-level tracking or identification. This model was trained on anonymized session behavior with no personally identifying fields, and predictions must not be tied to an identified individual.
Not for scoring sessions where PageValues is unavailable or unreliable at prediction time, given the leakage concern described below — a real-time scoring context that cannot supply a trustworthy PageValues signal will see degraded performance not reflected in the metrics below.
Not for any decision with legal, financial, or other significant consequence for an individual — this is a marketing-optimization tool, not a decision system for credit, employment, pricing discrimination, or similar high-stakes contexts.
Performance (test set, n=2,400)
Metric	Value (default 0.5 threshold)	Value (chosen 0.45 threshold)
PR-AUC	0.869	— (threshold-independent)
ROC-AUC	~0.97	— (threshold-independent)
Precision	0.591	0.576
Recall	0.950	0.960
Accuracy	0.913	—

At the chosen threshold (0.45): 362 true positives, 267 false positives, 15 false negatives, 1,756 true negatives. Cross-validated PR-AUC (5-fold, training data): 0.851, close to the test PR-AUC — no evidence of overfitting from hyperparameter tuning.

CatBoost was selected after comparing six algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost) under stratified cross-validation, leading on PR-AUC throughout; full comparison in notebooks/03_modeling.ipynb.

Explainability Summary

Both SHAP (TreeExplainer, exact) and LIME (local linear surrogate) were used to explain individual predictions, per notebooks/04_xai.ipynb. PageValues dominates the model's decisions by a wide margin in both global SHAP importance (mean |SHAP value| ~5–6x the next feature) and in nearly every local explanation examined. ProductRelated, TotalPages, and ExitRates are secondary but meaningful contributors. SHAP and LIME agree on prediction direction across all cases examined, but agree on the dominant feature only for confident Purchase predictions — for confident No-Purchase and borderline sessions, LIME's local linear approximation tends to under-weight PageValues relative to SHAP's exact, interaction-aware computation. Where the two methods disagree, SHAP should be treated as the more faithful explanation, since it is computed exactly from the trained model rather than approximated locally.

Known Limitations
PageValues leakage risk. PageValues is computed as an average of page values seen before conversion, meaning it is structurally influenced by the outcome it's predicting. It is the model's single strongest feature (SHAP mean |value| ~3.1, correlation with Converted of 0.64). This was deliberately kept in the model and documented as a known limitation (see notebooks/01_eda.ipynb) rather than dropped; a PageValues-ablation comparison (with vs. without) was identified as a stretch goal and not performed for this version.
SHAP/LIME disagreement on non-Purchase and borderline cases. As noted above, LIME systematically under-weights PageValues relative to SHAP outside of confident Purchase cases — explanations for ambiguous, near-threshold sessions should lean on SHAP.
Small-sample segment. The VisitorType = Other category has only 133 sessions in the full dataset; conclusions about this segment specifically (or predictions for it) should be treated with lower confidence than for Returning_Visitor or New_Visitor.
Threshold assumptions are illustrative. The $50/$5 value/cost figures used to select the 0.45 decision threshold are placeholders, not figures supplied by a real retailer; the net-value curve was found to be relatively flat between thresholds of ~0.20–0.50, meaning the specific threshold choice is not highly sensitive to moderate errors in these assumptions, but the underlying dollar figures should be replaced with real business data before production use.
Static training data. The model is trained on a single historical snapshot; no data-drift monitoring is implemented in this version (identified as a stretch goal).
Ethical Considerations

The training data is anonymized session-level behavior with no personally identifying fields. The modeling techniques demonstrated here generalize to real user-tracking data, which carries different privacy obligations; any production adaptation of this pipeline must not tie predictions to personally identifying information, and should be accompanied by a clear internal policy on what the model's output may and may not be used for (see Out-of-Scope Uses above).