"""
Model training and evaluation: a common interface across all six algorithms,
so adding or swapping a model is a one-line registry change, not new code.
"""

import logging

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config.settings import CONFIG

logger = logging.getLogger(__name__)

SEED = CONFIG["random_seed"]


def get_model_registry() -> dict:
    """
    Returns a dict of model_name -> unfitted estimator instance.
    class_weight='balanced' (or the boosting-library equivalent) is used
    as the Step 4a baseline imbalance strategy for every model.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=SEED,
            # scale_pos_weight set at train time, once we know the class ratio
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced", random_state=SEED, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            auto_class_weights="Balanced", random_state=SEED, verbose=0
        ),
    }


def train_and_evaluate(
    name: str, model, preprocessor, X_train, X_test, y_train, y_test
) -> dict:
    """
    Fit a model (wrapped with the shared preprocessor) and compute the
    full evaluation metric suite the PRD requires.
    """
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    # XGBoost needs scale_pos_weight set explicitly (no class_weight param)
    if name == "XGBoost":
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        pipeline.named_steps["classifier"].set_params(scale_pos_weight=ratio)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="average_precision"
    )

    results = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "cv_pr_auc_mean": cv_scores.mean(),
        "cv_pr_auc_std": cv_scores.std(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info(
        "%s — PR-AUC: %.4f | Recall: %.4f | Accuracy: %.4f",
        name,
        results["pr_auc"],
        results["recall"],
        results["accuracy"],
    )
    return results, pipeline


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_with_strategy(
    name: str, base_model, preprocessor, strategy: str, X_train, X_test, y_train, y_test
) -> dict:
    """
    Train one model under a specified imbalance strategy:
      - 'class_weight': uses the model's built-in class weighting (baseline)
      - 'smote': SMOTE oversampling applied inside the pipeline, train-fold only
      - 'threshold_only': no resampling/weighting, rely purely on threshold tuning

    Returns the same metric suite as train_and_evaluate, plus the strategy name.
    """
    if strategy == "smote":
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=CONFIG["random_seed"])),
                ("classifier", base_model),
            ]
        )
    else:
        # 'class_weight' and 'threshold_only' both use a plain Pipeline;
        # the difference is whether base_model itself has class_weight set
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", base_model)]
        )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "model": name,
        "strategy": strategy,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    logger.info(
        "%s [%s] — PR-AUC: %.4f | Recall: %.4f",
        name,
        strategy,
        results["pr_auc"],
        results["recall"],
    )
    return results, pipeline
