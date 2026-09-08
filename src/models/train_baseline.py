"""
Thin-slice baseline: trivial model trained end-to-end and saved.
Not meant to be a good model — meant to prove the pipeline works.
"""

import logging
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.config.settings import CONFIG
from src.data.loader import load_sessions_data
from src.preprocessing.pipeline import build_preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def train_baseline():
    df = load_sessions_data()

    target = CONFIG["target_column"]
    categorical_columns = CONFIG["categorical_columns"]
    numeric_columns = [
        col for col in df.columns
        if col not in categorical_columns + [target]
    ]

    X = df[numeric_columns + categorical_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["split"]["test_size"],
        stratify=y if CONFIG["split"]["stratify"] else None,
        random_state=CONFIG["random_seed"],
    )

    preprocessor = build_preprocessing_pipeline(numeric_columns, categorical_columns)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=CONFIG["random_seed"])),
    ])

    model.fit(X_train, y_train)
    logger.info("Baseline model trained. Test accuracy: %.3f", model.score(X_test, y_test))

    models_dir = Path(CONFIG["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / "baseline_model.joblib"
    joblib.dump(model, artifact_path)
    logger.info("Model artifact saved to %s", artifact_path)

    return model


if __name__ == "__main__":
    train_baseline()