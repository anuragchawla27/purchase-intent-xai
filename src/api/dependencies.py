"""
Shared application state: the fitted pipeline, its preprocessor/classifier
components, and the SHAP explainer, all loaded once at startup and reused
across every request.
"""

import logging
import joblib
import shap
from pathlib import Path

from src.config.settings import CONFIG

logger = logging.getLogger(__name__)


class ModelState:
    """Holds everything the API needs, loaded once."""
    pipeline = None
    preprocessor = None
    classifier = None
    explainer = None
    feature_names = None


model_state = ModelState()


def load_model_artifacts():
    """
    Loads the trained pipeline and builds the SHAP explainer.
    Called once at app startup via the FastAPI lifespan handler.
    """
    artifact_path = Path(CONFIG["paths"]["models_dir"]) / "final_catboost_pipeline.joblib"

    if not artifact_path.exists():
        logger.error("Model artifact not found at %s", artifact_path)
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    model_state.pipeline = joblib.load(artifact_path)
    model_state.preprocessor = model_state.pipeline.named_steps["preprocessor"]
    model_state.classifier = model_state.pipeline.named_steps["classifier"]
    model_state.feature_names = model_state.preprocessor.get_feature_names_out()
    model_state.explainer = shap.TreeExplainer(model_state.classifier)

    logger.info("Model, preprocessor, and SHAP explainer loaded successfully.")