"""
FastAPI service: serves purchase-intent predictions with SHAP-based
explanations, using the model/preprocessor/explainer loaded once at startup.
"""

import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config.settings import CONFIG
from src.api.schemas import SessionInput, PredictionResponse, ContributingFeature
from src.api.dependencies import load_model_artifacts, model_state
from src.features.engineer import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model artifacts...")
    load_model_artifacts()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Purchase Intent Prediction API",
    description="Predicts e-commerce session purchase likelihood with SHAP-based explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Readiness check — confirms the model is loaded and ready to serve."""
    if model_state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(session: SessionInput):
    """
    Predicts purchase probability for a single session and returns
    the top SHAP-contributing behaviours behind that prediction.
    """
    try:
        raw_df = pd.DataFrame([session.model_dump()])
        raw_df = engineer_features(raw_df)

        proba = model_state.pipeline.predict_proba(raw_df)[0, 1]
        threshold = CONFIG["decision_threshold"]
        prediction = "Purchase" if proba >= threshold else "No Purchase"

        distance_from_threshold = abs(proba - threshold)
        if distance_from_threshold > 0.3:
            confidence = "High"
        elif distance_from_threshold > 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"

        transformed = model_state.preprocessor.transform(raw_df)
        transformed_dense = np.asarray(transformed.todense()) if hasattr(transformed, "todense") else transformed
        shap_values = model_state.explainer.shap_values(transformed_dense)[0]

        top_idx = np.argsort(np.abs(shap_values))[::-1][:5]
        top_features = [
            ContributingFeature(feature=model_state.feature_names[i], contribution=round(float(shap_values[i]), 4))
            for i in top_idx
        ]

        logger.info("Prediction: %s (proba=%.4f, confidence=%s)", prediction, proba, confidence)

        return PredictionResponse(
            prediction=prediction,
            purchase_probability=round(float(proba), 4),
            confidence=confidence,
            decision_threshold=threshold,
            top_contributing_features=top_features,
        )

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")