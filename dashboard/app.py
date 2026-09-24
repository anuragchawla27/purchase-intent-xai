"""
Streamlit dashboard: purchase-intent prediction system.
Reads saved artifacts only — never retrains.
"""

import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)

from src.config.settings import CONFIG
from src.data.loader import load_sessions_data
from src.features.engineer import engineer_features

st.set_page_config(page_title="Purchase Intent Dashboard", layout="wide")


@st.cache_resource
def load_model():
    """Cached so the model loads once per session, not on every interaction."""
    artifact_path = Path(CONFIG["paths"]["models_dir"]) / "final_catboost_pipeline.joblib"
    return joblib.load(artifact_path)


@st.cache_data
def load_data():
    """Cached so the dataset loads once, not re-read from disk on every widget interaction."""
    df = load_sessions_data()
    df = engineer_features(df)
    return df


def main():
    st.title("E-Commerce Purchase Intent Dashboard")

    section = st.sidebar.radio(
        "Navigate",
        ["Overview", "Live Session Scoring", "Funnel & Cohort Analytics",
         "Model Comparison", "Explainability", "Performance Metrics"]
    )

    pipeline = load_model()
    df = load_data()

    if section == "Overview":
        show_overview(df)
    elif section == "Live Session Scoring":
        st.info("Coming next step.")
    elif section == "Funnel & Cohort Analytics":
        st.info("Coming next step.")
    elif section == "Model Comparison":
        st.info("Coming next step.")
    elif section == "Explainability":
        st.info("Coming next step.")
    elif section == "Performance Metrics":
        st.info("Coming next step.")


def show_overview(df):
    st.header("Project Overview")
    st.markdown("""
    This dashboard supports a growth/CRO team in identifying e-commerce sessions
    likely to end in a purchase, using a tuned CatBoost model with SHAP-based
    explanations. See the [model card](../docs/model_card.md) for full details.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", f"{len(df):,}")
    col2.metric("Conversion Rate", f"{df['Converted'].mean()*100:.2f}%")
    col3.metric("Decision Threshold", CONFIG["decision_threshold"])
    col4.metric("Model", "CatBoost (tuned)")

    st.subheader("Sample of Raw Data")
    st.dataframe(df.head(20))


if __name__ == "__main__":
    main()