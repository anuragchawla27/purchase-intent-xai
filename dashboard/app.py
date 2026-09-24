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
import numpy as np
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


@st.cache_resource
def load_explainer(_pipeline):
    """Cached so the SHAP explainer builds once, not on every form submission."""
    import shap
    classifier = _pipeline.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


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


def show_live_scoring(pipeline):
    st.header("Live Session Scoring")
    st.markdown("Enter session details to get a purchase prediction with explanation.")

    with st.form("scoring_form"):
        col1, col2 = st.columns(2)

        with col1:
            administrative = st.number_input("Administrative pages", min_value=0, value=1)
            administrative_duration = st.number_input("Administrative duration (s)", min_value=0.0, value=14.65)
            informational = st.number_input("Informational pages", min_value=0, value=0)
            informational_duration = st.number_input("Informational duration (s)", min_value=0.0, value=0.0)
            product_related = st.number_input("Product-related pages", min_value=0, value=19)
            product_related_duration = st.number_input("Product-related duration (s)", min_value=0.0, value=283.88)
            bounce_rates = st.slider("Bounce rate", 0.0, 1.0, 0.008)
            exit_rates = st.slider("Exit rate", 0.0, 1.0, 0.042)

        with col2:
            page_values = st.number_input("Page values", min_value=0.0, value=68.58)
            special_day = st.slider("Special day proximity", 0.0, 1.0, 0.0)
            month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June",
                                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=4)
            operating_systems = st.number_input("Operating system (ID)", min_value=1, value=1)
            browser = st.number_input("Browser (ID)", min_value=1, value=1)
            region = st.number_input("Region (ID)", min_value=1, value=1)
            traffic_type = st.number_input("Traffic type (ID)", min_value=1, value=1)
            visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
            weekend = st.checkbox("Weekend session")

        submitted = st.form_submit_button("Score Session")

    if submitted:
        session = pd.DataFrame([{
            "Administrative": administrative, "Administrative_Duration": administrative_duration,
            "Informational": informational, "Informational_Duration": informational_duration,
            "ProductRelated": product_related, "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates, "ExitRates": exit_rates, "PageValues": page_values,
            "SpecialDay": special_day, "Month": month, "OperatingSystems": operating_systems,
            "Browser": browser, "Region": region, "TrafficType": traffic_type,
            "VisitorType": visitor_type, "Weekend": weekend,
        }])
        session = engineer_features(session)

        proba = pipeline.predict_proba(session)[0, 1]
        threshold = CONFIG["decision_threshold"]
        prediction = "Purchase" if proba >= threshold else "No Purchase"

        col_a, col_b = st.columns(2)
        col_a.metric("Prediction", prediction)
        col_b.metric("Purchase Probability", f"{proba:.1%}")

        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(session)
        transformed_dense = np.asarray(transformed.todense()) if hasattr(transformed, "todense") else transformed

        explainer = load_explainer(pipeline)
        shap_values = explainer.shap_values(transformed_dense)[0]
        feature_names = preprocessor.get_feature_names_out()

        top_idx = np.argsort(np.abs(shap_values))[::-1][:5]
        st.subheader("Top Contributing Behaviours")
        contrib_df = pd.DataFrame({
            "Feature": [feature_names[i] for i in top_idx],
            "SHAP Contribution": [round(float(shap_values[i]), 4) for i in top_idx],
        })
        st.dataframe(contrib_df, hide_index=True)


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
        show_live_scoring(pipeline)
    elif section == "Funnel & Cohort Analytics":
        st.info("Coming next step.")
    elif section == "Model Comparison":
        st.info("Coming next step.")
    elif section == "Explainability":
        st.info("Coming next step.")
    elif section == "Performance Metrics":
        st.info("Coming next step.")


if __name__ == "__main__":
    main()