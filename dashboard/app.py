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
import plotly.express as px

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


@st.cache_data
def load_model_comparison():
    """Cached read of pre-computed model comparison results — never retrains."""
    comparison_df = pd.read_csv("reports/model_comparison.csv")
    final_summary_df = pd.read_csv("reports/final_model_summary.csv")
    return comparison_df, final_summary_df


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


def show_funnel_cohort(df):
    st.header("Funnel & Cohort Analytics")

    st.subheader("Conversion Funnel")
    total_sessions = len(df)
    engaged_sessions = len(df[df["ProductRelated"] > 0])
    converted_sessions = len(df[df["Converted"] == 1])

    funnel_col1, funnel_col2, funnel_col3 = st.columns(3)
    funnel_col1.metric("All Sessions", f"{total_sessions:,}")
    funnel_col2.metric("Viewed Product Pages", f"{engaged_sessions:,}",
                        f"{engaged_sessions/total_sessions*100:.1f}%")
    funnel_col3.metric("Converted", f"{converted_sessions:,}",
                        f"{converted_sessions/total_sessions*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Conversion Rate by Month")
        month_stats = df.groupby("Month").agg(
            sessions=("Converted", "count"),
            conversion_rate=("Converted", "mean")
        ).reset_index()
        month_stats["conversion_rate"] = month_stats["conversion_rate"] * 100

        fig_month = px.bar(
            month_stats, x="Month", y="conversion_rate",
            hover_data=["sessions"],
            labels={"conversion_rate": "Conversion Rate (%)"},
            title="Conversion Rate by Month"
        )
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        st.subheader("Conversion Rate by Visitor Type")
        visitor_stats = df.groupby("VisitorType").agg(
            sessions=("Converted", "count"),
            conversion_rate=("Converted", "mean")
        ).reset_index()
        visitor_stats["conversion_rate"] = visitor_stats["conversion_rate"] * 100

        fig_visitor = px.bar(
            visitor_stats, x="VisitorType", y="conversion_rate",
            hover_data=["sessions"],
            labels={"conversion_rate": "Conversion Rate (%)"},
            title="Conversion Rate by Visitor Type"
        )
        st.plotly_chart(fig_visitor, use_container_width=True)

    st.subheader("PageValues vs ExitRates (Cohort View)")
    sample_df = df.sample(min(2000, len(df)), random_state=CONFIG["random_seed"])
    fig_scatter = px.scatter(
        sample_df, x="ExitRates", y="PageValues",
        color=sample_df["Converted"].map({0: "No Purchase", 1: "Purchase"}),
        opacity=0.5,
        labels={"color": "Outcome"},
        title="PageValues vs ExitRates by Outcome (sampled for performance)",
        color_discrete_map={"No Purchase": "#EF553B", "Purchase": "#636EFA"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def show_model_comparison():
    st.header("Model Comparison")
    st.markdown(
        "Results from offline training and evaluation "
        "(`notebooks/03_modeling.ipynb`) — this dashboard reads saved results "
        "and never retrains models."
    )

    comparison_df, final_summary_df = load_model_comparison()

    st.subheader("Final Model Selection")
    st.dataframe(final_summary_df, hide_index=True, use_container_width=True)

    st.subheader("Six-Model Comparison (class_weight='balanced' baseline)")
    display_df = comparison_df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    st.dataframe(
        display_df.style.format({
            "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "roc_auc": "{:.3f}", "pr_auc": "{:.3f}",
            "cv_pr_auc_mean": "{:.3f}", "cv_pr_auc_std": "{:.3f}",
        }),
        use_container_width=True
    )

    st.subheader("PR-AUC by Model")
    fig_bar = px.bar(
        display_df, x="model", y="pr_auc",
        title="PR-AUC Comparison Across Six Models",
        labels={"pr_auc": "PR-AUC", "model": "Model"}
    )
    fig_bar.add_hline(
        y=0.157, line_dash="dash", line_color="gray",
        annotation_text="No-skill baseline (~15.7% positive class rate)",
        annotation_position="top left"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption(
        "CatBoost was selected as the final model based on leading PR-AUC and recall "
        "on the buying class, then improved further via hyperparameter tuning "
        "(see Final Model Selection table above)."
    )
def show_explainability():
    st.header("Explainability")
    st.markdown(
        "Global and local explanations from SHAP and LIME, generated in "
        "`notebooks/04_xai.ipynb`. See the [model card](../docs/model_card.md) "
        "for the full written analysis."
    )

    st.subheader("Global Feature Importance (SHAP)")
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/shap_global_summary.png", caption="SHAP Beeswarm — direction and magnitude per session")
    with col2:
        st.image("docs/shap_global_bar.png", caption="SHAP Mean |Value| — overall feature importance")

    st.subheader("Local Explanations (Example Sessions)")
    tab1, tab2, tab3 = st.tabs(["Confident Purchase", "Confident No Purchase", "Borderline Case"])

    with tab1:
        st.image("docs/shap_local_purchase_example.png", caption="SHAP — confident Purchase prediction")
        st.image("docs/lime_purchase_example.png", caption="LIME — same session, independent method")

    with tab2:
        st.image("docs/shap_local_no_purchase_example.png", caption="SHAP — confident No Purchase prediction")
        st.image("docs/lime_no_purchase_example.png", caption="LIME — same session, independent method")

    with tab3:
        st.image("docs/shap_local_borderline_example.png", caption="SHAP — borderline prediction near the 0.45 threshold")
        st.image("docs/lime_borderline_example.png", caption="LIME — same session, independent method")

    st.info(
        "SHAP and LIME agree on prediction direction across all examined cases, "
        "but agree on the dominant feature only for the confident Purchase case — "
        "LIME tends to under-weight PageValues for non-Purchase and borderline "
        "sessions. See the model card for full discussion."
    )
@st.cache_data
def load_final_metrics():
    """Cached read of the final model's metrics at the chosen threshold — never recomputed."""
    return pd.read_csv("reports/final_metrics.csv").iloc[0]


def show_performance_metrics():
    st.header("Performance Metrics")
    st.markdown(
        "Final model performance at the chosen decision threshold "
        f"({CONFIG['decision_threshold']}), evaluated on the held-out test set. "
        "See `notebooks/03_modeling.ipynb` for the full threshold-selection analysis."
    )

    metrics = load_final_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    col3.metric("Test PR-AUC", f"{metrics['test_pr_auc']:.3f}")
    col4.metric("CV PR-AUC", f"{metrics['cv_pr_auc_mean']:.3f}")

    st.divider()

    st.subheader("Confusion Matrix")
    cm_data = pd.DataFrame(
        [[int(metrics["true_negatives"]), int(metrics["false_positives"])],
         [int(metrics["false_negatives"]), int(metrics["true_positives"])]],
        index=["Actual: No Purchase", "Actual: Purchase"],
        columns=["Predicted: No Purchase", "Predicted: Purchase"]
    )

    fig_cm = px.imshow(
        cm_data, text_auto=True, color_continuous_scale="Blues",
        labels=dict(color="Count"),
        title=f"Confusion Matrix at threshold = {CONFIG['decision_threshold']}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Business Value")
    st.metric(
        "Net Value at Chosen Threshold",
        f"${metrics['net_business_value']:,.0f}",
        help="Illustrative: $50 per captured conversion, $5 per intervention. "
             "See model card for full assumptions and caveats."
    )

    missed_buyers = int(metrics["false_negatives"])
    total_buyers = int(metrics["false_negatives"]) + int(metrics["true_positives"])
    st.caption(
        f"At this threshold, {missed_buyers} of {total_buyers} actual buyers "
        f"({missed_buyers/total_buyers*100:.1f}%) are missed by the model — "
        f"the recall/precision trade-off chosen to maximize net business value "
        f"under the stated assumptions."
    )
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
        show_funnel_cohort(df)
    elif section == "Model Comparison":
        show_model_comparison()
    elif section == "Explainability":
        show_explainability()
    elif section == "Performance Metrics":
        show_performance_metrics()


if __name__ == "__main__":
    main()