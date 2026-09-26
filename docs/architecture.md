# System Architecture & Workflow

Two diagrams: the overall system architecture (§6 of the project brief) and
the workflow a single session follows from raw data to an explained
prediction. Both render directly in GitHub's markdown viewer.

---

## 1. System architecture

Four layers — data, ML, serving, MLOps — with configuration and logging
cutting across all of them via `src/config/config.yaml` and Python's
`logging` module.

```mermaid
flowchart TB
    subgraph DATA["Data Layer"]
        CSV["dataset/ecommerce_sessions.csv"]
        LOADER["src/data/loader.py<br/>validated ingestion"]
        FEAT["src/features/engineer.py<br/>TotalPages, ProductPageRatio"]
        PREP["src/preprocessing/pipeline.py<br/>ColumnTransformer"]
        CSV --> LOADER --> FEAT --> PREP
    end

    subgraph ML["ML Layer"]
        TRAIN["src/models/trainer.py<br/>6 algorithms, stratified CV"]
        TUNE["RandomizedSearchCV<br/>hyperparameter tuning"]
        XAI["SHAP + LIME<br/>explainability"]
        MLF[("MLflow<br/>experiment tracking +<br/>model registry")]
        ARTIFACT[("models/final_catboost_pipeline.joblib")]
        TRAIN --> TUNE --> ARTIFACT
        TUNE -.logs.-> MLF
        ARTIFACT --> XAI
    end

    subgraph SERVING["Serving Layer"]
        API["FastAPI<br/>/predict /health /docs"]
        DASH["Streamlit Dashboard<br/>6 sections"]
        ARTIFACT --> API
        ARTIFACT --> DASH
        XAI -.explanations.-> API
        XAI -.explanations.-> DASH
    end

    subgraph MLOPS["MLOps Layer"]
        DOCKER["Docker + docker-compose"]
        CI["GitHub Actions CI<br/>lint -> test -> build"]
        AWS["AWS-ready config<br/>(ECR/ECS, documented)"]
        API --> DOCKER
        DASH --> DOCKER
        DOCKER --> CI
        CI --> AWS
    end

    PREP --> TRAIN

    CONFIG["src/config/config.yaml<br/>(paths, seed, threshold,<br/>categorical columns)"]
    LOGGING["Python logging<br/>(structured, no print())"]
    CONFIG -.-> DATA
    CONFIG -.-> ML
    CONFIG -.-> SERVING
    LOGGING -.-> DATA
    LOGGING -.-> ML
    LOGGING -.-> SERVING
```

---

## 2. Workflow: one session, start to finish

Traces a single session from a raw CSV row to a scored, explained response —
the same path whether it's served through the API or the dashboard.

```mermaid
sequenceDiagram
    participant U as User / Client
    participant L as Data Loader
    participant F as Feature Engineering
    participant P as Preprocessing<br/>(ColumnTransformer)
    participant M as CatBoost Model
    participant X as SHAP Explainer
    participant R as Response

    U->>L: Raw session (JSON or form input)
    L->>F: Validated raw features
    F->>F: + TotalPages, ProductPageRatio
    F->>P: Engineered features
    P->>P: Scale numeric, one-hot categorical<br/>(unknown categories -> ignored, not errored)
    P->>M: Transformed feature vector
    M->>R: Purchase probability
    R->>R: Compare to threshold (0.45)<br/>-> Purchase / No Purchase + confidence
    P->>X: Transformed feature vector
    X->>R: Top 5 SHAP-contributing behaviours
    R->>U: prediction, probability, confidence,<br/>threshold, top_contributing_features
```

---

## Notes

- Both diagrams describe the *actual* implemented system — they are not
  aspirational. The dashed lines in the architecture diagram (config,
  logging, MLflow) indicate cross-cutting concerns rather than a
  data-flow direction.
- The workflow diagram applies identically to the FastAPI `/predict`
  endpoint and the Streamlit "Live Session Scoring" section — both call the
  same `src/` code (`engineer_features`, the saved pipeline, and the SHAP
  explainer), which is why their outputs are verified to match exactly for
  the same input (see `docs/model_card.md`).
