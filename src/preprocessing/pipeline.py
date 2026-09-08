"""
Preprocessing pipeline: encodes categoricals, scales numerics.
Thin-slice version — imbalance strategy and feature engineering come later.
"""

import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import CONFIG

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(numeric_columns: list, categorical_columns: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numeric columns and
    one-hot encodes categorical columns.

    Args:
        numeric_columns: list of numeric feature column names.
        categorical_columns: list of categorical feature column names
                              (should come from CONFIG["categorical_columns"]).

    Returns:
        An unfitted ColumnTransformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    logger.info(
        "Built preprocessing pipeline: %d numeric, %d categorical columns",
        len(numeric_columns), len(categorical_columns)
    )
    return preprocessor