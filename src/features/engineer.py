"""
Behavioral feature engineering: derives a small set of
business-interpretable features from raw session data.

Only TotalPages and ProductPageRatio are included — TotalDuration,
AvgTimePerProductPage, and IsReturningVisitor were tested in
notebooks/02_features.ipynb (model-based + permutation importance)
and dropped as redundant or overrated by model-based importance alone.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered behavioral features to the sessions DataFrame.

    Args:
        df: Raw (or validated) sessions DataFrame.

    Returns:
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]

    # Avoid divide-by-zero: replace 0 pages with NaN, then fill result with 0
    df["ProductPageRatio"] = (
        df["ProductRelated"] / df["TotalPages"].replace(0, np.nan)
    ).fillna(0)

    logger.info("Engineered 2 new features")
    return df