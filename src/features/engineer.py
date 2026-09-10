"""
Behavioral feature engineering: derives a small set of
business-interpretable features from raw session data.
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
    df["TotalDuration"] = (
        df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    )

    # Avoid divide-by-zero: replace 0 pages with NaN, then fill result with 0
    df["AvgTimePerProductPage"] = (
        df["ProductRelated_Duration"] / df["ProductRelated"].replace(0, np.nan)
    ).fillna(0)

    df["ProductPageRatio"] = (
        df["ProductRelated"] / df["TotalPages"].replace(0, np.nan)
    ).fillna(0)

    df["IsReturningVisitor"] = (df["VisitorType"] == "Returning_Visitor").astype(int)

    logger.info("Engineered %d new features", 5)
    return df