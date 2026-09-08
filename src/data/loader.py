"""
Data ingestion layer: loads and validates the raw sessions CSV.
"""

import logging
from pathlib import Path
import pandas as pd

from src.config.settings import CONFIG

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Converted",
]


def load_sessions_data(path: str = None) -> pd.DataFrame:
    """
    Load the e-commerce sessions dataset and validate its schema.

    Args:
        path: Optional override for the CSV path. Defaults to the path
              defined in config.yaml.

    Returns:
        A validated pandas DataFrame of the raw sessions data.

    Raises:
        FileNotFoundError: if the CSV is missing.
        ValueError: if expected columns are missing or the target is absent.
    """
    csv_path = Path(path or CONFIG["paths"]["dataset"])

    if not csv_path.exists():
        logger.error("Dataset not found at %s", csv_path)
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows, %d columns", df.shape[0], df.shape[1])

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        logger.error("Dataset is missing expected columns: %s", missing_columns)
        raise ValueError(
            f"Dataset schema mismatch — missing columns: {sorted(missing_columns)}"
        )

    target_column = CONFIG["target_column"]
    if target_column not in df.columns:
        logger.error("Target column '%s' not found in dataset", target_column)
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    logger.info("Schema validation passed — all expected columns present")
    return df