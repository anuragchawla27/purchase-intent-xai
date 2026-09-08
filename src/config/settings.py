"""
Loads project configuration from config.yaml into a single object
that every module (loader, preprocessing, API, dashboard) imports from,
instead of each one reading/parsing YAML itself.
"""

import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

# config.yaml lives in the same folder as this file
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """
    Load and return the project config as a dictionary.

    Raises:
        FileNotFoundError: if config.yaml is missing.
        yaml.YAMLError: if config.yaml is malformed.
    """
    if not path.exists():
        logger.error("Config file not found at %s", path)
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Config loaded successfully from %s", path)
    return config


# Load once at import time so every module that does
# `from src.config.settings import CONFIG` gets the same object.
CONFIG = load_config()