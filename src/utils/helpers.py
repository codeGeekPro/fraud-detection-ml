"""Helper functions for fraud detection."""

from typing import Optional, Any, Dict
import logging
import yaml


def setup_logging(config_path: str) -> None:
    """
    Configure le système de logging selon le fichier YAML.

    Args:
        config_path (str): Chemin du fichier de config
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Logging configuré au niveau {level}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.

    Args:
        config_path (str): Chemin du fichier YAML

    Returns:
        Dict[str, Any]: Dictionnaire de configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
