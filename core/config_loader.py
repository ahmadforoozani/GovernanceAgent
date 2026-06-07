import yaml
from pathlib import Path

REQUIRED_TOP_LEVEL_KEYS = [
    "system",
    "data",
    "validation_agent",
    "statistical_agent",
    "causal_agent",
    "risk_agent",
    "audit_agent",
    "explainability"
]


def load_config(config_path):
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Config file is empty.")

    validate_config_structure(config)

    return config


def validate_config_structure(config: dict):

    missing_keys = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in config]

    if missing_keys:
        raise ValueError(f"Missing required config sections: {missing_keys}")

    validate_data_config(config["data"])


def validate_data_config(data_cfg: dict):

    required_fields = [
        "dataset_path",
        "date_column",
        "location_column"
    ]

    missing = [f for f in required_fields if f not in data_cfg]

    if missing:
        raise ValueError(
            f"Missing required fields in data config: {missing}"
        )

    dataset_path = Path(data_cfg["dataset_path"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )
