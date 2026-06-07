# utils/utils.py
from pathlib import Path
import json
import numpy as np


def classify_risk(confidence: float, thresholds: dict) -> str:
    """
    Classify risk level according to configured thresholds.

    thresholds example:
    {
        "low_threshold": 0.3,
        "medium_threshold": 0.6,
        "high_threshold": 0.8,
        "critical_threshold": 0.95
    }
    """

    if confidence >= thresholds.get("critical_threshold", 0.95):
        return "CRITICAL"

    if confidence >= thresholds.get("high_threshold", 0.8):
        return "HIGH"

    if confidence >= thresholds.get("medium_threshold", 0.6):
        return "MEDIUM"

    if confidence >= thresholds.get("low_threshold", 0.3):
        return "LOW"

    return "INFO"


def generate_json_log(payload, path):

    # path_cfg = path  # self.config.get("output_paths", {})
    # out_path = Path(path)
    # "explainability_log", "explainability.json"))

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sanitize_for_json(obj):
    """Recursively convert NumPy types to Python primitives."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # handles np.nan, np.inf as well
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    else:
        return obj
