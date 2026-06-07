import os
import json
import logging
import traceback
import pandas as pd
from core.config_loader import load_config
from core.governance_orchestrator import GovernanceOrchestrator
from utils.utils import sanitize_for_json


# -----------------------------
#  Logging Configuration
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def save_output(obj, filename):
    """
    Save pipeline output into a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, indent=2, ensure_ascii=False)

    logging.info(f"Pipeline output saved to {filename}")


def main():

    logging.info("Pipeline started.")

    config = load_config("app\config.yaml")
    orchestrator = GovernanceOrchestrator(config)
    dataset = _load_dataset(config)
    logging.info(f"Dataset loaded. Shape={dataset.shape}")

    result = orchestrator.run(dataset)
    # -------------------------
    # Store result
    # -------------------------
    OUTPUT_PATH = "outputs/audit_log.json"
    save_output(result, OUTPUT_PATH)

    if result.get("pipeline_status", False):
        logging.info("Pipeline finished successfully.")
    else:
        logging.info("Pipeline failed.")


def _load_dataset(config: dict):

    data_cfg = config.get("data", {})
    path = data_cfg.get("dataset_path")
    if not path:
        raise ValueError("Dataset path is missing in configuration.")

    df = pd.read_csv(path)

    date_col = data_cfg.get("date_column")
    loc_col = data_cfg.get("location_column", "location_key")

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    df = df[~df[loc_col].str.contains("_")]
    sort_cols = data_cfg.get("sort_columns", [loc_col, date_col])
    available_sort_cols = [c for c in sort_cols if c in df.columns]

    if available_sort_cols:
        df = df.sort_values(available_sort_cols).reset_index(drop=True)

    return df


if __name__ == "__main__":
    main()
