import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path("dataset")

PATH_EPI = BASE_DIR / "epidemiology.csv"
PATH_HOSP = BASE_DIR / "hospitalizations.csv"
PATH_VACC = BASE_DIR / "vaccinations.csv"
OUTPUT_PATH = BASE_DIR / "weekly_merged_dataset.csv"

# -------------------------------------------------
# Unified schema definition
# -------------------------------------------------

ESSENTIAL_COLUMNS = {
    "epidemiology": [
        "date",
        "location_key",
        "new_confirmed",
        "new_deceased",
        "cumulative_confirmed",
        "cumulative_deceased",
    ],
    "hospitalizations": [
        "date",
        "location_key",
        "new_hospitalized_patients",
        "current_hospitalized_patients",
        "new_intensive_care_patients",
        "current_intensive_care_patients",
    ],
    "vaccinations": [
        "date",
        "location_key",
        "new_persons_vaccinated",
        "cumulative_persons_vaccinated",
        "new_persons_fully_vaccinated",
        "cumulative_persons_fully_vaccinated",
        "new_vaccine_doses_administered",
        "cumulative_vaccine_doses_administered",
    ],
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------


def select_columns(df, dataset_name):
    required = ESSENTIAL_COLUMNS[dataset_name]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name}: missing required columns: {missing}")
    return df[required].copy()


def optimize_memory(df):
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def validate_duplicates(df, date_col="date"):
    dup = df.duplicated(subset=["location_key", date_col])
    if dup.any():
        print(f"⚠️ {dup.sum()} duplicates removed for {df.name}")
        df = df.drop_duplicates(subset=["location_key", date_col])
    return df


def fix_negative_new_values(df):
    for col in df.columns:
        if col.startswith("new_"):
            df[col] = df[col].clip(lower=0)
    return df

# -------------------------------------------------
# Weekly Aggregation
# -------------------------------------------------


def aggregate_weekly(df, date_col="date"):
    df = df.sort_values(["location_key", date_col])
    df["week_ending_date"] = df[date_col] + pd.to_timedelta(
        6 - df[date_col].dt.dayofweek, unit="D"
    )

    agg_dict = {}
    for col in df.columns:
        if col in ["location_key", date_col, "week_ending_date"]:
            continue
        if col.startswith("new_"):
            agg_dict[col] = "sum"
        elif col.startswith(("cumulative_", "current_", "total_")):
            agg_dict[col] = "last"

    df_weekly = (
        df.groupby(["location_key", "week_ending_date"], observed=True)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"week_ending_date": "date"})
    )
    return df_weekly

# -------------------------------------------------
# Load datasets
# -------------------------------------------------


def load_dataset(path, dataset_name):
    df = pd.read_csv(path)
    df.name = dataset_name
    df = select_columns(df, dataset_name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = validate_duplicates(df)
    df = fix_negative_new_values(df)
    df = optimize_memory(df)
    return df

# -------------------------------------------------
# Merge datasets
# -------------------------------------------------


def merge_datasets(df_epi, df_hosp, df_vacc):
    print("Merging epidemiology + hospitalizations ...")
    df = pd.merge(
        df_epi, df_hosp, on=["location_key", "date"], how="left"
    )
    print("Merging vaccinations ...")
    df = pd.merge(
        df, df_vacc, on=["location_key", "date"], how="left"
    )
    return df

# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------


def main():
    print("🔹 Loading datasets ...")
    df_epi = load_dataset(PATH_EPI, "epidemiology")
    df_hosp = load_dataset(PATH_HOSP, "hospitalizations")
    df_vacc = load_dataset(PATH_VACC, "vaccinations")

    print("🔹 Aggregating weekly ...")
    df_weekly_epi = aggregate_weekly(df_epi)
    df_weekly_hosp = aggregate_weekly(df_hosp)
    df_weekly_vacc = aggregate_weekly(df_vacc)

    print("🔹 Merging all datasets ...")
    df_final = merge_datasets(df_weekly_epi, df_weekly_hosp, df_weekly_vacc)
    df_final = optimize_memory(df_final)

    print("🔹 Saving final dataset ...")
    df_final.to_csv(OUTPUT_PATH, index=False)
    print("✅ Done.")
    print("Final columns:", list(df_final.columns))
    print("Final shape:", df_final.shape)

# -------------------------------------------------


if __name__ == "__main__":
    main()
