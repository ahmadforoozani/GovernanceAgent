import pandas as pd
import numpy as np
from datetime import datetime, timezone


class DataQualityReport:

    # -------------------------------------------------
    # Dataset summary
    # -------------------------------------------------

    def dataset_summary(self, df):

        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "locations": df["location_key"].nunique(),
            "date_start": str(df["date"].min()),
            "date_end": str(df["date"].max()),
            "duplicate_rows": int(
                df.duplicated(["location_key", "date"]).sum()
            ),
        }

        return summary

    # -------------------------------------------------
    # Column statistics
    # -------------------------------------------------

    def column_statistics(self, df):

        stats = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:

            series = df[col]

            stats[col] = {
                "null_ratio": float(series.isna().mean()),
                "min": float(series.min()) if not series.isna().all() else None,
                "max": float(series.max()) if not series.isna().all() else None,
                "mean": float(series.mean()) if not series.isna().all() else None,
                "negative_values": int((series < 0).sum())
            }

        return stats

    # -------------------------------------------------
    # Outlier detection
    # -------------------------------------------------

    def detect_outliers(self, df):

        anomalies = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:

            series = df[col].dropna()

            if len(series) < 10:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0 or np.isnan(std):
                continue

            z_scores = (series - mean) / std

            outliers = series[np.abs(z_scores) > 5]

            if len(outliers) > 0:
                anomalies[col] = int(len(outliers))

        return anomalies

    # -------------------------------------------------
    # Cumulative decrease check
    # -------------------------------------------------

    def cumulative_decrease(self, df):

        issues = {}

        cumulative_cols = [
            c for c in df.columns if c.startswith("cumulative_")
        ]

        for col in cumulative_cols:

            decreases = 0

            for loc, group in df.groupby("location_key"):

                group = group.sort_values("date")

                diff = group[col].diff()

                decreases += int((diff < 0).sum())

            if decreases > 0:
                issues[col] = decreases

        return issues

    # -------------------------------------------------
    # Rule checks
    # -------------------------------------------------

    def rule_checks(self, df):

        violations = {}

        if "new_deceased" in df.columns and "new_confirmed" in df.columns:

            v = (df["new_deceased"] > df["new_confirmed"]).sum()

            if v > 0:
                violations["new_deceased_gt_new_confirmed"] = int(v)

        return violations

    # -------------------------------------------------
    # Full report
    # -------------------------------------------------

    def generate_report(self, df):

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_summary": self.dataset_summary(df),
            "column_statistics": self.column_statistics(df),
            "outliers": self.detect_outliers(df),
            "cumulative_decrease": self.cumulative_decrease(df),
            "rule_violations": self.rule_checks(df),
        }

        return report


def main():
    df = pd.read_csv("dataset/weekly_merged_dataset.csv")
    dq = DataQualityReport()
    report = dq.generate_report(df)
    print(report)


if __name__ == "__main__":
    main()
