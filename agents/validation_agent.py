import pandas as pd
import numpy as np
import uuid
import re
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ValidationAgent:
    """
    Data Governance Validation Agent

    Features
    --------
    - Time continuity validation
    - Null ratio validation
    - Negative value detection
    - Cross-variable logical rules
    - Cumulative monotonicity validation
    - Governance risk scoring

    Designed for panel datasets (multi-location).
    Inputs:
        - Data as DataFrame
        - Context as Dictionary

    Output: Dataset issues in Json format:
    {
            "status": "...",
            "agent": "ValidationAgent",
            "agent_version": ...,
            "validation": {
                "issues": ...,
                "quality_scores": ...
            },
            "engineered": ...,
            "logs": ...
    }
    """

    AGENT_VERSION = "1"

    def __init__(self, config, data_config):
        logging.info("Initializing Validation Agent.")
        self.config = config
        self.dataset_cfg = data_config

        self.allowed_null_ratio = config.get("allowed_null_ratio", 0.05)
        self.expected_freq = config.get("expected_freq", None)
        self.sort_columns = data_config.get(
            "sort_columns", ["location_key", "date"])

        self.location_column = data_config.get(
            "location_column", "location_key")
        self.date_column = data_config.get("date_column", "date")
        self.required_cols = data_config.get("required_columns", [])
        self.non_neg_columns = data_config.get("non_negative_columns", [])
        self.cumulative_columns = data_config.get("cumulative_columns", [])
        self.rules = config.get("cross_variable_rules", [])

        self.severity_weights = config.get("severity_weights", {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        })

        self.tolerance = self.config.get("risk_tolerance", 0.05)

        self.location_format = config.get(
            "location_format", r"^[A-Z]{2}(?:_[A-Z0-9]{1,3}_\d+)?$")

    def _issue(self, **kwargs):
        base = {
            "anomaly_id": str(uuid.uuid4()),
            "model_trigger": "ValidationAgent",
            "agent_version": self.AGENT_VERSION,
            "confidence": 0.95
        }
        base.update(kwargs)
        return base

    def _freq_to_days(self):
        if self.expected_freq == "D":
            return 1
        if self.expected_freq == "W":
            return 7
        if self.expected_freq == "M":
            return 30
        return None

    def _severity_from_risk(self, risk_score):
        if risk_score >= 0.85:
            return "critical"
        elif risk_score >= 0.68:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    def run(self, data: pd.DataFrame, context: dict) -> dict:
        logging.info("Running Validation Agent")
        issues = {}
        logs = []

        def add_issue(issue):
            loc = issue.get(self.location_column)
            key = loc if loc is not None else "_global"
            issues.setdefault(key, []).append(issue)

        # ------------------------------------------------------
        # A) Structural Validation
        # ------------------------------------------------------

        # Finding missing columns
        missing_cols = [c for c in self.required_cols if c not in data.columns]

        if missing_cols:
            add_issue(self._issue(
                type="missing_columns",
                risk_level="critical",
                recommended_action="reject_dataset",
                validation_rule="required_columns_check",
                why_flagged="Dataset missing mandatory schema fields",
                observed_value=missing_cols,
                expected_condition="All required columns must exist"
            ))

        # checking location_key pattern, 2-letter country code (CC) or full CC_XXX_##### style key
        location_pattern = re.compile(self.location_format)

        bad_format_rows = data[
            data[self.location_column].astype(str).apply(
                lambda x: not bool(location_pattern.match(str(x).strip()))
            )
        ]

        for _, r in bad_format_rows.iterrows():
            add_issue(self._issue(
                location_key=r[self.location_column],
                date=str(pd.to_datetime(r.get(self.date_column, "")).date()),
                type="invalid_location_format",
                validation_rule="location_pattern_check",
                risk_level="medium",
                recommended_action="verify_location_identifier",
                why_flagged="Location code does not match the allowed format",
                observed_value=r[self.location_column],
                expected_condition="Pattern: 2-letter country code (CC) or full CC_XXX_##### style key"
            ))

        # Finding duplicated rows
        dups = data.duplicated(self.sort_columns).sum()

        if dups > 0:
            add_issue(self._issue(
                type="duplicate_rows",
                risk_level="high",
                recommended_action="deduplicate_records",
                validation_rule="unique_location_date",
                why_flagged="Duplicate records for same location/date",
                observed_value=int(dups),
                expected_condition="Unique (location,date)"
            ))

        try:
            data[self.date_column] = pd.to_datetime(data[self.date_column])
        except Exception:
            add_issue(self._issue(
                type="invalid_date_format",
                risk_level="critical",
                recommended_action="fix_date_format",
                validation_rule="date_parse_rule",
                why_flagged="Date column could not be parsed",
                expected_condition="ISO date format"
            ))

        expected_days = self._freq_to_days()

        if expected_days:

            for loc, df_loc in data.groupby(self.location_column):

                df_loc = df_loc.sort_values(self.date_column)

                diffs = df_loc[self.date_column].diff().dt.days

                for idx, gap in diffs.items():

                    if pd.isna(gap):
                        continue

                    if gap > expected_days:

                        gap_ratio = gap / expected_days

                        if gap_ratio >= 5:
                            sev = "critical"
                        elif gap_ratio >= 3:
                            sev = "high"
                        elif gap_ratio >= 1.5:
                            sev = "medium"
                        else:
                            sev = "low"

                        add_issue(self._issue(
                            location_key=loc,
                            date=str(pd.to_datetime(
                                df_loc.loc[idx, self.date_column]).date()),
                            metric="date",
                            type="time_continuity_break",
                            recommended_action="fix_time_gap",
                            why_flagged="finding time gap between records",
                            observed_value=int(gap),
                            expected_condition=f"{expected_days} days",
                            risk_level=sev
                        ))

        # ------------------------------------------------------
        # B) Logical Validation
        # ------------------------------------------------------

        # Checking for negative values

        for col in self.non_neg_columns:

            neg_rows = data[data[col] < 0]

            for _, row in neg_rows.iterrows():

                add_issue(self._issue(
                    location_key=row[self.location_column],
                    date=str(pd.to_datetime(row[self.date_column]).date()),
                    type="negative_value",
                    metric=col,
                    observed_value=row[col],
                    expected_condition=">= 0",
                    why_flagged="Daily metrics cannot be negative",
                    validation_rule="non_negative_daily_values",
                    risk_level="high",
                    recommended_action="review_data_entry"
                ))

        # Checking Cumulative values must be monotonic
        for col in self.cumulative_columns:

            for loc, df_loc in data.groupby(self.location_column):

                df_loc = df_loc.sort_values(self.date_column)

                prev = None

                for _, r in df_loc.iterrows():

                    if prev is not None and r[col] < prev:

                        add_issue(self._issue(
                            location_key=loc,
                            date=str(pd.to_datetime(
                                r[self.date_column]).date()),
                            type="cumulative_backwards_drop",
                            metric=col,
                            observed_value=r[col],
                            expected_condition=f">= {prev}",
                            why_flagged="Cumulative values must be monotonic",
                            validation_rule="cumulative_monotonicity",
                            risk_level="high",
                            recommended_action="investigate_data_revision"
                        ))

                    prev = r[col]

        # Finding rows violating cross rules
        for rule in self.rules:

            col_a = rule.get("left")
            col_b = rule.get("right")
            relation = rule.get("relation")

            if col_a not in data.columns or col_b not in data.columns:
                continue

            mask = data[col_a].notna() & data[col_b].notna()

            subset = data[mask]

            if relation == "<=":
                violations = subset[subset[col_a] > subset[col_b]]

            elif relation == ">=":
                violations = subset[subset[col_a] < subset[col_b]]

            else:
                continue

            for _, r in violations.iterrows():

                add_issue(self._issue(
                    location_key=r[self.location_column],
                    date=str(pd.to_datetime(r[self.date_column]).date()),
                    type="cross_variable_rules",
                    metric=col_a,
                    observed_value=int(r[col_a]),
                    expected_condition=f"{relation} {col_b} (value= {r[col_b]})",
                    why_flagged=rule.get("why"),
                    validation_rule=rule.get("rule"),
                    risk_level="high",
                    recommended_action=rule.get("action")
                ))

        # ------------------------------------------------------
        # C) Completeness Checks
        # ------------------------------------------------------

        # Thresholds (configurable)
        loc_thresh = self.config.get("null_threshold_location", 0.0)
        date_thresh = self.config.get("null_threshold_date", 0.0)
        cumulative_thresh = self.config.get("null_threshold_cumulative", 0.02)
        new_col_thresh = self.config.get("null_threshold_new", 0.10)
        default_thresh = self.allowed_null_ratio  # e.g. 0.05

        quality_scores = {}

        for loc, df_loc in data.groupby(self.location_column):

            # (1) Per-column null ratio
            col_null_ratios = df_loc.isnull().mean()

            for col, null_ratio in col_null_ratios.items():

                # ---- Determine threshold based on column type ----
                if col == self.location_column:
                    threshold = loc_thresh

                elif col == self.date_column:
                    threshold = date_thresh

                elif col in self.cumulative_columns:
                    threshold = cumulative_thresh

                elif col in self.non_neg_columns:
                    threshold = new_col_thresh

                else:
                    threshold = default_thresh

                # ---- Flag if higher than threshold ----
                if null_ratio > threshold:
                    add_issue(self._issue(
                        location_key=loc,
                        type="excessive_null_ratio",
                        metric=col,
                        observed_value=float(null_ratio),
                        expected_condition=f"<= {threshold}",
                        why_flagged=f"Column '{col}' exceeds allowed null ratio",
                        validation_rule="column_specific_null_threshold",
                        risk_level="medium" if null_ratio < threshold * 3 else "high",
                        recommended_action="review_data_completeness"
                    ))

            # (2) Total issues for this location
            loc_issues = issues.get(loc, [])

            # (3) Completeness score
            # completeness_score = max(0, 1 - (total_issues / (len(df_loc) + 1)))

            severity_counter = Counter(
                i.get("risk_level", "medium") for i in loc_issues
            )

            weighted = sum(
                self.severity_weights.get(i.get("risk_level", "medium"), 0.4)
                for i in loc_issues
            )
            # NORMALIZATION by data size
            normalized_weighted = weighted / (max(len(df_loc), 1))

            # risk relative to tolerance
            risk_score = min(normalized_weighted / self.tolerance, 1.0)

            severity = self._severity_from_risk(risk_score)

            quality_scores[loc] = {
                "issue_count": len(loc_issues),
                "critical_count": severity_counter["critical"],
                "severity_distribution": dict(severity_counter),
                "weighted_issue_score": float(normalized_weighted),
                "data_quality_risk": float(risk_score),
                "data_quality_severity": severity
            }

        engineered = {
            "DataQualityRisk": {},
            "DataQualitySeverity": {},
            "DataQualityIssueCount": {},
            "DataQualityCriticalCount": {}
        }

        for loc, q in quality_scores.items():

            engineered["DataQualityRisk"][loc] = q["data_quality_risk"]
            engineered["DataQualitySeverity"][loc] = q["data_quality_severity"]
            engineered["DataQualityIssueCount"][loc] = q["issue_count"]
            engineered["DataQualityCriticalCount"][loc] = q["critical_count"]

        # ------------------------------------------------------
        # Final Output
        # ------------------------------------------------------
        logging.info("Generating Validation Agent''s Json output.")
        return {
            "status": "ok",
            "agent": "ValidationAgent",
            "agent_version": self.AGENT_VERSION,
            "validation": {
                "issues": issues,
                "quality_scores": quality_scores
            },
            "engineered": engineered,
            "logs": logs
        }
