import numpy as np
import pandas as pd
from datetime import datetime, timezone
import uuid
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AuditAgent:
    """
    Purpose:
        Consolidates outputs from Validation, Statistical, Causal,
        and Risk Escalation agents into a structured, explainable,
        and governance-ready audit report per location.

    Core Responsibilities:
        - Aggregate multi-agent risk signals
        - Classify approval status (auto-approved vs human review)
        - Generate structured explainability trace
        - Produce region-level and global risk summaries
        - Compute confidence score based on signal consistency

    Inputs (via context):
        - validation:
            * issues (raw data-quality anomalies)
            * engineered DataQualityRisk scores
        - statistical:
            * formatted anomaly messages
        - causal:
            * explanation and causal trace metadata
        - risk:
            * final risk score, level, drivers, escalation explanation

    Output:
        Structured JSON:
            {
                audit: {
                    report: {
                        regions: { ... },
                        global_summary: { ... },
                        generated_at: timestamp
                    }
                }
            }
    """

    AGENT_VERSION = "1.0"

    def __init__(self, config: dict):

        logging.info("Initializing Audit Agent.")
        audit_cfg = config.get("audit", {})

        # policy: which risk levels require human review
        self.review_levels = set(
            audit_cfg.get(
                "human_review_levels",
                ["high", "critical"]
            )
        )

        # statuses
        statuses = audit_cfg.get("statuses", {})

        self.status_review_required = statuses.get(
            "review_required",
            "pending_human_review"
        )

        self.status_auto = statuses.get(
            "auto_approved",
            "auto_approved"
        )

        # issue truncation
        self.max_issues = audit_cfg.get(
            "max_issues_per_region",
            5
        )

    # ---------------------------------------------------------
    # utility: create audit record
    # ---------------------------------------------------------
    def _record(self, **kwargs):

        base = {
            "audit_id": str(uuid.uuid4()),
            "model_trigger": "AuditAgent",
            "agent_version": self.AGENT_VERSION
        }

        base.update(kwargs)
        return base

    # ---------------------------------------------------------
    # run
    # ---------------------------------------------------------
    def run(self, data: pd.DataFrame, context: dict) -> dict:

        logging.info("Running Audit Agent.")
        dq_risk = context.get("engineered", {}).get("DataQualityRisk", {})
        dq_raw = context["validation"]["issues"]

        statistical = context["statistical"]
        stat_msgs = statistical["formatted_messages"]

        causal = context.get("causal", {})
        risk = context.get("risk", {}).get("locations", {})

        locations = sorted(risk.keys())

        report = {}

        global_low = 0
        global_med = 0
        global_high = 0
        global_critical = 0

        for loc in locations:

            r = risk.get(loc, {})

            level = r.get("risk_level")
            score = r.get("risk_score")

            drivers = r.get("drivers", {})
            rec_action = r.get("recommended_action")
            explanation = r.get("explanation")

            # ----------------------------------
            # Global counters
            # ----------------------------------
            if level == "critical":
                global_critical += 1
            elif level == "high":
                global_high += 1
            elif level == "medium":
                global_med += 1
            else:
                global_low += 1

            # ----------------------------------
            # Data quality
            # ----------------------------------
            issues = []

            if isinstance(dq_raw, dict):
                issues = dq_raw.get(loc, [])

            if self.max_issues and len(issues) > self.max_issues:
                issues = issues[:self.max_issues] + [
                    f"...(+{len(issues)-self.max_issues} more)"
                ]

            dq_score = dq_risk.get(loc, 0)

            # ----------------------------------
            # Statistical
            # ----------------------------------
            stat_list = []

            if isinstance(stat_msgs, dict):

                stat_list = stat_msgs.get(loc, [])

            elif isinstance(stat_msgs, list):

                prefix = f"{loc},"

                stat_list = [
                    msg for msg in stat_msgs
                    if isinstance(msg, str) and msg.startswith(prefix)
                ]

            if self.max_issues and len(stat_list) > 2*self.max_issues:
                stat_list = stat_list[:2*self.max_issues] + [
                    f"...(+{len(stat_list)-2*self.max_issues} more)"
                ]

            st_score = drivers.get('statistical', 0)
            # ----------------------------------
            # Causal
            # ----------------------------------
            causal_exp = None
            causal_frag = None

            if loc in causal:

                causal_exp = causal[loc].get("explanation")

                ct = causal.get(loc, {}).get("causal_trace") or {}

                causal_frag = ct.get("metrics", {}).get("fragility_score", 0)

            # Confidence Score
            signals = [
                float(dq_score),
                float(st_score),
                float(causal_frag)
            ]

            variance = np.var(signals)
            confidence = round(max(0.0, 1.0 - (variance * 2)), 3)

            # ----------------------------------
            # Governance reasoning
            # ----------------------------------
            governance_reasoning = (
                f"Risk classified as {level.upper()} with score={score:.3f}. "
                f"Drivers -> statistical={drivers.get('statistical',0):.3f}, "
                f"causal_fragility={drivers.get('causal_fragility',0):.3f}, "
                f"data_quality_Risk={drivers.get('data_quality',0):.3f}."
            )

            # ----------------------------------
            # Approval logic (config-driven)
            # ----------------------------------
            requires_review = level in self.review_levels

            approval_status = (
                self.status_review_required
                if requires_review
                else self.status_auto
            )

            # ----------------------------------
            # Build audit record
            # ----------------------------------
            report[loc] = self._record(

                location_key=loc,

                risk_level=level,
                risk_score=score,
                confidence_score=confidence,

                recommended_action=rec_action,

                approval_status=approval_status,
                requires_human_review=requires_review,

                governance_reasoning=governance_reasoning,

                data_quality={
                    "risk": dq_score,
                    "issues": issues
                },

                statistical_summary=stat_list,

                causal_summary={
                    "explanation": causal_exp,
                    "fragility_score": causal_frag
                },

                explainability_trace={
                    "risk_drivers": drivers,
                    "escalation_explanation": explanation
                }
            )

        # ----------------------------------
        # Global summary
        # ----------------------------------
        global_summary = {
            "critical_risk_regions": global_critical,
            "high_risk_regions": global_high,
            "medium_risk_regions": global_med,
            "low_risk_regions": global_low
        }
        logging.info("Generating Audit Agent's Json output.")
        return {
            "status": "ok",
            "agent": "AuditAgent",
            "agent_version": self.AGENT_VERSION,
            "audit": {
                "report": {
                    "regions": report,
                    "global_summary": global_summary,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        }
