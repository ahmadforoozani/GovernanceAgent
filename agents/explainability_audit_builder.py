import pandas as pd
from datetime import datetime, timezone
import uuid
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ExplainabilityAuditBuilder:
    """
    Purpose:
        Produces fully traceable, human-auditable explanations for the
        multi‑agent governance pipeline by reconstructing the reasoning
        path across Validation, Statistical, Causal, Risk Fusion, and
        Audit stages. This agent provides a structured and deterministic
        “audit trail of decisions” for each location.

    Core Responsibilities:
        - Aggregate raw signals from data-quality checks, statistical
          inferences, causal fragility estimates, and fused risk scores.
        - Compute weighted contributions to the final risk score using
          the same fusion weights defined in the RiskEscalationAgent.
        - Generate a step-by-step reasoning chain describing how the
          pipeline processed inputs and escalated risk.
        - Provide per‑location explainability records and a global
          pipeline graph for documentation and regulatory export.

    Inputs (via context):
        validation:
            - issues (raw anomalies per location)
            - engineered DataQualityRisk scores
        statistical:
            - statistical summary scores per location
        causal:
            - causal_trace with fragility score metadata
        risk:
            - final risk score, level, and driver decomposition

    Output:
        JSON explainability artifact structured as:
            {
                "explainability": {
                    "regions": { ... },
                    "global_graph": {
                        "pipeline": [...],
                        "summary": "..."
                    },
                    "generated_at": timestamp
                }
            }

    """

    AGENT_VERSION = "1.0"

    def __init__(self, config: dict):
        # ---------------------------
        # Risk weights (shared with escalation)
        # ---------------------------
        risk_cfg = config.get("risk_agent", {})
        weights = risk_cfg.get("weights", {})

        self.w_dq = float(weights.get("data_quality", 0.2))
        self.w_st = float(weights.get("statistical", 0.5))
        self.w_cf = float(weights.get("causal_fragility", 0.3))

        # ---------------------------
        # Explainability config
        # ---------------------------
        exp_cfg = config.get("explainability", {})

        self.pipeline = exp_cfg.get(
            "pipeline_stages",
            [
                "Input Data",
                "ValidationAgent",
                "StatisticalReasoningAgent",
                "CausalSimulationAgent",
                "RiskEscalationAgent",
                "AuditAgent"
            ]
        )

    # -------------------------------------------------------
    # Utility
    # -------------------------------------------------------
    def _record(self, **kwargs):

        base = {
            "explain_id": str(uuid.uuid4()),
            "model_trigger": "ExplainabilityAuditBuilder",
            "agent_version": self.AGENT_VERSION
        }

        base.update(kwargs)
        return base

    # -------------------------------------------------------
    # Main
    # -------------------------------------------------------
    def run(self, data: pd.DataFrame, context: dict) -> dict:

        dq_risk = context.get("engineered", {}).get("DataQualityRisk", {})
        dq_raw = context.get("validation", {}).get("issues", {})

        stat = context.get("statistical", {})
        stat_scores = stat.get("summary_scores", {})

        causal = context.get("causal", {})
        risk = context.get("risk", {}).get("locations", {})

        locations = sorted(risk.keys())
        explain_regions = {}

        # Normalize validation issues
        if isinstance(dq_raw, list):
            dq_issues = {loc: dq_raw for loc in locations}
        elif isinstance(dq_raw, dict):
            dq_issues = dq_raw
        else:
            dq_issues = {loc: [] for loc in locations}

        for loc in locations:

            dq_score = float(dq_risk.get(loc, 0))
            st_score = float(stat_scores.get(loc, 0))

            causal_trace = {}
            causal_frag = 0.0

            if loc in causal:
                causal_trace = causal[loc].get("causal_trace", {}) or {}
                metrics = causal_trace.get("metrics", {}) or {}
                causal_frag = float(metrics.get("fragility_score", 0))

            r = risk.get(loc, {})

            final_score = float(r.get("risk_score", 0))
            lvl = r.get("risk_level")
            drivers = r.get("drivers", {})

            # contributions
            contrib_dq = self.w_dq * dq_score
            contrib_st = self.w_st * st_score
            contrib_cf = self.w_cf * causal_frag

            reasoning_chain = [

                "Layer 1: ValidationAgent identified schema and consistency issues.",
                f"-> Data quality score = {dq_score:.3f}, issues = {len(dq_issues.get(loc, []))}.",

                "Layer 2: StatisticalReasoningAgent analyzed trend deviations.",
                f"-> Statistical risk = {st_score:.3f}.",

                "Layer 3: CausalSimulationAgent estimated intervention fragility.",
                f"-> Causal fragility = {causal_frag:.3f}.",

                "Layer 4: RiskEscalationAgent fused all signals.",

                f"-> Risk fusion = "
                f"{self.w_dq}*dq + {self.w_st}*st + {self.w_cf}*cf "
                f"= {final_score:.3f}.",

                f"-> Final risk level determined as {lvl.upper()}."
            ]

            explain_regions[loc] = self._record(

                location_key=loc,

                fusion_inputs={
                    "data_quality_score": dq_score,
                    "statistical_score": st_score,
                    "causal_fragility": causal_frag
                },

                fusion_weights={
                    "data_quality_weight": self.w_dq,
                    "statistical_weight": self.w_st,
                    "causal_weight": self.w_cf
                },

                fusion_contribution={
                    "contribution_data_quality": contrib_dq,
                    "contribution_statistical": contrib_st,
                    "contribution_causal": contrib_cf
                },

                final_fusion={
                    "final_score": final_score,
                    "risk_level": lvl
                },

                causal_trace=causal_trace,
                data_quality_issues=dq_issues.get(loc, []),

                reasoning_chain=reasoning_chain
            )

        global_graph = {
            "pipeline": self.pipeline,
            "summary":
            "Data flows through validation -> statistical inference -> causal analysis -> risk fusion -> audit layer."
        }

        return {
            "status": "ok",
            "agent": "ExplainabilityAuditBuilder",
            "agent_version": self.AGENT_VERSION,
            "explainability": {
                "regions": explain_regions,
                "global_graph": global_graph,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
        }
