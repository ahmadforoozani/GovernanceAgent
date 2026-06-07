import uuid
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class RiskEscalationAgent:
    """
    Fuses upstream governance signals into a policy escalation score.

    Inputs:

        - Data quality risk
        - Statistical anomaly scores
        - Causal fragility signals

    Risk score:

        risk =
            w_dq * data_quality +
            w_stat * statistical +
            w_causal * causal_fragility

    Risk levels are determined using fixed thresholds defined
    in config.yaml.

    Escalation actions are also fully configurable.
    """

    AGENT_VERSION = "1.0"

    def __init__(self, config: dict):
        logging.info("Initializing Risk Escalation Agent.")
        # weights
        weights = config.get("weights", {})

        self.w_dq = float(weights.get("data_quality", 0.2))
        self.w_stat = float(weights.get("statistical", 0.5))
        self.w_causal = float(weights.get("causal_fragility", 0.3))

        # thresholds
        thr = config.get("thresholds", {})

        self.t_low = float(thr.get("low", 0.2))
        self.t_medium = float(thr.get("medium", 0.4))
        self.t_high = float(thr.get("high", 0.7))
        self.t_critical = float(thr.get("critical", 0.9))

        # actions
        self.actions = config.get(
            "actions",
            {
                "low": "routine_monitoring",
                "medium": "monitor_closely",
                "high": "open_regional_review",
                "critical": "immediate_policy_review"
            }
        )

    def _record(self, **kwargs):

        base = {
            "escalation_id": str(uuid.uuid4()),
            "model_trigger": "RiskEscalationAgent",
            "agent_version": self.AGENT_VERSION
        }

        base.update(kwargs)
        return base

    def _risk_level(self, score):
        EPS = 0.001
        if score >= self.t_critical - EPS:
            return "critical"

        if score >= self.t_high - EPS:
            return "high"

        if score >= self.t_medium - EPS:
            return "medium"

        return "low"

    def _confidence(self, level, score):

        if level == "critical":
            return 1.0

        if level == "high":
            return min(1.0, score * 1.2)

        if level == "medium":
            return min(1.0, score)

        return min(1.0, score * 0.7)

    def run(self, data: pd.DataFrame, context: dict) -> dict:
        """
        Execute escalation analysis.

        Note:
            risk drivers come from upstream agents via context.
        """
        logging.info("Running Risk Escalation Agent.")
        dq_risk = context.get(
            "engineered", {}
        ).get(
            "DataQualityRisk", {}
        )

        statistical = context.get("statistical", {})
        causal = context.get("causal", {})

        st_risk = statistical.get("summary_scores", {})

        results = {}

        for loc in st_risk.keys():

            dq = float(dq_risk.get(loc, 0.0))
            st = float(st_risk.get(loc, 0.0))

            cfrag = 0.0

            if loc in causal:

                trace = causal.get(loc, {}).get(
                    "causal_trace",
                    {}
                )

                if isinstance(trace, dict):
                    cfrag = float(trace.get("metrics", {}).get(
                        "fragility_score", 0.0))

            risk_score = (
                self.w_dq * dq +
                self.w_stat * st +
                self.w_causal * cfrag
            )

            level = self._risk_level(risk_score)

            action = self.actions.get(level, "manual_review")

            confidence = self._confidence(level, risk_score)

            explanation = (
                f"Risk score={risk_score:.3f}, classified as {level.upper()}. "
                f"Drivers: statistical={st:.3f}, "
                f"causal_fragility={cfrag:.3f}, "
                f"data_quality={dq:.3f}. "
                f"Thresholds: "
                f"medium={self.t_medium}, "
                f"high={self.t_high}, "
                f"critical={self.t_critical}."
            )

            if cfrag > 0.3:

                policy_implication = (
                    "Elevated causal fragility indicates structural drivers "
                    "behind the risk signal."
                )

            else:

                policy_implication = (
                    "Risk appears primarily driven by statistical anomalies."
                )

            results[loc] = self._record(

                status="ok",

                location_key=loc,

                risk_score=float(risk_score),
                risk_level=level,

                drivers={
                    "data_quality": dq,
                    "statistical": st,
                    "causal_fragility": cfrag
                },

                recommended_action=action,

                confidence=float(confidence),

                explanation=explanation,

                policy_risk_implication=policy_implication,

                escalation_trace={

                    "thresholds": {
                        "low": self.t_low,
                        "medium": self.t_medium,
                        "high": self.t_high,
                        "critical": self.t_critical
                    },

                    "driver_weights": {
                        "data_quality": self.w_dq,
                        "statistical": self.w_stat,
                        "causal_fragility": self.w_causal
                    }
                }
            )

        logging.info("Generating Risk Escalation Agent's Json output.")
        return {
            "status": "ok",
            "agent": "RiskEscalationAgent",
            "agent_version": self.AGENT_VERSION,
            "risk": {
                "locations": results
            }
        }
