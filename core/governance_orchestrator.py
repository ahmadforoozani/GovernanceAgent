import os
import json
import uuid
import traceback
import logging

import pandas as pd

from datetime import datetime, timezone

from utils.utils import sanitize_for_json
from utils.human_report_builder import build_executive_summary, build_human_readable_report
from utils.pdf_report_builder import build_pdf

from agents.validation_agent import ValidationAgent
from agents.statistical_reasoning_agent import StatisticalReasoningAgent
from agents.causal_simulation_agent import CausalSimulationAgent
from agents.risk_escalation_agent import RiskEscalationAgent
from agents.audit_agent import AuditAgent
from agents.explainability_audit_builder import ExplainabilityAuditBuilder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class GovernanceOrchestrator:
    """

    Purpose:
        Central execution engine of the GovernanceAgent system.
        Orchestrates the full multi‑agent pipeline, manages context
        propagation between stages, tracks execution metadata, and
        generates structured governance outputs and reports.

    Core Responsibilities:
        - Initialize and register all pipeline agents
        - Execute pipeline stages in configured order
        - Route intermediate outputs through shared context
        - Capture execution trace (timing, success/failure, errors)
        - Generate consolidated audit, explainability, and summary artifacts
        - Export machine-readable and human-readable reports (JSON, MD, PDF)

    Pipeline Architecture:
        Input Data
            → ValidationAgent
            → StatisticalReasoningAgent
            → CausalSimulationAgent
            → RiskEscalationAgent
            → AuditAgent
            → ExplainabilityAuditBuilder

    Execution Model:
        - Stage execution is config-driven via `pipeline.stages`
        - Each agent declares whether it requires raw data and/or context
        - Context is progressively enriched after each stage
        - Errors are captured with full traceback for observability
        - Optional stages can fail without stopping the pipeline

    Context Structure:
        {
            "validation": {...},
            "engineered": {...},
            "statistical": {...},
            "causal": {...},
            "risk": {...}
        }

    Output Artifacts:
        - pipeline_full_log.json      → complete execution state
        - pipeline_summary.json       → executive-level risk summary
        - human_audit_report.md/pdf   → detailed governance report
        - executive_summary.md/pdf    → executive snapshot

    """

    def __init__(self, config: dict):

        self.config = config or {}

        # ------------------------------------------------
        # System config
        # ------------------------------------------------

        system_cfg = self.config.get("system", {})

        self.system_version = system_cfg.get(
            "system_version",
            "1.0"
        )

        self.pipeline_version = system_cfg.get(
            "pipeline_version",
            "1.0"
        )

        self.output_dir = system_cfg.get(
            "output_dir",
            "outputs"
        )

        os.makedirs(self.output_dir, exist_ok=True)

        # ------------------------------------------------
        # Pipeline config
        # ------------------------------------------------

        pipe_cfg = self.config.get("pipeline", {})

        self.pipeline_stages = pipe_cfg.get(
            "stages",
            []
        )

        self.continue_on_failure = pipe_cfg.get(
            "continue_on_failure",
            False
        )

        # ------------------------------------------------
        # Agent config
        # ------------------------------------------------

        self.agent_cfg = self.config.get("agents", {})

        # ------------------------------------------------
        # Registry
        # ------------------------------------------------

        self.agent_registry = {

            "validation":
            ValidationAgent(
                self.config.get("validation_agent", {}),
                self.config.get("data", {})
            ),

            "statistical":
            StatisticalReasoningAgent(
                self.config.get("statistical_agent", {}),
                self.config.get("data", {})
            ),

            "causal":
            CausalSimulationAgent(
                self.config.get("causal_agent", {}),
                self.config.get("data", {})
            ),

            "risk":
            RiskEscalationAgent(
                self.config.get("risk_agent", {})
            ),

            "audit":
            AuditAgent(
                self.config.get("audit_agent", {})
            ),

            "explainability":
            ExplainabilityAuditBuilder(
                self.config
            )
        }

        # ------------------------------------------------
        # Output config
        # ------------------------------------------------

        outputs_cfg = self.config.get("outputs", {})

        self.full_log_name = outputs_cfg.get(
            "full_log",
            "pipeline_full_log.json"
        )

        self.summary_log_name = outputs_cfg.get(
            "summary_log",
            "pipeline_summary.json"
        )

        self.high_risk_name = outputs_cfg.get(
            "high_risk_log",
            "pipeline_high_risk.json"
        )

    # ------------------------------------------------
    # Run single agent
    # ------------------------------------------------

    def _run_agent(self, stage_name, data, context, trace):

        cfg = self.agent_cfg.get(stage_name, {})

        if not cfg.get("enabled", True):
            return None

        agent = self.agent_registry.get(stage_name)

        if agent is None:
            raise ValueError(
                f"Agent '{stage_name}' not registered in orchestrator")

        requires = cfg.get("requires", {})

        use_data = requires.get("data", True)
        use_context = requires.get("context", True)

        input_data = data if use_data else None
        input_context = context if use_context else {}

        start = datetime.now(timezone.utc)

        try:

            result = agent.run(input_data, input_context)

            trace.append({

                "stage": stage_name,

                "status": "success",

                "requires_data": use_data,
                "requires_context": use_context,

                "start_time": start.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:

            trace.append({

                "stage": stage_name,

                "status": "failure",

                "error": str(e),

                "traceback": traceback.format_exc(),

                "time": datetime.now(timezone.utc).isoformat()
            })

            optional = cfg.get("optional", False)

            if optional or self.continue_on_failure:
                return None

            raise

    # ------------------------------------------------
    # Write file
    # ------------------------------------------------
    def _write_file(self, filename, payload):

        path = os.path.join(
            self.output_dir,
            filename
        )

        with open(path, "w", encoding="utf-8") as f:

            json.dump(
                sanitize_for_json(payload),
                f,
                indent=2,
                ensure_ascii=False
            )

        return path

    # ----------------------------------------------------
    # merge region-centric
    # ----------------------------------------------------

    def _merge_regions(self, dst: dict, src: dict):

        if not src:
            return dst

        for region, new_data in src.items():

            if region not in dst:
                dst[region] = {}

            for k, v in new_data.items():

                if isinstance(v, dict):
                    dst[region].setdefault(k, {})
                    dst[region][k].update(v)
                else:
                    dst[region][k] = v

        return dst

    # ------------------------------------------------
    # Main
    # ------------------------------------------------

    def run(self, data: pd.DataFrame):

        pipeline_id = str(uuid.uuid4())
        logging.info(f"Generating Run ID: {pipeline_id} and Running Agents")

        execution_trace = []

        context = {
            "validation": {},
            "engineered": {},
            "statistical": {},
            "causal": {},
            "risk": {}
        }

        stage_outputs = {}

        try:

            for stage in self.pipeline_stages:

                result = self._run_agent(
                    stage,
                    data,
                    context,
                    execution_trace
                )

                stage_outputs[stage] = result

                # ------------------------------------
                # Context routing
                # ------------------------------------

                if not result:
                    continue

                if stage == "validation":

                    context["validation"] = result["validation"]

                    context["engineered"] = result["engineered"]

                elif stage == "statistical":

                    stat = result["statistical"]

                    context["statistical"].setdefault("anomalies", [])
                    context["statistical"]["anomalies"].extend(
                        stat.get("anomalies", [])
                    )

                    context["statistical"].setdefault("formatted_messages", [])
                    context["statistical"]["formatted_messages"].extend(
                        stat.get("formatted_messages", [])
                    )

                    context["statistical"].setdefault("summary_scores", {})
                    context["statistical"]["summary_scores"].update(
                        stat.get("summary_scores", {})
                    )

                elif stage == "causal":

                    self._merge_regions(
                        context["causal"],
                        result["causal"]
                    )

                elif stage == "risk":

                    locations = result["risk"].get("locations", {})

                    context["risk"].setdefault("locations", {})

                    for region, payload in locations.items():
                        context["risk"]["locations"][region] = payload

            # ------------------------------------
            # Export high risk
            # ------------------------------------

            export_levels = set(
                self.config.get("audit", {})
                .get("export_risk_levels", ["high"])
            )

            # ------------------------------------
            # Final output
            # ------------------------------------
            audit_stage = stage_outputs.get("audit")
            audit_out = audit_stage.get("audit") if audit_stage else None

            explain_stage = stage_outputs.get("explainability")
            explain_out = explain_stage.get(
                "explainability") if explain_stage else None
            final_output = {

                "pipeline_status": "success",

                "pipeline_metadata": {
                    "pipeline_id": pipeline_id,

                    "system_version":
                    self.system_version,

                    "pipeline_version":
                    self.pipeline_version,
                    "generated_at":
                    datetime.now(timezone.utc).isoformat()
                },

                "execution_trace": execution_trace,

                "validation": context["validation"],
                "engineered": context["engineered"],
                "statistical": context["statistical"],
                "causal": context["causal"],
                "risk": context["risk"],

                "audit": audit_out,
                "explainability": explain_out
            }

            # ------------------------------------
            # Save files
            # ------------------------------------
            logging.info("Generating and saving output files...")
            self._write_file(self.full_log_name, final_output)
            risk_counts = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            }
            risk_regions = {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            }

            highest = None
            regions = context.get("risk", {}).get("locations", {}) or {}
            for r, v in regions.items():

                level = v.get("risk_level")

                if level in risk_counts:
                    risk_counts[level] += 1
                    risk_regions[level].append(r)

                if highest is None or v.get("risk_score", 0) > highest.get("risk_score", 0):
                    highest = v

            summary = {
                "pipeline_id": pipeline_id,
                "generated_at": final_output["pipeline_metadata"]["generated_at"],
                "num_regions": len(regions),
                "critical_risk_regions": risk_regions["critical"],
                "high_risk_regions": risk_regions["high"],
                "medium_risk_regions": risk_regions["medium"],
                "low_risk_regions": risk_regions["low"],
                "highest_risk_location": highest.get("location_key") if highest else None,
                "highest_risk_level": highest.get("risk_level") if highest else None,
                "highest_risk_score": highest.get("risk_score") if highest else None,
                "highest_risk_drivers": highest.get("drivers") if highest else None,
                "highest_confidence": highest.get("confidence") if highest else None,
                "highest_recommended": highest.get("recommended_action") if highest else None,
                "risk_levels": {
                    r: v.get("risk_level") for r, v in regions.items()
                }
            }

            self._write_file(self.summary_log_name, summary)
            reporting_cfg = self.config.get("reporting", {})
            generate_report = reporting_cfg.get("generate_human_report", True)

            if generate_report:
                human_context = {
                    "audit": audit_out,
                    "explainability": explain_out
                }

                md_path = build_human_readable_report(
                    context=human_context,
                    output_path=reporting_cfg.get(
                        "human_report_path",
                        os.path.join(self.output_dir, "human_audit_report.md")
                    )
                )
                pdf_path = md_path.replace(".md", ".pdf")
                build_pdf(md_path, pdf_path)

                summary_md = build_executive_summary(summary)

                build_pdf(summary_md, summary_md.replace(".md", ".pdf"))

            return final_output

        except Exception as e:

            return {
                "pipeline_status": "failure",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "execution_trace": execution_trace
            }
