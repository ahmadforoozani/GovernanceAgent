"""
Human Readable Governance Report Builder – v1.3
Fully corrected:
- Data quality normalization
- Dual-layer risk drivers + fusion contributions
- Correct scenario sign semantics (policy-oriented)
- Robust explainability handling
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

AGENT_VERSION = "1.3-human-report-builder"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _format_float(value, precision=3):
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return "N/A"

# ------------------------------------------------------------
# Main Builder
# ------------------------------------------------------------


def build_human_readable_report(
    context: Dict[str, Any],
    output_path: str = "outputs/human_audit_report.md",
) -> str:

    audit_section = context.get("audit", {})
    explainability_regions = context.get(
        "explainability", {}).get("regions", {})
    report = audit_section.get("report", {})
    regions = report.get("regions", {})

    lines = []

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    lines.append("# Multi-Layer Governance Audit Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()} UTC")
    lines.append(f"Report Builder Version: {AGENT_VERSION}")
    lines.append("")

    # ------------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------------

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Total Regions Evaluated: **{len(regions)}**")
    lines.append(
        "- Governance Layers: Validation → Statistical → Causal → Risk → Audit")
    lines.append(
        "- Output Format: Human-readable multi-layer risk explanation")
    lines.append("")

    # ------------------------------------------------------------
    # Per Region
    # ------------------------------------------------------------

    for region, info in regions.items():

        risk_level = info.get("risk_level", "N/A")
        risk_score = _format_float(info.get("risk_score"))
        approval_status = info.get("approval_status", "N/A")
        recommendation = info.get("recommended_action", "N/A")
        governance_reasoning = info.get("governance_reasoning", "")

        confidence = _format_float(info.get("confidence_score", 0.0))

        explainability = info.get("explainability_trace", {}) or {}
        drivers = explainability.get("risk_drivers", {}) or {}

        explain_layer = explainability_regions.get(region, {})
        fusion_contrib = explain_layer.get("fusion_contribution", {})

        stat_c = _format_float(fusion_contrib.get(
            "contribution_statistical", 0.0))
        cfrag_c = _format_float(fusion_contrib.get("contribution_causal", 0.0))
        dq_c = _format_float(fusion_contrib.get(
            "contribution_data_quality", 0.0))

        data_quality = info.get("data_quality", {})
        dq_issues = data_quality.get("issues", [])

        # Safeguard: Flatten or interpret dq_issues
        if isinstance(dq_issues, dict):
            flat = []
            for loc, issues_list in dq_issues.items():
                for issue in issues_list:
                    if isinstance(issue, dict):
                        issue.setdefault("location_key", loc)
                    flat.append(issue)
            dq_issues = flat
        elif isinstance(dq_issues, list):
            # if strings or mixed types
            new = []
            for issue in dq_issues:
                if isinstance(issue, dict):
                    new.append(issue)
                else:
                    # wrap string as dict
                    new.append(str(issue))
            dq_issues = new

        statistical_summary = info.get("statistical_summary", [])
        causal_summary = info.get("causal_summary", {})

        lines.append(
            "-----------------------------------------------------------------")
        lines.append(f"## Region: {region}")
        lines.append("")

        # ------------------------------------------------------------
        # Risk Overview
        # ------------------------------------------------------------

        lines.append("### 1. Risk Overview")
        lines.append(f"- Risk Level: **{risk_level}**")
        lines.append(f"- Risk Score: `{risk_score}`")
        lines.append(f"- Confidence: `{confidence}`")
        lines.append(f"- Approval Status: `{approval_status}`")
        lines.append(f"- Recommended Action: **{recommendation}**")
        lines.append("")

        # ------------------------------------------------------------
        # Risk Drivers (Escalation Layer)
        # ------------------------------------------------------------
        stat_sc = _format_float(drivers.get('statistical'))
        causal_sc = _format_float(drivers.get('causal_fragility'))
        data_sc = _format_float(drivers.get('data_quality'))
        lines.append("Risk Drivers:")
        lines.append(
            f"- Statistical Risk: `{stat_sc}`")
        lines.append(
            f"- Causal Fragility: `{causal_sc}`")
        lines.append(
            f"- Data Quality Risk: `{data_sc}`")
        lines.append("")

        # ------------------------------------------------------------
        # Data Quality
        # ------------------------------------------------------------

        lines.append("### 2. Validation & Data Quality Findings")

        if dq_issues:
            # lines.append(f"Total data quality issues: {len(dq_issues)}")
            for issue in dq_issues:

                if not isinstance(issue, dict):
                    lines.append(f"- {issue}")
                    continue

                risk = issue.get("risk_level", "unknown")
                metric = issue.get("metric", "unknown_metric")
                date = issue.get("date", "all records")
                loc = issue.get("location_key", region)
                problem = issue.get("type", "issue")
                observed = issue.get("observed_value", "NA")
                expected = issue.get("expected_condition", "")
                reason = issue.get("why_flagged", "")
                action = issue.get("recommended_action", "")

                lines.append(
                    f"- [{risk.upper()}] {loc} | {metric} on {date}: "
                    f"observed={observed} expected={expected} → {problem}. "
                    f"Reason: {reason}. Suggested action: {action}"
                )
        else:
            lines.append("- No major structural validation issues detected.")

        lines.append("")

        # ------------------------------------------------------------
        # Statistical Summary
        # ------------------------------------------------------------

        lines.append("### 3. Statistical Governance Summary")

        import re

        if statistical_summary:

            def extract_week_number(text):
                """
                Extracts the week number from a string like:
                'AR, Week 52: ...'
                Returns integer 52.
                """
                match = re.search(r"Week\s+(\d+)", text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                return 10**9  # if no week found → push to bottom

            # Sort by week number ASC
            sorted_items = sorted(statistical_summary, key=extract_week_number)

            for item in sorted_items:
                lines.append(f"- {item}")

        else:
            lines.append("- No statistical anomalies reported.")

        lines.append("")

        # ------------------------------------------------------------
        # Causal Summary
        # ------------------------------------------------------------
        lines.append("### 4. Causal Simulation & Fragility")
        lines.append(causal_summary.get('explanation')
                     or "No causal explanation available.")
        lines.append("")
        # ------------------------------------------------------------
        # Fusion Contributions
        # ------------------------------------------------------------
        lines.append("")
        lines.append("### 5. Governance Rationale & Strategic Reflection")

        # lines.append(f"**Strategic Reflection – {region}**")
        lines.append(
            f"The fused governance risk score is **{risk_score}**, corresponding to a "
            f"risk level of **{risk_level}** and confidence of **{confidence}** "
        )
        lines.append(
            "Fusion contributions indicate the relative influence of each governance "
            f"layer: Statistical `{stat_c}`, Causal Fragility `{cfrag_c}`, Data Quality Risk `{dq_c}`."
        )
        lines.append("")

        lines.append(
            f"The recommended action is **{recommendation}** and approval status is **{approval_status}**. ")

        lines.append("")

    # ------------------------------------------------------------
    # End of Report
    # ------------------------------------------------------------

    lines.append("------------------------------------------------")
    lines.append("## End of Report")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    safe_lines = ["" if x is None else str(x) for x in lines]
    output_file.write_text("\n".join(safe_lines), encoding="utf-8")

    return str(output_file)


def build_executive_summary(context, output_path="outputs/executive_summary.md"):

    lines = []

    lines.append("# Governance Risk Executive Summary")
    lines.append("")
    lines.append(f" Pipeline ID: {context['pipeline_id']} ")
    lines.append("")
    lines.append(f"Generated: {context['generated_at']} UTC")
    lines.append("")

    lines.append("## System Risk Overview")
    lines.append("")
    lines.append(f"- Regions evaluated: **{context['num_regions']}**")
    lines.append(
        f"- Critical risk regions: **{context['critical_risk_regions']}**")
    lines.append(f"- High risk regions: **{context['high_risk_regions']}**")
    lines.append(
        f"- Medium risk regions: **{context['medium_risk_regions']}**")
    lines.append(f"- Low risk regions: **{context['low_risk_regions']}**")
    lines.append("")

    lines.append("## Highest Risk Location")
    lines.append("")
    lines.append(f"- Region: **{context['highest_risk_location']}**")
    lines.append(f"- Risk Level: **{context['highest_risk_level']}**")
    lines.append(f"- Risk Score: `{context['highest_risk_score']:.3f}`")
    drivers = context['highest_risk_drivers']
    stat_sc = _format_float(drivers.get('statistical'))
    causal_sc = _format_float(drivers.get('causal_fragility'))
    data_sc = _format_float(drivers.get('data_quality'))
    lines.append("Risk Drivers:")
    lines.append(f"-- Statistical Risk: `{stat_sc}`")
    lines.append(f"-- Causal Fragility: `{causal_sc}`")
    lines.append(f"-- Data Quality Risk: `{data_sc}`")
    lines.append(f"- Risk confidence: `{context['highest_confidence']:.3f}`")
    lines.append(f"- Recommended Action: `{context['highest_recommended']}`")
    lines.append("")

    lines.append("## Governance Recommendation")
    lines.append("")
    lines.append(
        "Immediate investigation recommended for critical regions. "
        "Statistical deviations combined with causal fragility may "
        "indicate structural shifts in epidemiological dynamics."
    )
    lines.append("")

    lines.append("## Governance Pipeline")
    lines.append("")
    lines.append(
        "Data Validation → Statistical Inference → Causal Simulation → "
        "Risk Fusion → Governance Audit"
    )

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    return output_path
