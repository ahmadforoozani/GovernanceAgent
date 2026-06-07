"""
End-to-end integration tests for Governance Pipeline.

Goals:
- verify agents interact correctly through context
- verify required schema fields required by the task
- verify outputs are generated correctly
"""

import os
import json
import yaml
import pandas as pd
import pytest
from pathlib import Path

from core.governance_orchestrator import GovernanceOrchestrator
from agents.validation_agent import ValidationAgent
from agents.statistical_reasoning_agent import StatisticalReasoningAgent
from agents.causal_simulation_agent import CausalSimulationAgent
from agents.risk_escalation_agent import RiskEscalationAgent
from agents.audit_agent import AuditAgent
from agents.explainability_audit_builder import ExplainabilityAuditBuilder


# ------------------------------------------------------------
# Base utilities
# ------------------------------------------------------------

def load_config():
    base_dir = Path(__file__).resolve().parents[1]
    config_path = base_dir / "app" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config, base_dir


@pytest.fixture(scope="session")
def app_config():

    config, _ = load_config()

    return config

# ------------------------------------------------------------
# Dataset fixture
# ------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_df():

    config, base_dir = load_config()
    data_cfg = config["data"]

    data_path = base_dir / data_cfg["dataset_path"]

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)

    date_col = data_cfg["date_column"]
    loc_col = data_cfg["location_column"]

    df[date_col] = pd.to_datetime(df[date_col])

    # Only countries
    df = df[~df[loc_col].str.contains("_")]

    # 20 Random Countries for testing
    countries = df[loc_col].drop_duplicates().sample(20, random_state=42)

    df = df[df[loc_col].isin(countries)].copy()

    df = df.sort_values([loc_col, date_col]).reset_index(drop=True)

    return df


def get_agent_config(config, agent_name):

    return (
        config["agents"].get(agent_name, {}),
        config["data"]
    )

# ------------------------------------------------------------
# Context building fixtures (pipeline order)
# ------------------------------------------------------------


@pytest.fixture
def context_validation(sample_df, app_config):

    ctx = {}

    agent_cfg, data_cfg = get_agent_config(
        app_config,
        "validation_agent"
    )

    out = ValidationAgent(
        agent_cfg,
        data_cfg
    ).run(sample_df, ctx)

    assert out["status"] == "ok"
    assert "validation" in out
    assert "engineered" in out

    ctx["validation"] = out["validation"]

    return ctx


@pytest.fixture
def context_statistical(
    sample_df,
    context_validation,
    app_config
):

    ctx = dict(context_validation)

    agent_cfg, data_cfg = get_agent_config(
        app_config,
        "statistical_reasoning_agent"
    )

    out = StatisticalReasoningAgent(
        agent_cfg,
        data_cfg
    ).run(sample_df, ctx)

    assert out["status"] == "ok"
    assert "statistical" in out

    ctx["statistical"] = out["statistical"]

    return ctx


@pytest.fixture
def context_causal(
    sample_df,
    context_statistical,
    app_config
):

    ctx = dict(context_statistical)

    agent_cfg, data_cfg = get_agent_config(
        app_config,
        "causal_simulation_agent"
    )

    out = CausalSimulationAgent(
        agent_cfg,
        data_cfg
    ).run(sample_df, ctx)

    assert out["status"] == "ok"
    assert "causal" in out

    ctx["causal"] = out["causal"]

    return ctx


@pytest.fixture
def context_risk(
    sample_df,
    context_causal,
    app_config
):

    ctx = dict(context_causal)

    agent_cfg, data_cfg = get_agent_config(
        app_config,
        "risk_escalation_agent"
    )

    out = RiskEscalationAgent(
        agent_cfg,
    ).run(sample_df, ctx)

    assert out["status"] == "ok"
    assert "risk" in out

    ctx["risk"] = out["risk"]

    return ctx


@pytest.fixture
def context_audit(
    sample_df,
    context_risk,
    app_config
):

    ctx = dict(context_risk)

    agent_cfg, data_cfg = get_agent_config(
        app_config,
        "audit_agent"
    )

    out = AuditAgent(
        agent_cfg
    ).run(sample_df, ctx)

    assert out["status"] == "ok"
    assert "audit" in out

    ctx["audit"] = out["audit"]

    return ctx


@pytest.fixture
def context_explainability(
    sample_df,
    context_audit,
    app_config
):

    ctx = dict(context_audit)

    agent_cfg, data_cfg = get_agent_config(
        app_config,
        "explainability_audit_builder"
    )

    out = ExplainabilityAuditBuilder(
        agent_cfg,
    ).run(sample_df, ctx)

    assert out["status"] == "ok"
    assert "explainability" in out

    ctx["explainability"] = out["explainability"]

    return ctx


# ------------------------------------------------------------
# Full pipeline test
# ------------------------------------------------------------

def test_full_pipeline_execution(
    tmp_path,
    sample_df,
    monkeypatch,
    app_config
):

    monkeypatch.chdir(tmp_path)

    orch = GovernanceOrchestrator(app_config)

    result = orch.run(sample_df)

    assert isinstance(result, dict)
    assert result["pipeline_status"] == "success"

    for key in [
        "validation",
        "statistical",
        "engineered",
        "causal",
        "risk",
        "audit",
        "explainability",
    ]:
        assert key in result

    expected_files = [
        "outputs/human_audit_report.md",
        "outputs/pipeline_full_log.json",
        "outputs/pipeline_summary.json",
    ]

    for f in expected_files:
        assert os.path.exists(f)

    with open("outputs/pipeline_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    assert "high_risk_regions" in summary


# ------------------------------------------------------------
# Validation agent schema
# ------------------------------------------------------------

def test_validation_schema(context_validation):

    val = context_validation["validation"]

    assert "issues" in val
    assert isinstance(val["issues"], dict)

    assert val["issues"], "No validation issues found"

    first_key = next(iter(val["issues"]))  # get first record id
    first_issue_list = val["issues"][first_key]
    assert first_issue_list, f"No issues for record {first_key}"

    issue = first_issue_list[0]  # take first issue dict

    # Schema validation for important fields
    for field in ["anomaly_id", "confidence", "model_trigger", "agent_version"]:
        assert field in issue, f"{field} missing in validation issue"

    assert issue["model_trigger"] == "ValidationAgent"


# ------------------------------------------------------------
# Statistical anomalies schema
# ------------------------------------------------------------

def test_statistical_schema(context_statistical):

    assert isinstance(
        context_statistical["statistical"]["formatted_messages"], list)
    anomalies = context_statistical["statistical"]["anomalies"]
    assert isinstance(anomalies, list)

    if anomalies:
        a = anomalies[0]
        assert "anomaly_id" in a
        assert "risk_level" in a
        assert "confidence" in a
        assert "model_trigger" in a
        assert a["model_trigger"] == "StatisticalReasoningAgent"


# ------------------------------------------------------------
# Causal simulation schema
# ------------------------------------------------------------

def test_causal_schema(context_causal):

    causal = context_causal["causal"]

    first_loc = next(iter(causal))

    entry = causal[first_loc]

    assert "model_trigger" in entry
    assert entry["model_trigger"] == "CausalSimulationAgent"

    trace = entry["causal_trace"]

    assert "scenarios" in trace

    scenarios = trace["scenarios"]

    assert "vaccination_plus_20" in scenarios
    assert "vaccination_minus_20" in scenarios


# ------------------------------------------------------------
# Risk escalation schema
# ------------------------------------------------------------

def test_risk_schema(context_risk):

    risk = context_risk["risk"]["locations"]

    loc = next(iter(risk))

    entry = risk[loc]

    assert "model_trigger" in entry
    assert entry["model_trigger"] == "RiskEscalationAgent"
    assert "escalation_id" in entry
    assert "risk_score" in entry
    assert "risk_level" in entry
    assert "recommended_action" in entry
    assert "explanation" in entry
    assert "escalation_trace" in entry


# ------------------------------------------------------------
# Audit report schema
# ------------------------------------------------------------

def test_audit_schema(context_audit):

    report = context_audit["audit"]["report"]

    assert "regions" in report
    assert "global_summary" in report
    region = next(iter(report["regions"]))

    entry = report["regions"][region]

    assert entry["model_trigger"] == "AuditAgent"
    assert "audit_id" in entry
    assert "recommended_action" in entry
    assert "governance_reasoning" in entry
    assert "explainability_trace" in entry


# ------------------------------------------------------------
# Explainability schema
# ------------------------------------------------------------

def test_explainability_schema(context_explainability):

    explain = context_explainability["explainability"]

    assert "global_graph" in explain

    region = next(iter(explain["regions"]))

    entry = explain["regions"][region]

    assert entry["model_trigger"] == "ExplainabilityAuditBuilder"

    assert "explain_id" in entry
    assert "fusion_inputs" in entry
    assert "fusion_contribution" in entry
    assert "reasoning_chain" in entry
