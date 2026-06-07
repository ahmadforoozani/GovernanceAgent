# GovernanceAgent

GovernanceAgent is a production-oriented multi-agent governance pipeline for data validation, statistical monitoring, causal policy simulation, explainability, and audit reporting.

The system is designed to support:
- explicit data contract validation
- weekly anomaly monitoring using EWMA/CUSUM-style statistical reasoning
- causal policy simulation for vaccination-related signals
- escalation of risks into actionable governance decisions
- explainable audit logging and human review
- failure injection and stress testing

## Architecture

The pipeline is orchestrated by `GovernanceOrchestrator` and coordinated through the following agents:

- **ValidationAgent**  
  Validates dataset structure, null ratios, frequency expectations, monotonic cumulative fields, and cross-variable constraints.

- **StatisticalReasoningAgent**  
  Detects abnormal patterns in target metrics using threshold-based statistical reasoning.

- **CausalPolicyAgent**  
  Estimates policy-relevant causal effects using configurable causal simulation settings.

- **RiskEscalationAgent**  
  Aggregates validation, statistical, and causal signals into risk classifications and recommended actions.

- **AuditAgent**  
  Produces audit-ready records and approval-oriented outputs.

- **ExplainabilityAuditBuilder**  
  Builds human-readable trace artifacts for review, reporting, and verification.

## Key Outputs

Running the pipeline generates structured artifacts under `outputs/`, including:

- `audit_log.json`
- `pipeline_full_log.json`
- `pipeline_summary.json`
- `executive_summary.md`
- `executive_summary.pdf`
- `human_audit_report.md`
- `human_audit_report.pdf`

These outputs support traceability, verification, and human approval workflows.

## Configuration

Main configuration is defined in:

- `app/config.yaml`

Important sections include:

- dataset path and metadata
- validation thresholds
- cumulative and new variable definitions
- statistical monitoring parameters
- causal policy parameters

dataset_path: dataset/weekly_merged_dataset.csv
first, unzip the dataset
## Requirements

- Python 3.11+
- pandas
- numpy
- PyYAML
- scipy
- scikit-learn
- reportlab
- matplotlib
- markdown
- pytest

## One-Command Run
    first, unzip the dataset in dataset folder
    python -m app.main

    pytest -v tests/integration_full_pipeline_test.py
    pytest -v tests/test_failure_injection.py

    docker build -t governance-system .
    docker run governance-system

## Installation
    pip install -r requirements.txt