import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from core.governance_orchestrator import GovernanceOrchestrator
from core.config_loader import load_config


BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config():
    config_path = BASE_DIR / "app" / "config.yaml"
    return load_config(config_path)


def get_agent_config(config, agent_name):

    return (
        config["agents"].get(agent_name, {}),
        config["data"]
    )


@pytest.fixture(scope="session")
def sample_df(config):
    data_cfg = config["data"]

    data_path = BASE_DIR / data_cfg["dataset_path"]

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)

    date_col = data_cfg["date_column"]
    loc_col = data_cfg["location_column"]

    df[date_col] = pd.to_datetime(df[date_col])

    # Extracting only countries
    df = df[~df[loc_col].str.contains("_")]

    # 20 Random country
    countries = df[loc_col].drop_duplicates().sample(20, random_state=42)

    df = df[df[loc_col].isin(countries)].copy()

    df = df.sort_values([loc_col, date_col]).reset_index(drop=True)

    return df


def run(df, config):
    orch = GovernanceOrchestrator(config)
    return orch.run(df)


# -------------------------------
# Helper functions
# -------------------------------

def _safe_text(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join(str(i) for i in x).lower()
    if isinstance(x, dict):
        return " ".join(f"{k} {v}" for k, v in x.items()).lower()
    return str(x).lower()


def _collect_audit_text(result):
    audit = result.get("audit", {})
    chunks = []

    if isinstance(audit, dict):
        for v in audit.values():
            chunks.append(_safe_text(v))
    else:
        chunks.append(_safe_text(audit))

    return " ".join(chunks).lower()


def _collect_risk_text(result):
    risk = result.get("risk", {})
    chunks = []

    if isinstance(risk, dict):
        for v in risk.values():
            chunks.append(_safe_text(v))
    else:
        chunks.append(_safe_text(risk))

    return " ".join(chunks).lower()


def _get_anomalies(result):
    return result.get("statistical", {}).get("anomalies", [])


def _get_validation_issues(result):
    validation = result.get("validation", {})
    if isinstance(validation, dict):
        return validation.get("issues", [])
    return []


def _find_val_issues(result, rule=None):
    loc_issues = _get_validation_issues(result)
    for loc, issues in loc_issues.items():
        if rule is None:
            return issues
        return [i for i in issues if i.get("validation_rule") == rule]


def _risk_locations(result):
    risk = result.get("risk", {})
    locs = risk.get("locations", [])
    if locs is None:
        return []
    return locs


def _parse_date(x):
    return pd.to_datetime(x)


def _z(a):
    return abs(a.get("statistical_trace", {}).get("z_score", 0))


def _interp(a):
    return str(a.get("interpretation", "")).lower()


def _anomaly_matches_keywords(a, keywords):
    txt = " ".join([
        _interp(a),
        _safe_text(a.get("statistical_trace", {}))
    ]).lower()
    return any(k.lower() in txt for k in keywords)


# ============================================================
#  Test 0 — Clean Stable Data (Direct FP Test)
# ============================================================

def test_clean_data_false_positive_baseline(sample_df, config):

    df = sample_df.copy().head(1000)

    result = run(df, config)
    anoms = result["statistical"]["anomalies"]

    stat_cfg = config.get("statistical_agent")
    z_high = stat_cfg.get("z_high", 2.58)
    z_critical = stat_cfg.get("z_critical", 3.5)

    # Few FP should be generated
    n_strong = len([a for a in anoms if abs(
        a["statistical_trace"].get("z_score", 0)) >= z_high])
    n_extreme = len([a for a in anoms if abs(
        a["statistical_trace"].get("z_score", 0)) >= z_critical])

    assert n_strong <= 0.07 * len(df)
    assert n_extreme <= 0.03 * len(df)


# ============================================================
# 1) Missing Weeks
# ============================================================

def test_missing_weeks_detection(sample_df, config):
    df0 = sample_df.copy()

    sampled = df0["date"].drop_duplicates(
    ).sort_values().sample(5, random_state=1)
    missing_dates = pd.to_datetime(sampled).sort_values()

    injected_start = missing_dates.min()
    injected_end = missing_dates.max()

    affected_rows_before_drop = df0[df0["date"].isin(missing_dates)].copy()
    affected_locations = set(
        affected_rows_before_drop["location_key"].unique())

    df = df0[~df0["date"].isin(missing_dates)].copy()

    result = run(df, config)
    assert result["pipeline_status"] == "success"

    anoms = _get_anomalies(result)
    assert len(anoms) > 0, "Pipeline returned no anomalies at all."

    # 1) DETECTION: Anomaly should be around missing dates
    window_hits = []
    for a in anoms:
        if "date" not in a:
            continue
        d = _parse_date(a["date"])
        if injected_start <= d <= injected_end and _anomaly_matches_keywords(
            a, ["missing", "gap", "week", "deviation", "shift"]
        ):
            window_hits.append(a)

    assert len(window_hits) > 0, \
        "Injected missing-week window must be explicitly detected."

    # 2) RESPONSE: risks related to missing date locations
    risk_locs = set(_risk_locations(result))
    assert len(risk_locs) > 0, \
        "Risk output must contain affected locations for missing-week anomaly."

    assert len(risk_locs.intersection(affected_locations)) > 0, \
        "Risk must reference the actual locations impacted by missing dates."

    # 3) LOGGING / AUDIT
    audit_text = _collect_audit_text(result)
    assert any(k in audit_text for k in ["missing", "gap", "week"]), \
        "Audit must explicitly log the missing-week anomaly."

    # 4) FALSE POSITIVE CONTROL
    stat_cfg = config.get("statistical_agent")
    z_high = stat_cfg.get("z_high", 2.58)
    outside_window_fp = []
    for a in anoms:
        if "date" not in a:
            continue
        d = _parse_date(a["date"])
        if not (injected_start <= d <= injected_end) and _z(a) >= z_high:
            outside_window_fp.append(a)

    assert len(outside_window_fp) < max(3, int(0.07 * len(df))), \
        "Missing-week injection must not cause excessive FP outside injected window."


# ============================================================
# 2) Negative counts
# ============================================================
def test_negative_counts_detection(sample_df, config):
    df = sample_df.copy()

    neg_idx = df.sample(10, random_state=2).index
    df.loc[neg_idx, "new_confirmed"] = -50

    injected_locations = set(df.loc[neg_idx, "location_key"].unique())

    result = run(df, config)
    assert result["pipeline_status"] == "success"

    # 1) DETECTION: exact validation rule
    issues = _find_val_issues(result, rule="non_negative_daily_values")
    assert len(issues) > 0, \
        "Negative counts must trigger non_negative_daily_values rule."

    matched_neg_issues = [
        i for i in issues
        if i.get("observed_value", 0) < 0
    ]
    assert len(matched_neg_issues) > 0, \
        "Validation must record a negative observed_value for injected corruption."

    # 2) RESPONSE: risk
    risk_locs = set(_risk_locations(result))
    assert len(risk_locs.intersection(injected_locations)) > 0, \
        "Risk must respond to the same locations with injected negative counts."

    # 3) LOGGING
    audit_text = _collect_audit_text(result)

    assert any(k in audit_text for k in ["negative", "invalid", "non_negative", "value"]), \
        "Audit must explicitly mention negative/invalid values."

    # 4) FP CONTROL
    anoms = _get_anomalies(result)
    safe_locations = set(df["location_key"].unique()) - injected_locations

    stat_cfg = config.get("statistical_agent")
    z_high = stat_cfg.get("z_high", 2.58)
    fp = [
        a for a in anoms
        if a.get("location_key") in safe_locations and _z(a) >= z_high
    ]
    assert len(fp) < max(3, int(0.07 * len(df))), \
        "Negative-count injection must not create excessive anomalies in unaffected locations."


# ============================================================
# 3) Random Noise
# ============================================================

def test_random_noise_robustness(sample_df, config):
    df = sample_df.copy()

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 150, len(df))
    df["new_confirmed"] = df["new_confirmed"].fillna(0) + noise

    result = run(df, config)
    assert result["pipeline_status"] == "success"

    anoms = _get_anomalies(result)
    assert len(anoms) > 0, \
        "Heavy injected noise should produce detectable statistical anomalies."

    # 1) DETECTION
    traceful = [
        a for a in anoms
        if "statistical_trace" in a and a["statistical_trace"]
    ]
    assert len(traceful) > 0, \
        "Noise-induced anomalies must include statistical trace."

    # 2) LOGGING / EXPLAINABILITY
    audit_text = _collect_audit_text(result)
    assert any(k in audit_text for k in ["noise", "volatility", "deviation", "statistical"]), \
        "Audit must describe the statistical abnormality caused by injected noise."

    # 3) RESPONSE
    risk_text = _collect_risk_text(result)
    assert any(k in risk_text for k in ["uncertainty", "anomaly", "volatility", "monitor"]), \
        "Risk layer must react to noisy behavior, not merely exist."

    # 4) FP / OVER-FIRING CONTROL
    assert len(anoms) < int(0.40 * len(df)), \
        "Noise gating must prevent excessive anomaly inflation."


# ============================================================
# 4) Structural regime shift
# ============================================================

def test_regime_shift_detection(sample_df, config):
    df = sample_df.copy().sort_values(
        ["location_key", "date"]).reset_index(drop=True)

    # choosing one location
    target_loc = df["location_key"].dropna().iloc[0]
    loc_mask = df["location_key"] == target_loc
    loc_df = df.loc[loc_mask].sort_values("date")

    assert len(loc_df) >= 20, "Need enough rows for structural break injection."

    shift_start_idx = loc_df.index[len(loc_df) // 2]
    shift_start_date = df.loc[shift_start_idx, "date"]

    # regime shift
    df.loc[(df["location_key"] == target_loc) & (
        df["date"] >= shift_start_date), "new_confirmed"] += 300

    result = run(df, config)
    assert result["pipeline_status"] == "success"

    anoms = _get_anomalies(result)
    assert len(anoms) > 0, "Regime shift should produce anomalies."

    # 1) DETECTION: structural break shift for the target location
    breaks = [
        a for a in anoms
        if a.get("location_key") == target_loc
        and "date" in a
        and _parse_date(a["date"]) >= _parse_date(shift_start_date)
        and a.get("statistical_trace", {}).get("structural_break") is True
    ]

    assert len(breaks) > 0, \
        "Injected regime shift must be detected as structural break in the affected location."

    # 2) RESPONSE: risk escalation for target location
    risk_locs = set(_risk_locations(result))
    assert target_loc in risk_locs, \
        "Risk must escalate the exact location with injected regime shift."

    # 3) LOGGING / EXPLAINABILITY
    audit_text = _collect_audit_text(result)
    risk_text = _collect_risk_text(result)

    assert any(k in audit_text for k in ["break", "shift", "structural", "regime"]), \
        "Audit must explicitly mention structural/regime shift."

    assert any(k in risk_text for k in ["break", "shift", "structural", "regime", "escalat"]), \
        "Risk output must explicitly react to regime shift."

    # 4) FP CONTROL: shouldn't have many anomalies before dates shift
    stat_cfg = config.get("statistical_agent")
    z_high = stat_cfg.get("z_high", 2.58)
    pre_shift_fp = [
        a for a in anoms
        if a.get("location_key") == target_loc
        and "date" in a
        and _parse_date(a["date"]) < _parse_date(shift_start_date)
        and _z(a) >= z_high
    ]

    assert len(pre_shift_fp) < max(2, int(0.07 * len(loc_df))), \
        "Before regime-shift point, anomalies in the same location must remain limited."


# ============================================================
# 5) Corrupted categorical encoding
# ============================================================

def test_corrupted_categorical_encoding(sample_df, config):
    df = sample_df.copy()

    corrupt_idx = df.sample(8, random_state=7).index
    original_locations = set(df.loc[corrupt_idx, "location_key"].unique())

    df.loc[corrupt_idx, "location_key"] = "###INVALID###"

    result = run(df, config)
    if result["pipeline_status"] == "failure":
        print(result["error"])
        print(result["traceback"])
    assert result["pipeline_status"] == "success"

    # 1) DETECTION: validation/audit should reflect corruptionn in categorical field
    val_issues = _get_validation_issues(result)
    audit_text = _collect_audit_text(result)

    detected_categorical_problem = (
        any("location" in _safe_text(i) or "allowed format" in _safe_text(i) or "invalid" in _safe_text(i)
            for i in val_issues)
        or any(k in audit_text for k in ["invalid", "allowed format", "location_key", "encoding"])
    )

    assert detected_categorical_problem, \
        "Corrupted categorical encoding must be explicitly detected/logged."

    # 2) RESPONSE
    risk_text = _collect_risk_text(result)
    assert any(k in risk_text + " " + audit_text for k in ["invalid", "allowed format", "encoding", "data quality"]), \
        "System must react specifically to categorical corruption, not just survive it."

    # 3) FP CONTROL
    stat_cfg = config.get("statistical_agent")
    z_high = stat_cfg.get("z_high", 2.58)
    anoms = _get_anomalies(result)
    fp = [
        a for a in anoms
        if a.get("location_key") not in {"###INVALID###"}
        and a.get("location_key") not in original_locations
        and _z(a) >= z_high
    ]

    assert len(fp) < max(3, int(0.07 * len(df))), \
        "Categorical corruption must not produce broad false positives in unrelated groups."

    # 4) CONTAINMENT: Anomalies should be around target location
    corrupted_anoms = [
        a for a in anoms
        if a.get("location_key") == "###INVALID###"
    ]
    assert len(corrupted_anoms) < max(5, int(0.07 * len(df))), \
        "Corrupted category should be contained, not dominate anomaly output."

# ============================================================
# Extreme spike should produce critical anomalies
# ============================================================


def test_extreme_spike_produces_critical(sample_df, config):
    df = sample_df.copy().head(300)

    # Pick one location with enough data
    target_loc = df["location_key"].iloc[0]
    loc_mask = df["location_key"] == target_loc
    loc_df = df.loc[loc_mask]

    assert len(loc_df) > 30, "Need enough rows for injecting extreme anomaly."

    # Inject extreme spike: +300% jump (guaranteed z>10)
    spike_idx = loc_df.index[-1]
    df.loc[spike_idx, "new_confirmed"] = df.loc[spike_idx - 1, "new_confirmed"] * 4

    # Run pipeline
    result = run(df, config)
    assert result["pipeline_status"] == "success"

    anoms = _get_anomalies(result)

    assert len(anoms) > 0, "Extreme spike MUST produce anomalies."

    # Check statistical detection
    critical_hits = [
        a for a in anoms
        if a.get("location_key") == target_loc
        and _z(a) >= config["statistical_agent"].get("z_critical", 3.5)
    ]

    assert len(critical_hits) > 0, \
        "Extreme z-score anomaly MUST be labeled as critical."

    # Check audit / explainability
    audit_text = _collect_audit_text(result)
    assert any(k in audit_text for k in ["spike", "extreme", "surge", "outlier"]), \
        "Audit must explicitly reference extreme spike anomaly."

    # Check risk escalation
    risk_locs = _risk_locations(result)
    assert target_loc in risk_locs, \
        "Risk must escalate the location with extreme spike."

    print(target_loc, risk_locs[target_loc].get("risk_score", ""),
          risk_locs[target_loc].get("risk_level", ""),
          risk_locs[target_loc].get("drivers", ""))
    assert risk_locs[target_loc].get("risk_level", "") in ["high", "critical"]


# ============================================================
# Multi-day outbreak must escalate to critical
# ============================================================

def test_pipeline_extreme_event_escalation(sample_df, config):
    df = sample_df.copy().sort_values(
        ["location_key", "date"]).reset_index(drop=True)

    target_loc = df["location_key"].iloc[0]
    loc_mask = df["location_key"] == target_loc
    loc_df = df.loc[loc_mask]

    assert len(loc_df) >= 80, "Need 80+ days to train causal dependency."

    # ---------------------------
    # Data corruption phase
    # ---------------------------
    df.loc[df.index[0:5], "new_confirmed"] = -100
    df.loc[df.index[10], "cumulative_persons_fully_vaccinated"] = 1
    df.loc[df.index[20], "date"] = df.loc[df.index[19],
                                          "date"] + pd.Timedelta(days=50)

    # ---------------------------
    # A. Causal dependency (40 days)
    # vaccination up → cases down
    # ---------------------------
    phaseA = loc_df.index[-90:-50]
    for i, idx in enumerate(phaseA):
        vacc_boost = 1 + 0.03 * i
        case_drop = 1 / (1 + 0.05 * i)
        death_drop = 1 / (1 + 0.06 * i)
        df.loc[idx, "cumulative_persons_fully_vaccinated"] *= vacc_boost
        df.loc[idx, "new_confirmed"] *= case_drop
        df.loc[idx, "new_deceased"] *= death_drop

    # ---------------------------
    # B. Outbreak buildup (25 days)
    # sustained multi-week abnormal surge
    # ---------------------------
    phaseB = loc_df.index[-50:-25]
    for i, idx in enumerate(phaseB):
        case_mult = 8 + 0.25 * i
        death_mult = 9 + 0.30 * i
        df.loc[idx, "new_confirmed"] *= case_mult
        df.loc[idx, "new_deceased"] *= death_mult

    # ---------------------------
    # C. Collapse + runaway crisis (25 days)
    # severe vaccination collapse with worsening outbreak
    # ---------------------------
    phaseC = loc_df.index[-25:]
    v_max = df.loc[phaseA, "cumulative_persons_fully_vaccinated"].max()

    for i, idx in enumerate(phaseC):
        case_mult = 18 + 0.8 * i
        death_mult = 22 + 1.0 * i

        df.loc[idx, "new_confirmed"] *= case_mult
        df.loc[idx, "new_deceased"] *= death_mult

        # progressive vaccination collapse
        collapse_ratio = max(0.03, 0.15 - 0.004 * i)
        df.loc[idx, "cumulative_persons_fully_vaccinated"] = (
            collapse_ratio * v_max * np.random.uniform(0.9, 1.1)
        )

    result = run(df, config)

    assert result["pipeline_status"] == "success"

    dq_scores = result["validation"]["quality_scores"].get(target_loc, {})
    assert dq_scores.get("data_quality_severity") == "critical", \
        f"Expected critical, got {dq_scores.get('data_quality_severity')}"

    # --- Anomaly detection ---
    anoms = _get_anomalies(result)
    assert len(anoms) > 0, "Outbreak must yield statistical anomalies."

    high_or_critical = [
        a for a in anoms
        if a.get("location_key") == target_loc
        and _z(a) >= config["statistical_agent"].get("z_high", 2.58)
    ]

    assert len(high_or_critical) >= 3, \
        "Multi-day outbreak must generate multiple high/critical anomalies."

    # --- Risk Escalation ---
    risk_locs = _risk_locations(result)
    assert target_loc in risk_locs, \
        "Risk layer must escalate multi-day outbreak."

    print(target_loc, risk_locs[target_loc].get("risk_score", ""),
          risk_locs[target_loc].get("risk_level", ""),
          risk_locs[target_loc].get("drivers", ""))
    if risk_locs[target_loc].get("drivers", {}).get("causal_fragility", 0.0) > 0.7:
        assert risk_locs[target_loc].get("risk_level", "") == "critical"
    else:
        assert risk_locs[target_loc].get("risk_level", "") == "high"

    risk_text = _collect_risk_text(result)
    assert any(k in risk_text for k in ["critical", "high", "escalat", "severe"]), \
        "Risk text must reference high severity."
