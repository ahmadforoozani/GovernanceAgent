import pandas as pd
import numpy as np
import scipy.stats as stats
import uuid
import warnings
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class StatisticalReasoningAgent:
    """
    Purpose:
        Detect statistical anomalies and structural changes in time-series data
        across multiple locations and metrics using a hybrid statistical approach.

    Core Techniques:
        - EWMA (Exponentially Weighted Moving Average) for adaptive baseline estimation
        - Z-score deviation detection
        - CUSUM for cumulative shift detection
        - Welch’s t-test for distribution drift
        - Structural break detection via rolling window comparison

    Input: 
        - Data as Dataframe
        - Context as Dictionary
    Output:
        Returns structured anomaly records including:
            - Risk classification (low, medium, high, critical)
            - Confidence score
            - Statistical trace metadata
            - Human-readable explanation
            - Location-level risk scoring
        In Json Format:
        {
            "status": ....,
                "agent": "StatisticalReasoningAgent",
                "statistical": {
                    "anomalies": ....,
                    "formatted_messages": ....,
                    "summary_scores": ....,
                }
        }

    """

    AGENT_VERSION = "1.0"

    def __init__(self, config, data_config):
        """
        Initialize statistical detection parameters and dataset schema.
        The design keeps backward compatibility with previous versions.
        """
        logging.info("Initializing Statistical Reasoning Agent.")
        self.config = config

        # Dataset schema
        self.date_column = data_config.get("date_column", "date")
        self.location_column = data_config.get(
            "location_column", "location_key")

        # Metrics to analyze (backward compatible default)
        self.metrics = data_config.get(
            "metric_columns",
            ["new_confirmed", "new_deceased"]
        )

        # EWMA smoothing factor
        self.ewma_alpha = config.get("ewma_alpha", 0.3)

        # Significance level for drift detection
        self.drift_alpha = config.get("drift_alpha", 0.01)

        # Risk classification thresholds (Z-score)
        self.z_medium = config.get("z_medium", 1.96)
        self.z_high = config.get("z_high", 2.58)
        self.z_critical = config.get("z_critical", 3.5)

        # Minimum deviation required before running expensive tests
        self.z_threshold = config.get("zscore_threshold", 1.3)

        # Window size for statistical comparisons
        self.window_size = config.get("window_size", 4)

        # Window size for scoring
        self.score_window = config.get("score_window", 30)

        # CUSUM sensitivity parameters
        self.cusum_k = config.get("cusum_k", 0.5)
        self.cusum_h = config.get("cusum_h", 5)

        # Minimum number of consecutive significant deviations
        self.persistence_steps = config.get("persistence_steps", 2)

    # ---------- Helper: JSON-safe conversion ----------
    @staticmethod
    def _safe_float(x):
        """Convert numeric values to JSON-safe floats."""
        if x is None:
            return None
        try:
            if isinstance(x, (np.floating, float)):
                if np.isnan(x) or np.isinf(x):
                    return None
            return float(x)
        except Exception:
            return None

    # ---------- EWMA baseline computation ----------
    def ewma(self, series):
        """
        Compute Exponentially Weighted Moving Average baseline.
        This provides a smooth expectation of the time-series.
        """
        if len(series) == 0:
            return np.array([])

        ew, prev = [], series[0]
        for x in series:
            prev = self.ewma_alpha * x + (1 - self.ewma_alpha) * prev
            ew.append(prev)

        return np.array(ew)

    # ---------- CUSUM change detection ----------
    def cusum(self, series, mean, std):
        """
        CUSUM with scaling by standard deviation.
        Helps detect cumulative shifts in the residual signal.
        """

        if std is None or std < 1e-6:
            return [False] * len(series)

        k_eff = self.cusum_k * std
        h_eff = self.cusum_h * std

        pos, neg = 0.0, 0.0
        signals = []

        for x in series:
            pos = max(0.0, pos + (x - mean - k_eff))
            neg = min(0.0, neg + (x - mean + k_eff))
            signals.append(pos > h_eff or abs(neg) > h_eff)

        return signals

    # ---------- Safe t-test for distribution drift ----------
    def _safe_drift_test(self, w1, w2):
        """
        Perform Welch t-test between two windows.
        Used to detect distribution drift while avoiding numerical issues.
        """

        if len(w1) < 3 or len(w2) < 3:
            return False, 1.0

        if np.std(w1) < 1e-6 and np.std(w2) < 1e-6:
            return False, 1.0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                _, p_value = stats.ttest_ind(w1, w2, equal_var=False)

            if p_value is None or np.isnan(p_value):
                return False, 1.0

            return bool(p_value < self.drift_alpha), float(p_value)

        except Exception:
            return False, 1.0

    # ---------- Core pipeline execution ----------
    def run(self, data: pd.DataFrame, context: dict):
        """
        Main anomaly detection routine.

        The algorithm runs independently for each:
        - location
        - metric

        Statistical techniques used:
        - EWMA baseline
        - Z-score anomaly detection
        - CUSUM shift detection
        - Welch t-test drift detection
        """
        logging.info("Running Statistical Reasoning Agent.")
        anomalies = []
        messages = []
        scores = {}

        df = data.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        # ---------- Iterate through each location ----------
        for loc, g in df.groupby(self.location_column):

            g = g.sort_values(self.date_column)
            dates = g[self.date_column].values

            location_abs_z = []

            # ---------- Run analysis for each configured metric ----------
            for metric in self.metrics:

                if metric not in g.columns:
                    continue

                x = g[metric].fillna(0).values.astype(float)

                # Skip short series
                if len(x) < self.window_size * 2:
                    continue

                # ---------- Compute baseline ----------
                baseline = self.ewma(x)

                # # freeze baseline for last persistence window to prevent shock adaptation
                # freeze_k = max(self.persistence_steps, 3)
                # baseline[-freeze_k:] = baseline[-freeze_k-1]

                if len(baseline) == 0:
                    continue

                residuals = x - baseline

                sigma = np.std(residuals)
                sigma = 1.0 if sigma < 1e-6 else sigma

                z_scores = residuals / sigma

                ewma_std = sigma * \
                    np.sqrt(self.ewma_alpha / (2 - self.ewma_alpha))

                UCL = baseline + 3 * ewma_std
                LCL = baseline - 3 * ewma_std

                # Detect cumulative shifts
                cusum_signals = self.cusum(residuals, 0.0, sigma)

                # ---------- Iterate through time points ----------
                # Track consecutive significant deviations per metric
                consecutive_hits = 0

                for i in range(len(x)):

                    val = x[i]
                    date = pd.to_datetime(dates[i])

                    z = z_scores[i]
                    absz = abs(z)

                    location_abs_z.append(absz)

                    # persistence tracking
                    if absz >= self.z_medium:
                        consecutive_hits += 1
                    else:
                        consecutive_hits = 0

                    distribution_drift = False
                    p_value = None
                    structural_break = False
                    cusum_flag = bool(cusum_signals[i])

                    # ---------- Statistical gate to reduce noise ----------
                    if absz >= self.z_threshold:

                        if i >= 2 * self.window_size:

                            w1 = x[i - 2 * self.window_size: i -
                                   self.window_size]
                            w2 = x[i - self.window_size: i]

                            distribution_drift, p_value = self._safe_drift_test(
                                w1, w2)

                            prev_mean = np.mean(w1)
                            curr_mean = np.mean(w2)

                            pooled_std = np.std(w1)
                            pooled_std = 1.0 if pooled_std < 1e-6 else pooled_std

                            if abs(curr_mean - prev_mean) > 3 * pooled_std:
                                structural_break = True

                    else:
                        distribution_drift = False
                        structural_break = False
                        cusum_flag = False

                    ewma_shift = bool(x[i] > UCL[i] or x[i] < LCL[i])

                    # ---------- Risk classification ----------
                    critical_signal = (
                        absz >= self.z_critical and (
                            structural_break
                            or distribution_drift
                            or cusum_flag
                        )
                    )
                    if critical_signal:
                        risk = "critical"
                    elif absz >= self.z_high:
                        risk = "high"
                    elif absz >= self.z_medium:
                        risk = "medium"
                    else:
                        risk = "low"

                    confidence = min(1.0, absz / 3.0)

                    # ---------- Human-readable explanation ----------
                    explanation = (
                        f"{loc}, Week {date.isocalendar().week}, {metric}: "
                        f"{absz:.2f}σ deviation from EWMA baseline "
                    )

                    if absz >= self.z_critical:
                        explanation += "exceeds extreme-event threshold (>99.95% CI)"
                    elif absz >= self.z_high:
                        explanation += "exceeds 99% CI"
                    elif absz >= self.z_medium:
                        explanation += "exceeds 95% CI"
                    else:
                        explanation += "within normal confidence interval"

                    if structural_break:
                        explanation += ", probable structural shift"

                    if distribution_drift:
                        explanation += ", distribution drift detected (p<0.01)"

                    if cusum_flag:
                        explanation += ", cumulative shift detected (CUSUM)"

                    explanation += "."

                    p_value_out = self._safe_float(p_value)

                    if risk == "critical":
                        action = "immediate_escalation"
                    elif risk == "high":
                        action = "escalate"
                    elif risk == "medium":
                        action = "monitor"
                    else:
                        action = "none"

                    # ---------- Anomaly record ----------
                    record = {
                        "anomaly_id": str(uuid.uuid4()),
                        "model_trigger": "StatisticalReasoningAgent",
                        "agent_version": self.AGENT_VERSION,
                        "location_key": loc,
                        "date": str(date.date()),
                        "metric": metric,
                        "observed_value": float(val),
                        "z_score": float(z),
                        "confidence": float(confidence),
                        "risk_level": risk,
                        "recommended_action": action,
                        "interpretation": explanation,
                        "statistical_trace": {
                            "method": "EWMA + Z-score + CUSUM + DriftTest",
                            "alpha": self.ewma_alpha,
                            "baseline": float(baseline[i]),
                            "residual": float(residuals[i]),
                            "z_score": float(z),
                            "control_limits": {
                                "UCL": float(UCL[i]),
                                "LCL": float(LCL[i])
                            },
                            "distribution_drift": distribution_drift,
                            "p_value": p_value_out,
                            "structural_break": structural_break,
                            "ewma_shift": ewma_shift,
                            "cusum_signal": cusum_flag,
                        },
                    }

                    # ---------- Only report meaningful events ----------
                    persistence_ok = consecutive_hits >= self.persistence_steps

                    if (risk != "low" and persistence_ok) or structural_break or distribution_drift:

                        anomalies.append(record)
                        messages.append(explanation)

            # ---------- Location level summary score ----------
            if len(location_abs_z) > 0:
                z = np.array(location_abs_z)

                # Focus on recent behaviour
                recent_window = min(len(z), self.score_window)
                z_recent = z[-recent_window:]

                # -------------------------------
                # 1) Tail-sensitive severity
                # -------------------------------

                top_k = min(5, len(z_recent))
                top_values = np.sort(z_recent)[-top_k:]

                # Use RMS instead of mean to amplify extreme spikes
                severity = float(np.sqrt(np.mean(top_values ** 2)))

                # Dynamic scaling (no hard division by z_critical)
                severity_norm = 1 - np.exp(-2.5 * severity / self.z_critical)

                # -------------------------------
                # 2) Frequency (nonlinear boost)
                # -------------------------------

                freq_raw = float(np.mean(z_recent >= self.z_medium))

                # emphasize heavy anomaly regimes
                frequency = 1 - np.exp(-3 * freq_raw)

                # -------------------------------
                # 3) Persistence (run-length intensity)
                # -------------------------------

                runs = []
                current = 0
                for v in z_recent >= self.z_medium:
                    if v:
                        current += 1
                    else:
                        runs.append(current)
                        current = 0
                runs.append(current)

                max_run = max(runs) if len(runs) > 0 else 0

                persistence_raw = max_run / recent_window

                persistence = 1 - np.exp(-5 * persistence_raw)

                # -------------------------------
                # 4) Extreme spike bonus
                # -------------------------------

                extreme_bonus = 0.0
                if np.max(z_recent) > 3 * self.z_critical:
                    extreme_bonus = 0.15
                elif np.max(z_recent) > 2 * self.z_critical:
                    extreme_bonus = 0.1

                # -------------------------------
                # 5) Final statistical score
                # -------------------------------

                score = (
                    0.6 * severity_norm +
                    0.2 * frequency +
                    0.2 * persistence +
                    extreme_bonus
                )

                scores[loc] = float(min(score, 1))

        logging.info("Generating Statistical Reasoning Agent's Json output.")
        return {
            "status": "ok",
            "agent": "StatisticalReasoningAgent",
            "statistical": {
                "anomalies": anomalies,
                "formatted_messages": messages,
                "summary_scores": scores,
            },
        }
