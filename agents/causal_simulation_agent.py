import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import uuid
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class CausalSimulationAgent:
    """
    Causal simulation agent for evaluating vaccination policy impact.

    This agent builds a simplified Structural Causal Model (SCM) between:

        vaccination -> cases -> death

    The model is trained on historical epidemiological signals and then used
    to simulate future trajectories under different vaccination scenarios.
    Inputs:
        - Data as Dataframe
        - Context as Dictionary

    Output includes:
        - scenario simulations (baseline, vaccination +20%, vaccination -20%)
        - uncertainty intervals via Monte-Carlo simulation
        - policy recommendation signals for downstream governance agents
        In Json formart:
        {
            "status": "...",
            "agent": "CausalSimulationAgent",
            "agent_version": ....,
            "scm_graph": {
                "vaccination": ["cases"],
                "cases": ["death"],
                "death": []
            },
            "vaccine_lag_weeks": ...,
            "causal":
                {
                    status= ...,
                    location_key= ...,
                    causal_trace= ...,
                    recommended_action= ...,
                    confidence= ...,
                    risk_level= ...,
                    explanation= ...
                }
            )
        }
    """

    AGENT_VERSION = "1.0"

    def __init__(self, config, data_config):
        logging.info("Initializing Causal Simulation Agent.")
        self.date_column = data_config.get("date_column", "date")
        self.location_column = data_config.get(
            "location_column", "location_key")
        self.vacc_ratio_column = data_config.get(
            "vaccination_ratio_column",
            "vaccination_ratio"
        )

        self.vacc_cumulative_columns = data_config.get(
            "vaccination_cumulative_columns",
            [
                "cumulative_persons_fully_vaccinated",
                "cumulative_persons_vaccinated",
                "cumulative_vaccine_doses_administered"
            ]
        )

        # simulation horizon (number of future steps)
        # NOTE: because the dataset is weekly, each step roughly corresponds to one week
        self.horizon = config.get("horizon", 30)

        # regularization strength for Ridge regression used in SCM equations
        self.alpha = config.get("ridge_alpha", 0.5)

        # number of Monte‑Carlo bootstrap simulations for uncertainty estimation
        self.bootstrap_samples = config.get("bootstrap_samples", 100)

        # delay between vaccination and observable epidemiological impact
        # real-world immune protection takes time to build
        self.vaccine_lag_weeks = config.get("vaccine_lag_weeks", 2)

        # vaccination counterfactual scenarios
        # baseline = current level
        # +20% / -20% simulate policy changes in vaccination intensity
        self.scenario_multipliers = config.get(
            "scenario_multipliers",
            {
                "baseline": 1.0,
                "vaccination_plus_20": 1.2,
                "vaccination_minus_20": 0.8
            }
        )

        self.metrics = data_config.get(
            "metric_columns",
            ["new_confirmed", "new_deceased"]
        )

        # --- causal fragility configuration ---

        fragility_cfg = config.get("causal_fragility", {})

        self.fragility_mode = fragility_cfg.get(
            "mode",
            "symmetric"   # symmetric | downside
        )

        self.fragility_thresholds = fragility_cfg.get(
            "risk_thresholds",
            {
                "medium": 0.3,
                "high": 0.5,
                "critical": 0.7
            }
        )

        self.fragility_cap = fragility_cfg.get(
            "risk_cap",
            1.0
        )

    # ---------------- UTIL ----------------

    def _record(self, **kwargs):
        base = {
            "causal_id": str(uuid.uuid4()),
            "model_trigger": "CausalSimulationAgent",
            "agent_version": self.AGENT_VERSION
        }
        base.update(kwargs)
        return base

    def _num(self, x, default=0.0):
        try:
            x = float(x)
            if np.isnan(x) or np.isinf(x):
                return float(default)
            return x
        except Exception:
            return float(default)

    def _sanitize_array(self, x, default=0.0):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=default, posinf=default, neginf=default)

    def _safe(self, d, col):
        if col in d.columns:
            return self._sanitize_array(d[col].values)
        return np.zeros(len(d))

    def _safe_last(self, d, col):
        if col in d.columns and len(d) > 0:
            return self._num(d[col].iloc[-1], 0.0)
        return 0.0

    def _fragility_risk_level(self, fragility: float):

        t = self.fragility_thresholds

        if fragility >= t.get("critical", 0.7):
            return "critical"
        elif fragility >= t.get("high", 0.5):
            return "high"
        elif fragility >= t.get("medium", 0.3):
            return "medium"
        else:
            return "low"

    # ---------------- VACCINATION ----------------

    def _build_vaccination_ratio(self, d):
        """
        Build a normalized vaccination signal from available columns.

        Different datasets may provide different vaccination indicators.
        We try common cumulative vaccination metrics and normalize them
        into a [0,1] ratio representing relative vaccination progress.
        """
        vacc_source = None

        for col in self.vacc_cumulative_columns:
            if col in d.columns:
                vacc_source = self._sanitize_array(d[col].values)
                break

        if vacc_source is None:
            return np.zeros(len(d))

        max_v = self._num(np.nanmax(vacc_source), 0.0)
        if max_v <= 0:
            return np.zeros(len(d))

         # normalize cumulative values to a 0‑1 ratio
        ratio = vacc_source / max_v
        ratio = self._sanitize_array(ratio, default=0.0)

        # vaccination cannot exceed population
        ratio = np.clip(ratio, 0, 1)
        # gentle smoothing instead of forced monotonic growth ---
        if len(ratio) > 3:
            smooth = pd.Series(ratio).rolling(
                window=3,
                min_periods=1,
                center=False
            ).mean().values
            ratio = self._sanitize_array(smooth)
        # enforce monotonic increase since vaccination coverage
        # should not decrease in cumulative statistics
        # ratio = np.maximum.accumulate(ratio)
        return ratio

    # ---------------- LAG ----------------

    def _lag(self, x, lag):
        """
        Apply a lag to the vaccination signal.

        Vaccination effects are not immediate. This function shifts the
        vaccination signal forward so that earlier vaccination influences
        later epidemiological outcomes.

        In a weekly dataset this lag effectively shifts observations
        across weekly steps.
        """
        x = self._sanitize_array(x)

        if lag <= 0:
            return x

        out = np.zeros_like(x)

        if len(x) > lag:
            out[lag:] = x[:-lag]

        return out

    # ---------------- SEASON ----------------

    def _season(self, d):
        """
        Seasonal signal based on day-of-year.

        Respiratory diseases often exhibit strong seasonal patterns.
        A sine wave approximation is used as a simple seasonal proxy.
        """
        doy = d[self.date_column].dt.dayofyear.values
        s = np.sin(2 * np.pi * doy / 365.0)

        return self._sanitize_array(s)

    def _future_season(self, last_date, horizon):
        # generate seasonal values for future simulation horizon
        dates = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        doy = dates.dayofyear.values
        s = np.sin(2 * np.pi * doy / 365.0)

        return self._sanitize_array(s)

    # ---------------- VACC FUTURE ----------------

    def _build_vaccination_trajectory(self, last_vacc, multiplier):
        """
        Build a gradual vaccination trajectory instead of a flat value.

        This better reflects real vaccination dynamics where coverage
        changes gradually rather than jumping instantly.
        """

        target = np.clip(last_vacc * multiplier, 0, 1)

        traj = np.linspace(last_vacc, target, self.horizon)

        traj = np.clip(traj, 0, 1)

        return self._sanitize_array(traj)

    # ---------------- MODEL FIT ----------------

    def _fit(self, X, y):
        """
        Fit a Ridge regression model for a structural equation.

        Ridge is used instead of OLS to improve stability when
        features are correlated or data is limited.
        """
        X = self._sanitize_array(X)
        y = self._sanitize_array(y)

        if len(X) < 5:
            return None, float(np.std(y)) if len(y) > 0 else 0.0

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_std = np.std(X, axis=0) + 1e-6
        X_scaled = X / X_std

        model = Ridge(alpha=self.alpha)
        model.fit(X_scaled, y)

        model.scale_factor_ = X_std

        preds = model.predict(X_scaled)
        residuals = y - preds
        sigma = self._num(np.std(residuals), 0.0)

        return model, sigma

    def _predict(self, model, x, sigma=0.0):
        """
        Predict next state value with stochastic noise.

        Noise sampled from residual variance approximates uncertainty
        in the structural equation during simulation.
        """
        x = self._sanitize_array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if model is None:
            base = 0.0
        else:

            x_scaled = x / getattr(model, "scale_factor_", 1.0)
            base = self._num(model.predict(x_scaled)[0], 0.0)

        noise = np.random.normal(0, self._num(sigma))
        return max(0.0, base + noise)

    # ---------------- SCM ----------------

    def _fit_scm(self, d, vacc):
        """
        Fit structural equations for the causal graph.

        Graph structure:

        vaccination -> cases
        cases -> death

        The cases equation depends on:
            - previous cases
            - lagged vaccination signal
            - seasonal component

        The death equation depends on:
            - previous deaths
            - current projected cases
        Each node is modeled using an autoregressive structural equation.
        """

        season = self._season(d)
        vacc_lag = self._lag(vacc, self.vaccine_lag_weeks)

        cases = self._safe(d, self.metrics[0])
        death = self._safe(d, self.metrics[1])

        models = {}

        X = np.column_stack([
            cases[:-1],
            vacc_lag[:-1],
            season[:-1]
        ])

        y = cases[1:]

        models["cases"] = self._fit(X, y)

        X = np.column_stack([
            death[:-1],
            cases[:-1]
        ])

        y = death[1:]

        models["death"] = self._fit(X, y)

        return models

        # scenario interpretation

    def _interpret_direction(self, effect):
        if effect is None:
            return None
        if effect > 0:
            return "reduces"
        if effect < 0:
            return "increases"
        return "does not change"

    # ---------------- SIMULATION ----------------

    def _simulate(self, state, vacc_future, season_future, models):
        """
        Run forward simulation using the fitted SCM.

        At each simulation step:

            vaccination -> cases -> death

        Projected vaccination levels influence future cases,
        and projected cases influence future deaths.

        Stochastic residual noise is injected at each step
        to approximate uncertainty propagation.
        """
        cases = self._num(state.get("cases", 0.0))
        death = self._num(state.get("death", 0.0))

        traj = []

        for t in range(self.horizon):

            vacc = vacc_future[t]
            seas = season_future[t]

            m, s = models["cases"]
            x = np.array([[cases, vacc, seas]])
            cases = self._predict(m, x, s)

            m, s = models["death"]
            x = np.array([[death, cases]])
            death = self._predict(m, x, s)

            traj.append((cases, death))

        return np.array(traj)

    # ---------------- RUN ----------------

    def run(self, data: pd.DataFrame, context: dict):
        """
        Main entry point used by the governance orchestrator.

        For each location:
            1. Build vaccination signal
            2. Fit the SCM
            3. Simulate configured vaccination counterfactual scenarios
            4. Estimate uncertainty via Monte‑Carlo sampling
            5. Produce downstream policy recommendation signals
        """
        logging.info("Running Causal Simulation Agent.")
        df = data.copy()
        df[self.date_column] = pd.to_datetime(
            df[self.date_column], errors="coerce")
        df = df.dropna(subset=[self.date_column])

        out = {}

        for loc, d in df.groupby(self.location_column):

            d = d.sort_values(self.date_column).reset_index(drop=True)

            if len(d) < 20:

                out[loc] = self._record(
                    status="skipped",
                    reason="insufficient history",
                    causal_trace=None
                )

                continue

            if "vaccination_ratio" in d.columns:
                vacc = self._sanitize_array(d["vaccination_ratio"].values)
            else:
                vacc = self._build_vaccination_ratio(d)

            models = self._fit_scm(d, vacc)

            state = {
                "cases": self._safe_last(d, self.metrics[0]),
                "death": self._safe_last(d, self.metrics[1])
            }

            last_vacc = vacc[-1]

            season_future = self._future_season(
                d[self.date_column].iloc[-1],
                self.horizon
            )

            scenarios = {}

            for scn, mult in self.scenario_multipliers.items():

                vacc_future = self._build_vaccination_trajectory(
                    last_vacc, mult)

                samples = []

                for _ in range(self.bootstrap_samples):

                    sim = self._simulate(
                        state.copy(),
                        vacc_future,
                        season_future,
                        models
                    )

                    samples.append(sim)

                samples = np.array(samples)

                mean = samples.mean(axis=0)
                p10 = np.percentile(samples, 10, axis=0)
                p90 = np.percentile(samples, 90, axis=0)

                scenarios[scn] = {
                    "cases_final": float(mean[-1, 0]),
                    "death_final": float(mean[-1, 1]),
                    "cases_traj": mean[:, 0].tolist(),
                    "death_traj": mean[:, 1].tolist(),
                    "cases_interval": [float(p10[-1, 0]), float(p90[-1, 0])],
                    "death_interval": [float(p10[-1, 1]), float(p90[-1, 1])]
                }

            baseline_traj = np.array(scenarios["baseline"]["cases_traj"])
            plus_traj = np.array(
                scenarios["vaccination_plus_20"]["cases_traj"])
            minus_traj = np.array(
                scenarios["vaccination_minus_20"]["cases_traj"])

            baseline_death_traj = np.array(scenarios["baseline"]["death_traj"])
            plus_death_traj = np.array(
                scenarios["vaccination_plus_20"]["death_traj"])
            minus_death_traj = np.array(
                scenarios["vaccination_minus_20"]["death_traj"])

            baseline_cases = float(baseline_traj[-1])
            plus_cases = float(plus_traj[-1])
            minus_cases = float(minus_traj[-1])

            baseline_death = float(scenarios["baseline"]["death_final"])
            plus_death = float(scenarios["vaccination_plus_20"]["death_final"])
            minus_death = float(
                scenarios["vaccination_minus_20"]["death_final"])

            eps = 1e-6

            reduction_pct = 0.0
            increase_pct = 0.0

            if baseline_cases > eps:
                increase_pct = (minus_cases - baseline_cases) / baseline_cases
                reduction_pct = (baseline_cases - plus_cases) / baseline_cases

            death_reduction_pct = 0.0
            death_increase_pct = 0.0

            if baseline_death > eps:
                death_reduction_pct = (
                    baseline_death - plus_death) / baseline_death
                death_increase_pct = (
                    minus_death - baseline_death) / baseline_death

            fragility = 0.0

            log_cases_baseline = np.log1p(baseline_traj)
            log_cases_plus = np.log1p(plus_traj)
            log_cases_minus = np.log1p(minus_traj)

            log_death_baseline = np.log1p(baseline_death_traj)
            log_death_plus = np.log1p(plus_death_traj)
            log_death_minus = np.log1p(minus_death_traj)

            cases_up = np.mean(np.abs(log_cases_plus - log_cases_baseline))
            cases_down = np.mean(np.abs(log_cases_minus - log_cases_baseline))

            death_up = np.mean(np.abs(log_death_plus - log_death_baseline))
            death_down = np.mean(np.abs(log_death_minus - log_death_baseline))

            up_effect = 0.4 * cases_up + 0.6 * death_up
            down_effect = 0.4 * cases_down + 0.6 * death_down

            baseline_var_cases = np.std(log_cases_baseline)
            baseline_var_death = np.std(log_death_baseline)

            baseline_var = 0.4 * baseline_var_cases + 0.6 * baseline_var_death + 1e-6

            if self.fragility_mode == "downside":
                fragility = down_effect
            else:
                # normalize relative to baseline variability
                fragility = max(up_effect, down_effect) / baseline_var

            fragility = float(min(self.fragility_cap, fragility))

            recommended_action = (
                "adjust_vaccination_policy"
                if death_reduction_pct > 0.05
                else "maintain_policy"
            )

            confidence = float(min(1.0, abs(reduction_pct) * 4))
            risk_level = self._fragility_risk_level(fragility)

            policy_msg = (
                f"Vaccination policy impact simulation shows that increasing vaccination by 20% "
                f"**{self._interpret_direction(reduction_pct)}** projected cases by **{abs(reduction_pct)*100:.1f}**% "
                f"and **{self._interpret_direction(death_reduction_pct)}** deaths by **{abs(death_reduction_pct)*100:.1f}**%. "
                f"A 20% decrease in vaccination could **{self._interpret_direction(increase_pct)}** cases by **{abs(increase_pct)*100:.1f}**% "
                f"and **{self._interpret_direction(death_increase_pct)}** deaths by **{abs(death_increase_pct)*100:.1f}**%. "
                f"System fragility score is **{fragility:.2f}** ({risk_level} risk)."
            )

            causal_trace = {

                "model": {
                    "type": "SCM",
                    "graph": {
                        "vaccination": ["cases"],
                        "cases": ["death"],
                        "death": []
                    },
                    "simulation": {
                        "horizon": self.horizon,
                        "bootstrap_samples":
                        self.bootstrap_samples,
                        "vaccine_lag_weeks":
                        self.vaccine_lag_weeks,
                        "ridge_alpha":
                        self.alpha
                    }
                },
                "metrics": {
                    "cases_reduction_pct":
                    float(reduction_pct),
                    "death_reduction_pct":
                    float(death_reduction_pct),
                    "cases_increase_pct":
                    float(increase_pct),
                    "death_increase_pct":
                    float(death_increase_pct),
                    "fragility_score":
                    float(fragility),
                    "confidence":
                    float(confidence)
                },

                "policy": {
                    "recommended_action":
                    recommended_action,
                    "risk_level":
                    risk_level
                },
                "scenarios":
                scenarios,
                "explanation":
                policy_msg
            }

            out[loc] = self._record(
                status="ok",
                location_key=loc,
                causal_trace=causal_trace,
                recommended_action=recommended_action,
                confidence=confidence,
                risk_level=risk_level,
                explanation=policy_msg
            )
        logging.info("Generating Causal Simulation Agent's Json ouput.")
        return {
            "status": "ok",
            "agent": "CausalSimulationAgent",
            "agent_version": self.AGENT_VERSION,
            "scm_graph": {
                "vaccination": ["cases"],
                "cases": ["death"],
                "death": []
            },
            "vaccine_lag_weeks": self.vaccine_lag_weeks,
            "causal": out
        }
