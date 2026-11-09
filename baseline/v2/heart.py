"""Rolling calibration and a two-state forward Kalman filter."""
from dataclasses import dataclass
from statistics import mean

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import swim
from evaluate import by_date


def features(past, elapsed_s):
    earlier, previous, current = past
    return [current, previous, earlier, np.mean(past), current - earlier, np.log1p(elapsed_s / 60)]


def ridge_fit(sessions, horizon, alpha, weighted_scale):
    records = swim.cases(sessions, horizon)
    weights = swim.weights(records)
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    fit_args = {"ridge__sample_weight": weights}
    if weighted_scale:
        fit_args["standardscaler__sample_weight"] = weights
    return pipeline.fit(swim.matrix(records), swim.values(records), **fit_args)


class LocalTrend:
    """Linear local trend with time step one (30 seconds) and observation variance one."""

    transition = np.array([[1., 1.], [0., 1.]])

    def __init__(self, level_variance, slope_variance):
        self.q = np.diag([level_variance, slope_variance])
        self.x = None
        self.p = np.diag([1., 1.])

    def update(self, observation):
        if self.x is None:
            if observation is not None:
                self.x = np.array([observation, 0.])
            return
        self.x = self.transition @ self.x
        self.p = self.transition @ self.p @ self.transition.T + self.q
        if observation is not None:
            gain = self.p[:, 0] / (self.p[0, 0] + 1.)
            self.x += gain * (observation - self.x[0])
            residual = np.eye(2) - np.outer(gain, [1., 0.])
            self.p = residual @ self.p @ residual.T + np.outer(gain, gain)

    def forecast(self, steps):
        return float(self.x[0] + steps * self.x[1])


def fold_score(sessions, predictions, horizon):
    rows = [{"id": s.key, "date": s.date, "observed": float(y), "prediction": float(p)}
            for (s, _, _, _, y), p in zip(swim.cases(sessions, horizon), predictions)]
    return by_date(rows, "prediction")


def choose_settings(sessions, horizon, config):
    dates = sorted({s.date for s in sessions})
    ridge_losses = {a: [] for a in config["ridge_alphas"]}
    state_losses = {q: [] for q in config["kalman_slope_variances"]}
    for day in dates[config["heart_inner_warmup_dates"]:]:
        train = [s for s in sessions if s.date < day]
        test = [s for s in sessions if s.date == day]
        records = swim.cases(test, horizon)
        if not records:
            continue
        for alpha, losses in ridge_losses.items():
            predictions = ridge_fit(train, horizon, alpha, True).predict(swim.matrix(records))
            losses.append(fold_score(test, predictions, horizon)[day])
        for q, losses in state_losses.items():
            forecast = {}
            for session in test:
                model = LocalTrend(config["kalman_level_variance"], q)
                for i, observed in enumerate(session.heart):
                    model.update(float(observed) if np.isfinite(observed) else None)
                    if model.x is not None:
                        forecast[session.key, i] = model.forecast(horizon // swim.BIN)
            predictions = [forecast[s.key, i] for s, i, *_ in records]
            losses.append(fold_score(test, predictions, horizon)[day])
    if not any(ridge_losses.values()):
        return 15, .05
    return min(ridge_losses, key=lambda a: mean(ridge_losses[a])), min(state_losses, key=lambda q: mean(state_losses[q]))


@dataclass
class Calibrated:
    original_ridge: object
    weighted_ridge: object
    trend: LocalTrend
    alpha: float
    slope_variance: float

    def predict(self, past, elapsed_s, horizon):
        row = [features(past, elapsed_s)]
        return {"persistence": float(past[-1]), "ridge_v1": float(self.original_ridge.predict(row)[0]),
                "ridge": float(self.weighted_ridge.predict(row)[0]),
                "kalman": self.trend.forecast(horizon // swim.BIN)}


def calibrate(train, horizon, config):
    alpha, slope = choose_settings(train, horizon, config)
    return Calibrated(ridge_fit(train, horizon, 15, False), ridge_fit(train, horizon, alpha, True),
                      LocalTrend(config["kalman_level_variance"], slope), alpha, slope)
