"""Frozen and forward-updated calendar baselines and one-lag Ridge."""
from statistics import mean, median

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v3.replay import record

MODELS = ("D-B0","D-B1","D-B2","D-B3","D-R0")


def pipeline_params(model):
    if model is None:
        return None
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coefficient=ridge.coef_.tolist(), intercept=float(ridge.intercept_), alpha=ridge.alpha)


def train_model(points, slots, step, config):
    data = {t:points[t].value for t in slots if t in points}
    adjacent = [(data[t-step], y) for t,y in data.items() if t-step in data]
    model = None
    if (len(data)>=config["daily_min_train_points"]
            and len(adjacent)>=config["daily_min_train_pairs"]):
        x,y = np.asarray(adjacent).T
        if np.ptp(x)>0:
            model = make_pipeline(StandardScaler(),Ridge(alpha=config["daily_alpha"])).fit(x[:,None],y)
    last = max(data) if data else None
    return data, adjacent, last, model


def evaluate_series(target_id, window, points, step, config, rolling=False):
    if not window.test:
        return [], []
    records, fits = [], []
    batches = [[t] for t in window.test] if rolling else [window.test]
    for ticks in batches:
        cutoff = ticks[0]
        train = [t for t in window.train + (window.test if rolling else []) if t<cutoff]
        data,pairs,last,model = train_model(points,train,step,config)
        history = list(data.values())
        med = float(median(history)) if history else None
        avg = float(mean(history)) if history else None
        frozen = data.get(last)
        model_version = f"{window.protocol_id}:{window.segment_id}:{cutoff.isoformat()}"
        meta = dict(protocol_id=window.protocol_id,segment_id=window.segment_id,train_end=cutoff,
                    model_version=model_version)
        fits.append(dict(meta, target_id=target_id, train_points=len(data),train_pairs=len(pairs),
                         training_slots=list(data), last_slot=last, median=med, mean=avg,
                         frozen_last=frozen,ridge=pipeline_params(model)))
        open_prediction = frozen
        open_at = last
        for tick in ticks:
            actual = points[tick].value if tick in points else None
            previous = tick-step
            one_step = points[previous].value if previous in points else None
            if model is not None and not rolling:
                while open_at<tick:
                    open_prediction = float(model.predict([[open_prediction]])[0])
                    open_at += step
            common = {"D-B0":med,"D-B1":avg,"D-B3":frozen}
            modes = {"online_one_step":dict(common, **{"D-B2":one_step,"D-R0":
                       float(model.predict([[one_step]])[0]) if model is not None and one_step is not None else None})}
            if not rolling:
                modes["open_loop"] = dict(common, **{"D-R0":open_prediction if model is not None else None})
            for mode,predictions in modes.items():
                for key,value in predictions.items():
                    uses_adjacent = mode=="online_one_step" and key in ("D-B2","D-R0")
                    last_input = previous if uses_adjacent and one_step is not None else last
                    info = last_input+step if last_input is not None else None
                    reason = ("insufficient_history" if not history else "model_unavailable"
                              if key=="D-R0" and model is None else "missing_adjacent_input") if value is None else None
                    records.append(record(key,meta,target_id,mode,tick,tick+step,actual,value,
                                          issued_at=tick if value is not None else None,
                                          input_end=info if value is not None else None,reason=reason,
                                          observed_id=f"{target_id}:{tick.isoformat()}" if actual is not None else None,
                                          input_id=f"{target_id}:{last_input.isoformat()}" if info is not None else None))
    return records,fits
