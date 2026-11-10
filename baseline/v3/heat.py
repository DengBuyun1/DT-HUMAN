"""Conditional device-energy estimates with explicit training support."""
from datetime import timedelta

import numpy as np

import build
from v3.data_quality import local_time
from v3.replay import record
from v3.splits import prior_sessions

MODELS = ("E-B0","E-R0","E-R1","E-R2")


def fit_heat(training):
    if not training:
        return {}
    old = build.heat_fit(training)
    t = np.array([x["minutes"] for x in training])
    y = np.array([x["observed"] for x in training])
    k = max(0.,float(t@y/(t@t)))
    return {"E-B0":old["historical"],"E-R0":old["rate"],"E-R1":old["line"],"E-R2":[0.,k]}


def experiment(data,sessions,cutoff,rolling=False,active=False):
    protocol = "V3-ROLL" if rolling else "V3-F50"
    if active:
        protocol = "V3-SENS"
    key = "swim_heat_active" if active else "swim_heat_total"
    workouts = {local_time(r["G"]):r for r in data["单次运动"]}
    samples = {}
    for session in sessions:
        raw = workouts[session.start]
        minutes = float(raw["J" if active else "I"])
        observed = float(raw["O"])
        if np.isfinite(minutes) and minutes>0 and np.isfinite(observed):
            samples[session.key] = dict(minutes=minutes,observed=observed)
    test = [s for s in sessions if s.start.date()>=cutoff.date() and s.key in samples]
    boundaries = {s.start.replace(hour=0,minute=0,second=0,microsecond=0) for s in test} if rolling else {cutoff}
    rows,fits = [],[]
    for boundary in sorted(boundaries):
        train = [s for s in prior_sessions(sessions,boundary) if s.key in samples]
        training = [samples[s.key] for s in train]
        coefficients = fit_heat(training)
        support = [min(x["minutes"] for x in training),max(x["minutes"] for x in training)] if training else None
        meta = dict(protocol_id=protocol,segment_id=key,train_end=boundary,
                    model_version=f"{protocol}:{key}:{boundary.isoformat()}")
        fits.append(dict(meta,target_id=key,training_sessions=[s.key for s in train],
                         support_minutes=support,coefficients=coefficients))
        for s in test:
            if rolling and s.date!=boundary.date().isoformat():
                continue
            y = samples[s.key]["observed"]
            t = samples[s.key]["minutes"]
            in_support = support is not None and support[0]<=t<=support[1]
            end = s.start+timedelta(seconds=s.duration)
            for name in MODELS:
                value = build.kcal(coefficients[name],t) if name in coefficients else None
                rows.append(record(name,meta,key,"conditional_duration",s.start,end,y,value,
                                   issued_at=end if value is not None else None,input_end=end if value is not None else None,
                                   available_at=end,session=s.key,minutes=t,in_train_support=in_support,
                                   support_min=support[0] if support else None,support_max=support[1] if support else None,
                                   formal_output=in_support and name in ("E-R0","E-R2"),
                                   reference_output=value is not None and name=="E-R0",
                                   reason="outside_training_support" if not in_support else None,
                                   observed_id="workout:"+s.key,input_id="workout:"+s.key))
    return rows,fits
