"""Short-horizon forecasts issued without inspecting their future target."""
from datetime import timedelta

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import swim
from v3.daily import pipeline_params
from v3.replay import record
from v3.splits import prior_sessions

MODELS = ("HR-B0","HR-B1","HR-R0","HR-R1")
BIN = timedelta(seconds=swim.BIN)


def make_predictors(sessions,horizon,min_dates,alpha):
    cases = swim.cases(sessions,horizon)
    if not cases:
        return {"HR-B0":lambda x: float(x[0])}, dict(train_windows=0, train_dates=0, training_targets=[])
    x,y,w = swim.matrix(cases),swim.values(cases),swim.weights(cases)
    dates = {s.date for s,*_ in cases}
    historical = float(np.average(y,weights=w))
    models = {"HR-B0":lambda row:float(row[0]),"HR-B1":lambda row:historical}
    fit = dict(train_windows=len(cases),train_dates=len(dates),historical_mean=historical,
               training_targets=[f"{s.key}:{j}" for s,i,j,_,_ in cases])
    if len(dates)>=min_dates:
        for key,delta in (("HR-R0",False),("HR-R1",True)):
            target = y-x[:,0] if delta else y
            estimator = make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(x,target,ridge__sample_weight=w)
            def predict(row, fitted=estimator, offset=delta):
                return float(fitted.predict([row])[0]+(row[0] if offset else 0))
            models[key] = predict
            fit[key] = pipeline_params(estimator)
    return models,fit


def replay_swim(session,horizon,models,meta):
    ends = [session.start+timedelta(seconds=int(s)) for s in session.seconds]
    stop = session.start+timedelta(seconds=session.duration)
    n = horizon//swim.BIN
    issued = []
    # Every candidate origin is visited before labels or strict continuity are inspected.
    for i,origin in enumerate(ends):
        if i<2 or origin>=stop:
            continue
        past = session.heart[i-2:i+1]
        x = None
        if np.isfinite(past).all():
            earlier,previous,current = past
            x = [current,previous,earlier,np.mean(past),current-earlier,np.log1p(session.seconds[i]/60)]
        forecasts = {key:models[key](x) if key in models and x is not None else None for key in MODELS}
        issued.append((i,origin,x is not None,forecasts))
    output = []
    for i,origin,input_valid,forecasts in issued:
        target = origin+timedelta(seconds=horizon)
        j = i+n
        in_record = j<len(ends)
        y = float(session.heart[j]) if in_record and np.isfinite(session.heart[j]) else None
        strict = bool(in_record and np.isfinite(session.heart[i-2:j+1]).all())
        for key,pred in forecasts.items():
            reason = ("input_missing" if not input_valid else "insufficient_training"
                      if pred is None else "outside_record" if not in_record else "target_missing" if y is None else None)
            output.append(record(key,meta,"swim_hr_"+str(horizon),"online_one_step",target-BIN,target,y,pred,
                                 issued_at=origin if pred is not None else None,
                                 input_end=origin if pred is not None else None,
                                 status=reason or "scored",reason=reason,origin=origin,
                                 session=session.key,date=session.date,horizon=horizon,
                                 in_record=in_record,strict=strict,input_valid=input_valid,
                                 observed_id=f"{session.key}:{j}" if y is not None else None,
                                 input_id=";".join(f"{session.key}:{k}" for k in range(i-2,i+1)) if input_valid else None))
    return output


def experiment(sessions,cutoff,config,rolling=False):
    protocol = "V3-ROLL" if rolling else "V3-F50"
    test = [s for s in sessions if s.start.date()>=cutoff.date()]
    test_days = sorted({s.start.replace(hour=0,minute=0,second=0,microsecond=0) for s in test})
    batches = [(day,[s for s in test if s.date==day.date().isoformat()]) for day in test_days] if rolling else [(cutoff,test)]
    rows,parameters = [],[]
    for boundary,group in batches:
        train = prior_sessions(sessions,boundary)
        for h in config["heart_horizons"]:
            models,fit = make_predictors(train,h,config["heart_min_train_dates"],config["heart_alpha"])
            meta = dict(protocol_id=protocol,segment_id="swim",train_end=boundary,
                        model_version=f"{protocol}:swim:{h}:{boundary.isoformat()}")
            parameters.append(dict(meta,target_id=f"swim_hr_{h}",train_sessions=[s.key for s in train],**fit))
            for session in group:
                rows.extend(replay_swim(session,h,models,meta))
    return rows,parameters
