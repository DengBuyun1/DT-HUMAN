"""Prediction evidence shared by calendar, swim and conditional estimates."""
def record(model_id, meta, target_id, mode, target_start, target_end, observed, prediction,
           issued_at=None, input_end=None, status=None, **extra):
    if issued_at is not None:
        assert meta["train_end"] <= issued_at
        assert input_end is None or input_end <= issued_at
    session = extra.get("session","")
    target_key = f"{meta['segment_id']}:{target_id}:{mode}:{session}:{target_end.isoformat()}"
    state = status or ("input_unavailable" if prediction is None else
                       "target_missing" if observed is None else "scored")
    if state!="scored" and extra.get("reason") is None:
        extra["reason"] = state
    return dict(meta, target_id=target_id, information_mode=mode, model_id=model_id,
                target_key=target_key, prediction_id=f"{meta['protocol_id']}:{model_id}:{target_key}",
                date=extra.pop("date",target_start.date().isoformat()), target_start=target_start,
                target_end=target_end, available_at=extra.pop("available_at",target_end),
                input_end=input_end, issued_at=issued_at, observed=observed,
                prediction=prediction, fallback_used=False,
                status=state,
                **extra)
