"""Issue predictions on observation events and score only on target arrival."""
from collections import defaultdict
from datetime import timedelta

import numpy as np

from build import Registry, TZ
import swim
from heart import LocalTrend, calibrate
from evaluate import comparison, metrics


MODELS = ("persistence", "ridge_v1", "ridge", "kalman")
REFERENCE = {30: "persistence", 120: "ridge_v1"}


def heart_replay(sessions, path, config):
    dates = sorted({s.date for s in sessions})
    horizons = config["heart_horizons_seconds"]
    evaluation_dates = dates[config["heart_warmup_dates"]:]
    now = min(s.start for s in sessions if s.date in evaluation_dates).replace(tzinfo=TZ)
    registry = Registry(path, clock=lambda: now.isoformat())
    rows, folds = [], []
    errors = defaultdict(lambda: defaultdict(list))
    try:
        hypotheses = {h: registry.register(f"swim_hr_{h}s", {"base": "swim_heart", "args": {"horizon": h}})
                      for h in horizons}
        for day in evaluation_dates:
            train = [s for s in sessions if s.date < day]
            fit_through = max(s.date for s in train)
            models = {h: calibrate(train, h, config) for h in horizons}
            folds.append({"date": day, "fit_through": fit_through, "train_swims": len(train),
                          "parameters": {h: {"ridge_alpha": m.alpha, "kalman_slope_variance": m.slope_variance}
                                         for h, m in models.items()}})
            intervals = {}
            for h in horizons:
                past_days = list(errors[h].values())
                if len(past_days) >= config["min_reference_error_dates"]:
                    intervals[h] = (len(past_days), *map(float, np.quantile(sum(past_days, []), [.1, .9])))
            for session in sorted((s for s in sessions if s.date == day), key=lambda s: s.start):
                start = session.start.replace(tzinfo=TZ)
                pending = defaultdict(list)
                history = []
                for model in models.values():
                    model.trend = LocalTrend(config["kalman_level_variance"], model.slope_variance)
                for i, value in enumerate(session.heart):
                    elapsed = int(session.seconds[i])
                    stamp = start + timedelta(seconds=elapsed)
                    if stamp < now:
                        raise ValueError("Replay sessions must have nonoverlapping observation times")
                    now = stamp
                    observation = float(value) if np.isfinite(value) else None
                    for row, prediction_id in pending.pop(i, []):
                        row["observed"] = observation
                        if prediction_id is not None and observation is not None:
                            registry.score_prediction(prediction_id, observation, outcome_ref=row["id"])
                            row["status"] = "scored"
                            errors[row["horizon"]][day].append(observation - row["selected"])
                        elif prediction_id is not None:
                            registry.withdraw_prediction(prediction_id, "missing_target")
                            row["status"] = "missing_target"
                    history.append(observation)
                    for model in models.values():
                        model.trend.update(observation)
                    if i < 2:
                        continue
                    for h in horizons:
                        j = i + h // swim.BIN
                        target = now + timedelta(seconds=h)
                        key = f"{session.key}:{elapsed + h}:{h}"
                        baseline = REFERENCE[h]
                        row = {"id": key, "session": session.key, "date": day, "horizon": h,
                               "origin_s": elapsed, "target_s": elapsed + h,
                               "origin": now.isoformat(), "available": target.isoformat(),
                               "fit_through": fit_through, "observed": None, "status": "missing_input",
                               "model": baseline, "selected": None, "in_record": j < len(session.heart),
                               **dict.fromkeys(MODELS), "ref_low": None, "ref_high": None, "ref_days": 0}
                        prediction_id = None
                        if None not in history[-3:]:
                            row.update(models[h].predict(history[-3:], elapsed, h))
                            row.update(selected=row[baseline], status="pending")
                            if h in intervals:
                                days, low, high = intervals[h]
                                row.update(ref_low=row["selected"] + low, ref_high=row["selected"] + high, ref_days=days)
                            prediction_id = registry.predict(hypotheses[h], "swim_heart", target, {
                                "family": "point", "value": row["selected"], "model": baseline,
                                "run_id": path.parent.name, "subject_id": "local-01", "input_cutoff": row["origin"],
                                "fit_through": fit_through, "target_start": (target - timedelta(seconds=swim.BIN)).isoformat(),
                                "available_at": row["available"], "window_id": key})
                        pending[j].append((row, prediction_id))
                        rows.append(row)
                for waiting in pending.values():
                    for row, prediction_id in waiting:
                        if prediction_id is not None:
                            registry.withdraw_prediction(prediction_id, "record_ended_before_target")
                            row["status"] = "outside_recording"
        ledger = registry.ledger()
        issued = [r for r in rows if r["selected"] is not None]
        if len(ledger) != len(issued) or sum(r["scored_at"] is not None for r in ledger) != sum(
                r["status"] == "scored" for r in issued):
            raise RuntimeError("Replay and ledger disagree")
    finally:
        registry.con.close()
    reports = {}
    for h in horizons:
        group = [r for r in rows if r["horizon"] == h and r["in_record"]]
        baseline = REFERENCE[h]
        band = [r for r in group if r["ref_low"] is not None and r["observed"] is not None]
        reports[h] = {
            "model": baseline, "eligible": len(group), "issued": sum(r["selected"] is not None for r in group),
            "missing_target": sum(r["status"] == "missing_target" for r in group),
            "outside_recording_issued": sum(r["horizon"] == h and r["status"] == "outside_recording" for r in rows),
            "metrics": {m: metrics(group, m, tolerance=10) for m in MODELS},
            "comparisons": {m: comparison(group, m, baseline, config["min_relative_gain"])
                            for m in MODELS if m != baseline},
            "empirical_band": {"n": len(band), "days": len({r["date"] for r in band}),
                               "hits": sum(r["ref_low"] <= r["observed"] <= r["ref_high"] for r in band),
                               "mean_width": float(np.mean([r["ref_high"] - r["ref_low"] for r in band])) if band else None}}
        report = reports[h]
        qualified = [m for m, result in report["comparisons"].items() if result["passed"]]
        report["recommended_next_model"] = min([baseline, *qualified], key=lambda m: report["metrics"][m]["date_mae"])
    series = [{"id": s.key, "date": s.date, "held_out": s.date in evaluation_dates,
               "windows": [[int(t), float(v) if np.isfinite(v) else None] for t, v in zip(s.seconds, s.heart)]}
              for s in sessions]
    return {"rows": rows, "metrics": reports, "sessions": series, "folds": folds,
            "ledger": {"issued": len(issued), "scored": sum(r["status"] == "scored" for r in issued),
                       "withdrawn": sum(r["status"] in ("missing_target", "outside_recording") for r in issued)}}
