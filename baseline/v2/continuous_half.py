"""Audit observed continuity and run a fixed chronological half split."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, time, timedelta
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import build
import swim
from evaluate import metrics
from heart import LocalTrend


DAY = timedelta(days=1)
HOUR = timedelta(hours=1)
SENSORS = (("实时心率", "心率", "H", "次/分"),
           ("血氧", "血氧饱和度", "G", "%"),
           ("压力", "压力", "G", "分"))


def continuous_runs(ticks, step):
    groups = []
    for tick in sorted(set(ticks)):
        if not groups or tick - groups[-1][-1] != step:
            groups.append([])
        groups[-1].append(tick)
    return sorted(groups, key=lambda group: (-len(group), group[0]))


def hourly(data):
    series = {}
    for name, sheet, tcol, unit in SENSORS:
        by_stamp = defaultdict(set)
        for r in data[sheet]:
            if sheet == "心率" and r["G"] != name:
                continue
            by_stamp[datetime.fromisoformat(r[tcol])].add(float(r["F"]))
        per_hour = defaultdict(list)
        for stamp, values in by_stamp.items():
            if len(values) == 1:
                per_hour[stamp.replace(minute=0, second=0, microsecond=0)].append(next(iter(values)))
        series[name] = {"unit": unit, "points": {t: (median(v), len(v)) for t, v in per_hour.items()},
                        "conflict_timestamps": sum(len(v) > 1 for v in by_stamp.values())}
    return series


def continuity(label, unit, granularity, points, step, total, min_readings=1):
    eligible = {t: (y, n) for t, (y, n) in points.items() if n >= min_readings}
    runs = continuous_runs(eligible, step)
    best = runs[0] if runs else []
    result = {"metric": label, "unit": unit, "resolution": granularity, "min_readings": min_readings,
              "observed_slots": len(eligible), "calendar_slots": total,
              "longest_slots": len(best), "longest_start": best[0].isoformat() if best else None,
              "longest_end": best[-1].isoformat() if best else None,
              "raw_readings_in_longest": sum(eligible[t][1] for t in best),
              "runs": [{"start": seq[0].isoformat(), "end": seq[-1].isoformat(), "n": len(seq)} for seq in runs]}
    return result, [(t, *eligible[t]) for t in best]


def fit_daily(train):
    """Fix all estimates from the training prefix."""
    y = np.asarray(train)
    model = None
    if len(y) >= 8 and np.ptp(y[:-1]) > 0:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(y[:-1, None], y[1:])
    return float(median(y)), model


def evaluate_series(item, sequence, step):
    mid = len(sequence) // 2
    if mid < 4:
        return [], {"status": "insufficient", "train_slots": mid, "test_slots": len(sequence)-mid}
    train, test = sequence[:mid], sequence[mid:]
    assert train[-1][0] + step == test[0][0]
    historical, model = fit_daily([y for _, y, _ in train])
    prior = recursive = train[-1][1]
    rows = []
    for phase, samples in (("train", train), ("test", test)):
        for t, actual, count in samples:
            record = {"metric": item["metric"], "resolution": item["resolution"], "slot": t.isoformat(),
                      "split": phase, "observed": actual, "readings": count, "historical": None, "last": None,
                      "ridge_one_step": None, "ridge_open_loop": None, "frozen_last": None, "input_slot": None}
            if phase == "test":
                record.update(historical=historical, last=prior, frozen_last=train[-1][1],
                              input_slot=(t-step).isoformat())
                if model is not None:
                    record["ridge_one_step"] = float(model.predict([[prior]])[0])
                    recursive = float(model.predict([[recursive]])[0])
                    record["ridge_open_loop"] = recursive
                prior = actual
            rows.append(record)
    hold = [dict(r, date=r["slot"], id=r["slot"]) for r in rows if r["split"] == "test"]
    results = {k: metrics(hold, k) for k in ("historical", "last", "ridge_one_step", "ridge_open_loop", "frozen_last")}
    if model is not None:
        assert sum(x["split"] == "test" for x in rows) == results["ridge_open_loop"]["n"]
    return rows, {"status": "ridge" if model is not None else "baseline_only", "train_slots": len(train),
                  "train_pairs": len(train)-1 if model is not None else 0, "test_slots": len(test),
                  "train_start": train[0][0].isoformat(), "train_end": train[-1][0].isoformat(),
                  "test_start": test[0][0].isoformat(), "test_end": test[-1][0].isoformat(),
                  "results": results}


def strict_cases(sessions, horizon, contiguous_only):
    for case in swim.cases(sessions, horizon, include_unobserved=True):
        s, i, j, _, _ = case
        intact = np.isfinite(s.heart[i-2:j+1]).all()
        if intact or not contiguous_only:
            yield case, bool(intact)


def swimming(sessions, boundary):
    train = [s for s in sessions if datetime.fromisoformat(s.date) < boundary]
    test = [s for s in sessions if datetime.fromisoformat(s.date) >= boundary]
    assert all(s.start + timedelta(seconds=s.duration) <= boundary for s in train)
    rows, report = [], {}
    for h in (30, 120):
        fit = [c for c, _ in strict_cases(train, h, True)]
        x, y, w = swim.matrix(fit), swim.values(fit), swim.weights(fit)
        model = make_pipeline(StandardScaler(), Ridge(alpha=15)).fit(x, y, ridge__sample_weight=w)
        weighted = make_pipeline(StandardScaler(), Ridge(alpha=15)).fit(
            x, y, standardscaler__sample_weight=w, ridge__sample_weight=w)
        state = {}
        for s in test:
            trend = LocalTrend(.1, .05)
            for i, y in enumerate(s.heart):
                if not np.isfinite(y):
                    trend = LocalTrend(.1, .05)
                    continue
                trend.update(float(y))
                state[s.key, i] = trend.forecast(h//swim.BIN)
        hrows = []
        for (s, i, j, features, y), intact in strict_cases(test, h, False):
            origin = s.start + timedelta(seconds=int(s.seconds[i]))
            target = s.start + timedelta(seconds=int(s.seconds[j]))
            assert boundary <= origin < target
            hrows.append({"id": f"{s.key}:{j}:{h}", "session": s.key, "date": s.date, "horizon": h,
                          "origin_min": s.seconds[i]/60, "target_min": s.seconds[j]/60,
                          "observed": float(y) if np.isfinite(y) else None, "continuous": intact,
                          "persistence": features[0], "ridge_v1": float(model.predict([features])[0]),
                          "ridge_weighted": float(weighted.predict([features])[0]), "kalman": state[s.key, i]})
        rows.extend(hrows)
        evaluated = [r for r in hrows if r["continuous"]]
        keys = ("persistence", "ridge_v1", "ridge_weighted", "kalman")
        report[h] = {"train_windows": len(fit), "strict_test_windows": len(evaluated),
                     "issued_in_record": len(hrows), "target_present": sum(r["observed"] is not None for r in hrows),
                     "eligible_in_record": sum(max(0, len(s.heart)-2-h//swim.BIN) for s in test),
                     "strict": {k: metrics(evaluated, k, 10) for k in keys},
                     "all_observed_sensitivity": {k: metrics(hrows, k, 10) for k in keys}}
    counts = []
    for s in sessions:
        runs = continuous_runs(s.seconds[np.isfinite(s.heart)], swim.BIN)
        counts.append({"session": s.key, "date": s.date, "split": "train" if s in train else "test",
                       "total_windows": len(s.heart), "observed_windows": sum(len(r) for r in runs),
                       "longest_continuous_windows": len(runs[0]) if runs else 0,
                       "longest_continuous_minutes": len(runs[0])*swim.BIN/60 if runs else 0})
    return rows, counts, {"train_swims": len(train), "test_swims": len(test),
                          "train_dates": len({s.date for s in train}), "test_dates": len({s.date for s in test}),
                          "results": report}


def heat(data, sessions, boundary):
    workouts = {datetime.fromisoformat(r["G"]): r for r in data["单次运动"]}
    rows = []
    report = {}
    for col, label in (("I", "总时长"), ("J", "活动时间")):
        samples = [{"id": s.key, "date": s.date, "minutes": float(workouts[s.start][col]),
                    "observed": float(workouts[s.start]["O"])} for s in sessions if float(workouts[s.start][col]) > 0]
        train = [r for r in samples if datetime.fromisoformat(r["date"]) < boundary]
        test = [r for r in samples if datetime.fromisoformat(r["date"]) >= boundary]
        parameters = build.heat_fit(train)
        lo, hi = min(r["minutes"] for r in train), max(r["minutes"] for r in train)
        predictions = [{**r, "duration_kind": label, "in_train_support": lo <= r["minutes"] <= hi,
                        **{k: build.kcal(coeff, r["minutes"]) for k, coeff in parameters.items()}} for r in test]
        rows.extend(predictions)
        report[label] = {"train_n": len(train), "test_n": len(test), "support_minutes": [lo, hi],
                         "all": {k: metrics(predictions, k) for k in parameters},
                         "within": {k: metrics([r for r in predictions if r["in_train_support"]], k) for k in parameters}}
    return rows, report


def main():
    payload = (ROOT / "data/健康数据.xlsx").read_bytes()
    data = swim.read_workbook(payload)
    obs = build.observations(data)
    daily = {k: {"unit": v["unit"], "points": {datetime.fromisoformat(d): (y,n) for d,y,n in v["points"]}}
             for k,v in obs["series"].items()}
    start = min(t for s in daily.values() for t in s["points"])
    end = max(t for s in daily.values() for t in s["points"])
    days = (end-start).days+1
    cutoff = start+DAY*(days//2)
    hdata = hourly(data)
    inventory, series_rows = [], []
    for resolution, signals, step, counts in (("day", daily, DAY, days), ("hour", hdata, HOUR, days*24)):
        for label, source in signals.items():
            item, seq = continuity(label, source["unit"], resolution, source["points"], step, counts,
                                   min_readings=1 if resolution == "day" else 3)
            rows, result = evaluate_series(item, seq, step)
            inventory.append({**item, "test": result})
            series_rows.extend(rows)
    overlap = set.intersection(*[{t for t, (_, n) in v["points"].items() if n >= 3} for v in hdata.values()])
    joint_runs = continuous_runs(overlap, HOUR)
    one_read = {label: {"hours": len(v["points"]), "longest": len(continuous_runs(v["points"], HOUR)[0])}
                for label, v in hdata.items()}
    with threadpool_limits(limits=1):
        sessions, _ = swim.build_swims(data)
        heart_rows, swim_counts, swim_report = swimming(sessions, cutoff)
        heat_rows, heat_report = heat(data, sessions, cutoff)
    out = ROOT / "runs" / ("连续半段-" + datetime.now(build.TZ).strftime("%Y%m%d-%H%M%S-%f"))
    out.mkdir()
    report = {"source_sha256": sha256(payload).hexdigest(), "source_rows": sum(map(len,data.values())),
              "start": start.date().isoformat(), "end": end.date().isoformat(), "calendar_days": days,
              "exercise_cutoff": cutoff.isoformat(), "daily_hourly": inventory,
              "hourly_at_least_one": one_read, "joint_hourly": {"observed_hours": len(overlap),
              "longest_slots": len(joint_runs[0]) if joint_runs else 0,
              "longest_start": joint_runs[0][0].isoformat() if joint_runs else None},
              "swim": swim_report, "heat": heat_report}
    build.dump(out / "metrics.json", report)
    for filename, rows in [("连续性概览.csv", [{k:v for k,v in r.items() if k not in ("runs","test")} for r in inventory]),
                           ("逐时间点预测.csv", series_rows), ("逐游泳场次连续性.csv", swim_counts),
                           ("游泳心率预测.csv", heart_rows), ("热量估算.csv", heat_rows)]:
        build.csv_rows(out / filename, rows)
    from continuous_half_figures import render
    render(out, report, daily, hdata, series_rows, swim_counts, heart_rows, heat_rows)
    print(out)


if __name__ == "__main__":
    main()
