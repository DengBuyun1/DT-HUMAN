"""Build an offline demo and an immutable historical replay bundle."""
import csv
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
import sys

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import swim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from garmin_lab.registry import Registry

TZ = timezone(timedelta(hours=8))
DAY = timedelta(days=1)
HEAT_MODELS = ("historical", "rate", "line", "selected")
REST_MODELS = ("historical", "last", "minutes", "steps")


def midnight(day):
    return datetime.combine(day, time(), TZ).isoformat()


def stats(rows, model):
    by_day = defaultdict(lambda: defaultdict(list))
    errors = []
    for row in rows:
        if row["observed"] is not None:
            error = row[model] - row["observed"]
            errors.append(error)
            by_day[row["date"]][row.get("session", row["id"])].append(abs(error))
    if not errors:
        return None
    date_mae = mean(mean(map(mean, sessions.values())) for sessions in by_day.values())
    return {"n": len(errors), "days": len(by_day), "mae": mean(map(abs, errors)),
            "date_mae": date_mae, "bias": mean(errors)}


def acceptance(rows, candidate, baselines):
    """Require a material improvement and wins on a majority of test dates."""
    if not rows:
        return {"passed": False, "comparisons": {}}
    estimate = stats(rows, candidate)["date_mae"]
    comparisons = {}
    for baseline in baselines:
        reference = stats(rows, baseline)["date_mae"]
        differences = defaultdict(list)
        for row in rows:
            differences[row["date"]].append(abs(row[baseline] - row["observed"])
                                            - abs(row[candidate] - row["observed"]))
        wins = sum(mean(day) > 1e-9 for day in differences.values())
        gain = (reference - estimate) / reference if reference > 0 else 0
        comparisons[baseline] = {"gain": gain, "wins": wins, "days": len(differences),
                                 "passed": gain >= .05 and wins > len(differences) / 2}
    return {"passed": all(test["passed"] for test in comparisons.values()),
            "comparisons": comparisons}


def heat_fit(rows):
    duration = np.array([r["minutes"] for r in rows])
    energy = np.array([r["observed"] for r in rows])
    centered = duration - duration.mean()
    denominator = centered @ centered
    slope = max(0., float(centered @ (energy - energy.mean()) / denominator)) if denominator else 0.
    return {"historical": [float(np.median(energy)), 0.],
            "rate": [0., float(np.median(energy / duration))],
            "line": [float(energy.mean() - slope * duration.mean()), slope]}


def kcal(coefficients, duration):
    intercept, slope = coefficients
    return max(0., intercept + slope * duration)


def heat_choice(previous):
    if len({row["date"] for row in previous}) >= 3:
        if acceptance(previous, "line", ["rate"])["passed"]:
            return "line"
    return "rate"


def backtest_heat(data, swims):
    workouts = {datetime.fromisoformat(row["G"]): row for row in data["单次运动"]}
    samples = []
    for session in swims:
        row = workouts[session.start]
        duration, energy = float(row["I"]), float(row["O"])
        if duration > 0 and energy > 0:
            samples.append({"id": session.key, "date": session.date, "minutes": duration,
                            "observed": energy, "origin": session.start.replace(tzinfo=TZ).isoformat(),
                            "available": datetime.fromisoformat(row["H"]).replace(tzinfo=TZ).isoformat()})
    rows = []
    for day in sorted({row["date"] for row in samples})[4:]:
        train = [row for row in samples if row["date"] < day]
        params = heat_fit(train)
        selected = heat_choice(rows)
        lower, upper = min(r["minutes"] for r in train), max(r["minutes"] for r in train)
        for sample in (row for row in samples if row["date"] == day):
            predictions = {key: kcal(value, sample["minutes"]) for key, value in params.items()}
            rows.append({**sample, **predictions, "selected": predictions[selected],
                         "model": selected, "fit_through": train[-1]["date"],
                         "train_n": len(train), "support_min": lower, "support_max": upper,
                         "in_support": lower <= sample["minutes"] <= upper})
    supported = [row for row in rows if row["in_support"]]
    decision = acceptance(supported, "selected", ["historical"])
    selected = heat_choice(rows)
    lower, upper = min(r["minutes"] for r in samples), max(r["minutes"] for r in samples)
    return {"samples": samples, "rows": rows, "metrics": {m: stats(rows, m) for m in HEAT_MODELS},
            "subgroups": {key: {m: stats(subset, m) for m in HEAT_MODELS}
                          for key, subset in [("in", supported), ("out", [r for r in rows if not r["in_support"]])]},
            "decision": decision, "model": selected, "coefficients": heat_fit(samples)[selected],
            "support": [lower, upper], "fit_through": samples[-1]["date"]}


def backtest_rest(data):
    rest = {date.fromisoformat(r["H"][:10]): float(r["F"]) for r in data["心率"] if r["G"] == "静息心率"}
    activity = {date.fromisoformat(r["N"][:10]): r for r in data["活动统计"]}
    samples = []
    for target, observed in sorted(rest.items()):
        history = [value for day, value in sorted(rest.items()) if day < target][-7:]
        previous = activity.get(target - DAY)
        if previous and history:
            samples.append({"id": target.isoformat(), "date": target.isoformat(),
                            "origin": midnight(target), "available": midnight(target + DAY),
                            "observed": observed, "historical": median(history), "last": history[-1],
                            "input_minutes": float(previous["H"]), "input_steps": float(previous["F"])})
    rows = []
    for index, sample in enumerate(samples[14:], start=14):
        train = samples[:index]
        residuals = [r["observed"] - r["historical"] for r in train]
        predictions = {}
        for model in ("minutes", "steps"):
            feature = f"input_{model}"
            fit = make_pipeline(StandardScaler(), Ridge(alpha=10, fit_intercept=False))
            fit.fit([[r[feature]] for r in train], residuals)
            predictions[model] = sample["historical"] + float(fit.predict([[sample[feature]]])[0])
        rows.append({**sample, **predictions, "model": "minutes", "selected": predictions["minutes"],
                     "fit_through": train[-1]["date"], "train_n": len(train)})
    return {"rows": rows, "pairs": len(samples), "metrics": {m: stats(rows, m) for m in REST_MODELS},
            "decision": acceptance(rows, "minutes", ["historical", "last"])}


def sleep_exploration(data):
    resting = {date.fromisoformat(r["H"][:10]): float(r["F"]) for r in data["心率"] if r["G"] == "静息心率"}
    pressure = defaultdict(lambda: defaultdict(list))
    for r in data["压力"]:
        when = datetime.fromisoformat(r["G"])
        if 13 <= when.hour < 18:
            pressure[when.date()][when.hour].append(float(r["F"]))
    result = []
    for r in sorted(data["睡眠"], key=lambda row: row["G"]):
        if r["O"] != "科学睡眠":
            continue
        end = datetime.fromisoformat(r["G"])
        hours = [median(values) for values in pressure[end.date()].values() if len(values) >= 2]
        result.append({"date": end.date().isoformat(), "minutes": float(r["H"]),
                       "next_rest": resting.get(end.date() + DAY), "pressure_hours": len(hours),
                       "afternoon_pressure": mean(hours) if end.hour < 13 and len(hours) >= 3 else None})
    return result


def backtest_swim(swims):
    dates = sorted({session.date for session in swims})
    train = [session for session in swims if session.date in dates[:-3]]
    held_out = [session for session in swims if session.date in dates[-3:]]
    rows, reports = [], {}
    for horizon, selected in [(30, "persistence"), (120, "ridge")]:
        fitted = swim.fit_models(swim.cases(train, horizon))
        records = swim.cases(held_out, horizon, include_unobserved=True)
        predictions = {model: fitted[model](swim.matrix(records)) for model in ("ridge", "persistence")}
        horizon_rows = []
        for index, (session, anchor, target, _, observed) in enumerate(records):
            origin_s = int(session.seconds[anchor])
            target_s = int(session.seconds[target])
            start = session.start.replace(tzinfo=TZ)
            horizon_rows.append({"id": f"{session.key}:{target_s}:{horizon}", "session": session.key,
                                 "date": session.date, "horizon": horizon, "origin_s": origin_s,
                                 "target_s": target_s, "origin": (start + timedelta(seconds=origin_s)).isoformat(),
                                 "available": (start + timedelta(seconds=target_s)).isoformat(),
                                 "observed": float(observed) if np.isfinite(observed) else None,
                                 "ridge": float(predictions["ridge"][index]),
                                 "persistence": float(predictions["persistence"][index]),
                                 "selected": float(predictions[selected][index]), "model": selected,
                                 "fit_through": dates[-4]})
        reports[horizon] = {"model": selected, "metrics": {m: stats(horizon_rows, m) for m in predictions},
                            "issued": len(horizon_rows),
                            "eligible": sum(max(0, len(s.heart) - 2 - horizon // 30) for s in held_out)}
        rows.extend(horizon_rows)
    sessions = [{"id": s.key, "date": s.date, "held_out": s in held_out,
                 "windows": [[int(t), float(v) if np.isfinite(v) else None] for t, v in zip(s.seconds, s.heart)]}
                for s in swims]
    return {"rows": rows, "metrics": reports, "sessions": sessions}


def write_ledger(path, groups):
    events = []
    ids = {}
    for task, rows in groups.items():
        for row in rows:
            if row["fit_through"] >= row["date"] or row["origin"] >= row["available"]:
                raise ValueError(f"Invalid replay timing for {task} {row['id']}")
            events.append((row["origin"], 1, task, row["id"], row))
            if row["observed"] is not None:
                events.append((row["available"], 0, task, row["id"], row))
    now = min(event[0] for event in events)
    registry = Registry(path, clock=lambda: now)
    try:
        hypotheses = {task: registry.register(task, {"base": task, "args": {"mode": "offline"}})
                      for task in groups}
        scored = set()
        for now, kind, task, identifier, row in sorted(events, key=lambda event: event[:4]):
            key = (task, identifier)
            if kind == 1:
                if key in ids:
                    raise ValueError(f"Duplicate replay prediction: {key}")
                metadata = {"family": "point", "value": row["selected"], "model": row["model"],
                            "fit_through": row["fit_through"], "input_cutoff": row["origin"],
                            "run_id": path.parent.name, "subject_id": "local-01",
                            "target_start": row["origin"], "available_at": row["available"]}
                if task == "swim_heart":
                    target_end = datetime.fromisoformat(row["available"])
                    metadata["target_start"] = (target_end - timedelta(seconds=swim.BIN)).isoformat()
                if task == "swim_energy_conditional":
                    metadata["condition_minutes"] = row["minutes"]
                    metadata["input_mode"] = "retrospective_duration_condition"
                ids[key] = registry.predict(hypotheses[task], task, row["available"], metadata)
            else:
                if key in scored or now < row["available"]:
                    raise ValueError(f"Invalid replay score: {key}")
                registry.score_prediction(ids[key], row["observed"], outcome_ref=f"{task}:{identifier}")
                scored.add(key)
        ledger = registry.ledger()
        if len(ledger) != len(ids) or sum(r["scored_at"] is not None for r in ledger) != len(scored):
            raise RuntimeError("Incomplete prediction ledger")
        return {"issued": len(ids), "scored": len(scored)}
    finally:
        registry.con.close()


def observations(data):
    specs = [
        ("实时心率", "心率", "F", "H", "次/分"), ("静息心率", "心率", "F", "H", "次/分"),
        ("血氧", "血氧饱和度", "F", "G", "%"), ("压力", "压力", "F", "G", "分"),
        ("步数", "活动统计", "F", "N", "步"), ("日活动热量", "活动统计", "G", "N", "千卡"),
        ("运动分钟", "活动统计", "H", "N", "分钟"), ("活动小时", "活动统计", "I", "N", "小时"),
        ("主睡眠时长", "睡眠", "H", "G", "分钟"), ("小睡时长", "睡眠", "H", "G", "分钟"),
        ("体重", "体重", "F", "U", "kg"), ("BMI", "体重", "G", "U", "kg/m²"),
        ("体脂量", "体重", "H", "U", "kg"), ("体脂率", "体重", "I", "U", "%"),
        ("肌肉量", "体重", "J", "U", "kg"), ("基础代谢", "体重", "K", "U", "千卡/天"),
        ("水分量", "体重", "L", "U", "kg"), ("水分率", "体重", "M", "U", "%"),
        ("骨盐量", "体重", "O", "U", "kg"), ("蛋白质率", "体重", "P", "U", "%"),
        ("身体年龄", "体重", "Q", "U", "岁"), ("身体得分", "体重", "R", "U", "分"),
        ("骨骼肌量", "体重", "S", "U", "kg"),
    ]
    series = {}
    for label, sheet, value, when, unit in specs:
        daily = defaultdict(list)
        for row in data[sheet]:
            if sheet == "心率" and row["G"] != label:
                continue
            if sheet == "睡眠" and row["O"] != ("科学睡眠" if label == "主睡眠时长" else "零星小睡"):
                continue
            if row.get(value):
                daily[row[when][:10]].append(float(row[value]))
        series[label] = {"unit": unit, "sheet": sheet,
                         "points": [[day, median(values), len(values)] for day, values in sorted(daily.items())]}
    return {"series": series, "sheets": {k: len(v) for k, v in data.items()},
            "sleep": [{"start": r["F"], "end": r["G"], "minutes": float(r["H"]), "type": r["O"],
                       "stages": [[s["H"], s["F"]] for s in data["日常活动状态"]
                                  if r["F"] <= s["H"] < r["G"] and s["F"] in
                                  {"浅睡", "深睡", "快速眼动", "清醒", "零星小睡"}]}
                      for r in sorted(data["睡眠"], key=lambda r: r["F"])],
            "warnings": dict(Counter(r["F"] for r in data["预警数据"])),
            "workouts": [{"start": r["G"], "end": r["H"], "minutes": float(r["I"]),
                          "kcal": float(r["O"])} for r in sorted(data["单次运动"], key=lambda r: r["G"])]}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def csv_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    workbook = ROOT / "data" / "健康数据.xlsx"
    payload = workbook.read_bytes()
    data = swim.read_workbook(payload)
    run = ROOT / "runs" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    with threadpool_limits(limits=1):
        sessions, labels = swim.build_swims(data)
        heart = backtest_swim(sessions)
        heat = backtest_heat(data, sessions)
        rest = backtest_rest(data)
    groups = {"swim_heart": heart["rows"], "swim_energy_conditional": heat["rows"], "resting_hr": rest["rows"]}
    for task, rows in groups.items():
        csv_rows(run / f"{task}.csv", rows)
    ledger = write_ledger(run / "predictions.sqlite", groups)
    info = {"source_sha256": sha256(payload).hexdigest(), "run_id": run.name,
            "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "code_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__), ROOT / "swim.py", ROOT / "demo.html",
                                      ROOT / "vendor/garmin_lab/registry/store.py"]},
            "ledger": ledger, "activity_counts": labels}
    sleep = sleep_exploration(data)
    csv_rows(run / "sleep_exploration.csv", sleep)
    results = {"info": info, "heart": heart, "heat": heat, "rest": rest,
               "sleep_exploration": sleep, "history": observations(data)}
    dump(run / "metrics.json", {"info": info, "heart": heart["metrics"],
                                "heat": {k: v for k, v in heat.items() if k not in ("samples", "rows")},
                                "rest": {k: v for k, v in rest.items() if k != "rows"}})
    template = (ROOT / "demo.html").read_text(encoding="utf-8")
    inline = json.dumps(results, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    staged = run / "index.html"
    staged.write_text(template.replace("/*__DT_DATA__*/null", inline), encoding="utf-8")
    publish = ROOT / f".index-{run.name}.tmp"
    publish.write_bytes(staged.read_bytes())
    os.replace(publish, ROOT / "index.html")
    print((run / "metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
