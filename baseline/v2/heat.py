"""Compare heat estimators on the same dates and separate duration definitions."""
from datetime import datetime

from build import TZ, heat_choice, heat_fit, kcal
from evaluate import comparison, metrics


def rolling(data, sessions, config, column):
    workouts = {datetime.fromisoformat(row["G"]): row for row in data["单次运动"]}
    samples = []
    for s in sessions:
        source = workouts[s.start]
        raw_time = source.get(column)
        if raw_time is None or float(raw_time) <= 0:
            continue
        samples.append({"id": s.key, "date": s.date, "minutes": float(raw_time), "observed": float(source["O"]),
                        "origin": s.start.replace(tzinfo=TZ).isoformat(),
                        "available": datetime.fromisoformat(source["H"]).replace(tzinfo=TZ).isoformat()})
    rows = []
    for day in sorted({r["date"] for r in samples})[config["heat_warmup_dates"]:]:
        train = [r for r in samples if r["date"] < day]
        parameters = heat_fit(train)
        legacy = heat_choice(rows)
        lower, upper = min(r["minutes"] for r in train), max(r["minutes"] for r in train)
        for sample in (r for r in samples if r["date"] == day):
            values = {m: kcal(p, sample["minutes"]) for m, p in parameters.items()}
            rows.append({**sample, **values, "legacy_selected": values[legacy], "legacy_model": legacy,
                         "fit_through": train[-1]["date"], "train_n": len(train), "support_min": lower,
                         "support_max": upper, "in_support": lower <= sample["minutes"] <= upper})
    supported = [r for r in rows if r["in_support"]]
    models = ("historical", "rate", "line", "legacy_selected")
    subsets = {"all": rows, "in": supported, "out": [r for r in rows if not r["in_support"]]}
    comparisons = {s: {m: comparison(group, m, "rate", config["min_relative_gain"])
                       for m in ("line", "legacy_selected")} for s, group in subsets.items()}
    use_line = comparisons["in"]["line"]["passed"] and comparisons["all"]["line"].get("tail_ok", False)
    selected = "line" if use_line else "rate"
    for row in rows:
        row.update(selected=row["rate"], model="rate")
    support = [min(r["minutes"] for r in samples), max(r["minutes"] for r in samples)]
    return {"rows": rows, "samples": samples, "model": selected, "fit_through": samples[-1]["date"],
            "support": support, "coefficients": heat_fit(samples)[selected],
            "metrics": {m: metrics(rows, m) for m in models},
            "subgroups": {s: {m: metrics(group, m) for m in models} for s, group in subsets.items()},
            "comparisons": comparisons, "decision": {"passed": use_line},
            "duration_column": column, "sample_count": len(samples)}
