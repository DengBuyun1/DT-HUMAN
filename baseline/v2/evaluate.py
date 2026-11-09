"""Common date-level scoring for the V2 experiments."""
from collections import defaultdict
from statistics import mean

import numpy as np


def scored(rows, *keys):
    return [row for row in rows if row["observed"] is not None
            and all(row.get(key) is not None for key in keys)]


def by_date(rows, key):
    sessions = defaultdict(lambda: defaultdict(list))
    for row in scored(rows, key):
        sessions[row["date"]][row.get("session", row["id"])].append(abs(row[key] - row["observed"]))
    return {day: mean(map(mean, swims.values())) for day, swims in sessions.items()}


def metrics(rows, key, tolerance=None):
    valid = scored(rows, key)
    if not valid:
        return None
    errors = np.array([row[key] - row["observed"] for row in valid])
    absolute = np.abs(errors)
    dates = by_date(valid, key)
    result = {"n": len(valid), "days": len(dates), "mae": float(absolute.mean()),
              "date_mae": mean(dates.values()), "bias": float(errors.mean()),
              "p90_abs": float(np.quantile(absolute, .9)), "max_abs": float(absolute.max())}
    if tolerance is not None:
        result["within_tolerance"] = int((absolute <= tolerance).sum())
    return result


def comparison(rows, candidate, reference, min_gain):
    shared = scored(rows, candidate, reference)
    reference_n = sum(row.get(reference) is not None for row in rows)
    candidate_n = sum(row.get(candidate) is not None for row in rows)
    coverage_ok = all(row.get(candidate) is not None for row in rows if row.get(reference) is not None)
    result = {"candidate": candidate, "reference": reference, "issued_candidate": candidate_n,
              "issued_reference": reference_n, "coverage_ok": coverage_ok, "passed": False,
              "candidate_metrics": metrics(shared, candidate), "reference_metrics": metrics(shared, reference)}
    if not shared:
        return result
    cand, ref = result["candidate_metrics"], result["reference_metrics"]
    a, b = by_date(shared, candidate), by_date(shared, reference)
    gain = (ref["date_mae"] - cand["date_mae"]) / ref["date_mae"] if ref["date_mae"] else 0.0
    wins = sum(a[day] + 1e-9 < b[day] for day in b)
    tail_ok = cand["p90_abs"] <= ref["p90_abs"] + 1e-9
    result.update(gain=gain, wins=wins, days=len(b), tail_ok=tail_ok,
                  passed=gain >= min_gain and wins > len(b) / 2 and tail_ok and coverage_ok)
    return result
