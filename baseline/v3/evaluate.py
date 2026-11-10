"""Score matched predictions and apply predeclared engineering gates."""
from collections import defaultdict
from statistics import mean

import numpy as np

from v2.evaluate import metrics as v2_metrics, by_date


def valid(rows):
    return [r for r in rows if r["observed"] is not None and r["prediction"] is not None]


def scoring_rows(rows):
    return [dict(r, id=r["prediction_id"], session=r.get("session") or r["prediction_id"]) for r in valid(rows)]


def stats(rows, tolerance=None):
    rows = scoring_rows(rows)
    if not rows:
        return None
    result = v2_metrics(rows, "prediction", tolerance)
    result["n_dates"] = result.pop("days")
    y = np.array([r["observed"] for r in rows])
    pred = np.array([r["prediction"] for r in rows])
    residual = y-pred
    variance = float(np.var(y))
    result.update(rmse=float(np.sqrt(np.mean(residual**2))), observed_sd=float(np.std(y)),
                  prediction_sd=float(np.std(pred)), variation_ratio=float(np.std(pred)/np.std(y)) if variance else None,
                  r2=1-float(np.mean(residual**2))/variance if variance else None)
    return result


def compare(rows, candidate, reference, config, min_abs=0, support=False, heat=False):
    models = defaultdict(dict)
    for row in rows:
        if not support or row.get("in_train_support"):
            models[row["model_id"]][row["target_key"]] = row
    a, b = models[candidate], models[reference]
    coverage_ok = all(a.get(k, {}).get("prediction") is not None
                      for k, r in b.items() if r["prediction"] is not None)
    shared = sorted(k for k in a.keys() & b.keys()
                    if a[k]["prediction"] is not None and b[k]["prediction"] is not None
                    and a[k]["observed"] is not None and b[k]["observed"] is not None)
    result = dict(candidate=candidate, reference=reference, coverage_ok=coverage_ok, n_shared=len(shared),
                  n_dates=0, passed=False, status="insufficient_evidence")
    if not shared:
        return result
    cand = [a[k] for k in shared]
    ref = [b[k] for k in shared]
    ac, bc = stats(cand), stats(ref)
    da, db = by_date(scoring_rows(cand),"prediction"), by_date(scoring_rows(ref),"prediction")
    diffs = {k:db[k]-da[k] for k in sorted(db)}
    wins = sum(v>config["numeric_tolerance"] for v in diffs.values())
    drop = {k:mean(v for d,v in diffs.items() if d!=k) for k in diffs} if len(diffs)>1 else {}
    delta = bc["date_mae"]-ac["date_mae"]
    gain = delta/bc["date_mae"] if bc["date_mae"]>0 else None
    enough = len(diffs)>=config["min_adoption_dates"] and len(shared)>=config["min_test_points"]
    tol = config["numeric_tolerance"]
    tail_ok = ac["p90_abs"]<=bc["p90_abs"]+tol
    stable = bool(drop) and min(drop.values())>=-tol
    common = (enough and coverage_ok and gain is not None and gain>=config["relative_gain"]
              and delta>=min_abs and tail_ok)
    passed = common and (ac["mae"]<=bc["mae"]+tol if heat else wins>len(diffs)/2 and stable)
    result.update(n_dates=len(diffs), candidate_metrics=ac, reference_metrics=bc,
                  delta_mae=delta, relative_gain=gain, dates_won=wins, per_date_delta=diffs,
                  leave_one_date_out=drop, tail_ok=tail_ok, stable=stable, passed=passed,
                  status="research_candidate" if passed else "baseline_reference" if enough else "insufficient_evidence")
    return result


def decision(rows, config, task, open_loop=False):
    if task == "heart":
        primary = rows[0]["horizon"] == 120
        ref = "HR-B0"
        g0 = compare(rows, "HR-R0", ref, config, config["heart_min_gain_bpm"])
        g1 = [compare(rows, "HR-R1", k, config, config["heart_min_gain_bpm"])
              for k in ([ref, "HR-R0"] if primary else [ref])]
        selected = "HR-R1" if all(x["passed"] for x in g1) else "HR-R0" if g0["passed"] else ref
        gates = [g0, *g1]
    elif task == "heat":
        ref = "E-R0"
        gate = compare(rows,"E-R2",ref,config,support=True,heat=True)
        gates = [gate]
        selected = "E-R2" if gate["passed"] else ref
    else:
        refs = ["D-B0","D-B1","D-B3"] + ([] if open_loop else ["D-B2"])
        gates = [compare(rows,"D-R0",r,config) for r in refs]
        model_stats = stats([r for r in rows if r["model_id"]=="D-R0"])
        dynamics = model_stats is not None and model_stats["r2"] is not None and model_stats["r2"]>0
        selected = "D-R0" if all(x["passed"] for x in gates) and (not open_loop or dynamics) else "D-B0"
        ref = "D-B0"
    if selected != ref:
        state = "research_candidate"
    elif all(x["status"]=="insufficient_evidence" for x in gates):
        state = "insufficient_evidence"
    else:
        state = "baseline_reference"
    return dict(status=state, selected=selected, comparisons=gates,
                dynamic_diagnostic=(dynamics if task=="daily" else None))
