"""Offline inference with the original, frozen Chronos-2 weights."""
import os
from pathlib import Path
import sys

import numpy as np

from evaluate import comparison, metrics, scored
from heart import ridge_fit
import swim


def run(sessions, config, models_dir, device):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(models_dir / "chronos-forecasting/src"))
    import torch
    import transformers
    from chronos import Chronos2Pipeline

    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    pipeline = Chronos2Pipeline.from_pretrained(
        str(models_dir / "chronos-2-weights"), device_map=device, torch_dtype=torch.float32,
        local_files_only=True)
    horizon = 120
    contexts = config["chronos_context_windows"]
    dates = sorted({s.date for s in sessions})
    rows = []
    for date_index, day in enumerate(dates[config["heart_inner_warmup_dates"]:],
                                     start=config["heart_inner_warmup_dates"]):
        train = [s for s in sessions if s.date < day]
        records = swim.cases([s for s in sessions if s.date == day], horizon, include_unobserved=True)
        baseline = ridge_fit(train, horizon, 15, False).predict(swim.matrix(records))
        predictions = {c: {} for c in contexts}
        for c in contexts:
            indices, histories = [], []
            for index, (s, i, _, _, _) in enumerate(records):
                if i >= c - 1 and np.isfinite(s.heart[i + 1 - c:i + 1]).all():
                    indices.append(index)
                    histories.append(torch.tensor(s.heart[i + 1 - c:i + 1], dtype=torch.float32))
            if histories:
                with torch.inference_mode():
                    quantiles, _ = pipeline.predict_quantiles(
                        histories, prediction_length=4, quantile_levels=[.5], batch_size=24)
                if len(quantiles) != len(indices):
                    raise ValueError("Incomplete Chronos inference")
                predictions[c] = {i: float(q.reshape(4)[3].item()) for i, q in zip(indices, quantiles)}
                if not all(np.isfinite(v) for v in predictions[c].values()):
                    raise ValueError("Chronos returned a nonfinite prediction")
        keys = [f"context_{c}" for c in contexts]
        comparable = scored(rows, *keys)
        selected = min(keys, key=lambda k: metrics(comparable, k)["date_mae"]) if comparable else keys[0]
        for index, ((s, i, j, _, actual), ridge) in enumerate(zip(records, baseline)):
            candidates = {f"context_{c}": predictions[c].get(index) for c in contexts}
            rows.append({"id": f"{s.key}:{int(s.seconds[j])}:120", "date": day, "session": s.key,
                         "target_s": int(s.seconds[j]), "input_s": int(s.seconds[i]),
                         "fit_through": max(t.date for t in train), "ridge_v1": float(ridge),
                         "phase": "development" if date_index < config["heart_warmup_dates"] else "evaluation",
                         "observed": float(actual) if np.isfinite(actual) else None, **candidates,
                         "selected_context": selected, "selected": candidates[selected]})
    dev = [r for r in rows if r["phase"] == "development"]
    test = [r for r in rows if r["phase"] == "evaluation"]
    gate = comparison(dev, "selected", "ridge_v1", config["min_relative_gain"])
    # The planned trigger is development gain, majority of dates and comparable coverage.
    lora_trigger = (gate.get("gain", 0) >= config["min_relative_gain"]
                    and gate.get("wins", 0) > gate.get("days", 0) / 2 and gate["coverage_ok"])
    return rows, {
        "device": device, "torch": torch.__version__, "transformers": transformers.__version__,
        "weights": "amazon/chronos-2", "horizon": horizon, "issued_reference": len(test),
        "evaluation": {k: comparison(test, k, "ridge_v1", config["min_relative_gain"])
                       for k in [*(f"context_{c}" for c in contexts), "selected"]},
        "common_context_metrics": {k: metrics(scored(test, *(f"context_{c}" for c in contexts)), k)
                                   for k in [*(f"context_{c}" for c in contexts), "ridge_v1"]},
        "development_gate": gate, "lora_trigger": lora_trigger,
        "lora_status": "eligible_for_separate_training" if lora_trigger else "development_gate_not_met"}
