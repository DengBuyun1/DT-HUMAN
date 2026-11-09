"""Build the local V2 evaluation bundle and offline replay page."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import build
import swim
from evaluate import metrics, scored
from heat import rolling
from replay import heart_replay


def render(run, report, history, heart, heat, rest, sleep):
    template = (ROOT / "demo.html").read_text(encoding="utf-8")
    data = {"info": report["info"], "heart": {**heart, "rows": [r for r in heart["rows"] if r["selected"] is not None]},
            "heat": heat, "rest": rest, "sleep_exploration": sleep, "history": history, "report": report}
    inline = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    changes = {
        "/*__DT_DATA__*/null": inline,
        "本地教学原型 · 单人可穿戴记录": "第二版 · 按观测时间回放 · 单人可穿戴记录",
        '<footer id="footer">': '<section class="panel" id="evaluation" hidden><h2>第二版实验对照</h2><div id="v2Results"></div></section><footer id="footer">',
        "</nav>": '<button data-panel="evaluation" aria-pressed="false">第二版实验</button></nav>',
        "const historyDomain=": (HERE / "ui.js").read_text(encoding="utf-8") + "\nconst historyDomain=",
        "sleepExplore();\n  el": "sleepExplore();renderV2();\n  el",
        "const passed=h.decision.passed;": "const passed=Boolean(h.coefficients);",
    }
    for old, new in changes.items():
        if template.count(old) != 1:
            raise ValueError(f"Unexpected page template anchor: {old}")
        template = template.replace(old, new)
    template = template.replace("if(h.decision.passed) {", "if(h.coefficients) {")
    template = template.replace('passed?`范围内回顾验证通过展示门槛。当前曲线采用 ${names[h.model]}，训练时长 ${h.support.join("–")} 分钟。`:"范围内回顾验证未达到启用门槛。展示历史同类场次及回测误差。"', '`历史条件估算采用 ${names[h.model]}。全部场次校准范围 ${h.support.join("–")} 分钟。强基线对照与采用决定见实验页。`')
    template = template.replace('h.subgroups.in.selected', 'h.subgroups.in[h.model]').replace('h.subgroups.out.selected', 'h.subgroups.out[h.model]')
    template = template.replace('入门槛比较仅使用当折训练范围内的样本。', '日期等权误差按当折时长支持范围分组。')
    template = template.replace('const report=data.heart.metrics[horizon];',
        'if(last?.ref_days) el("replayStatus").textContent+=` · 历史误差参考 ${num(last.ref_low)}–${num(last.ref_high)} 次/分（此前 ${last.ref_days} 个日期）`;\n  const report=data.heart.metrics[horizon];')
    (run / "index.html").write_text(template, encoding="utf-8")


def run_experiment(run, options, config, started):
    raw = options.workbook.read_bytes()
    data = swim.read_workbook(raw)
    sessions, labels = swim.build_swims(data)
    with threadpool_limits(limits=1):
        old = build.backtest_swim(sessions)
        old_heat = build.backtest_heat(data, sessions)
        rest = build.backtest_rest(data)
        heart = heart_replay(sessions, run / "heart_replay.sqlite", config)
        heat = rolling(data, sessions, config, "I")
        active = rolling(data, sessions, config, "J")
    old_run = ROOT / "runs" / "20261109-110040-088694" / "metrics.json"
    reference = "recomputed"
    if old_run.is_file():
        original = json.loads(old_run.read_text(encoding="utf-8"))
        if original["info"]["source_sha256"] == sha256(raw).hexdigest():
            assert json.loads(json.dumps(old["metrics"])) == original["heart"], "Baseline heart changed"
            assert old_heat["metrics"] == original["heat"]["metrics"], "Baseline heat changed"
            assert rest["metrics"] == original["rest"]["metrics"], "Baseline resting heart rate changed"
            reference = "matches_archived_metrics"
    sleep = build.sleep_exploration(data)
    for name, rows in [("swim_heart", heart["rows"]), ("swim_energy_total", heat["rows"]),
                       ("swim_energy_active", active["rows"]), ("resting_hr", rest["rows"]),
                       ("sleep_exploration", sleep)]:
        build.csv_rows(run / f"{name}.csv", rows)
    daily = build.write_ledger(run / "daily_reference.sqlite",
                               {"swim_energy_conditional": heat["rows"], "resting_hr": rest["rows"]})
    chronos = {"status": "not_requested"}
    if options.chronos:
        from chronos_check import run as run_chronos
        try:
            rows, chronos = run_chronos(sessions, config, options.models_dir, options.device)
            chronos["status"] = "completed"
            build.csv_rows(run / "chronos_120s.csv", rows)
        except Exception:
            (run / "chronos_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            chronos = {"status": "failed", "failure_log": "chronos_error.txt"}
    history = build.observations(data)
    info = {"run_id": run.name, "started_at": started.isoformat(),
            "source_sha256": sha256(raw).hexdigest(), "activity_counts": labels,
            "ledger": {"scored": heart["ledger"]["scored"] + daily["scored"]},
            "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                            for p in [ROOT / "swim.py", ROOT / "build.py", ROOT / "demo.html",
                                      ROOT / "vendor/garmin_lab/registry/store.py", *sorted(HERE.glob("*.py")),
                                      HERE / "protocol.json", HERE / "ui.js"]}}
    heat_summary = lambda result: {k: v for k, v in result.items() if k not in ("rows", "samples")}
    common = {r["id"]: r for r in active["rows"]}
    paired = [{**r, "active_rate": common[r["id"]]["rate"], "active_line": common[r["id"]]["line"]}
              for r in heat["rows"] if r["id"] in common]
    report = {"info": info, "protocol": config, "coverage": {"swims": len(sessions),
                  "swim_dates": len({s.date for s in sessions}), "source_rows": sum(map(len, data.values())),
                  "sheet_rows": history["sheets"]}, "baseline_reproduction": reference,
              "heart": heart["metrics"], "heart_folds": heart["folds"], "heart_ledger": heart["ledger"],
              "heat_total": heat_summary(heat), "heat_active": heat_summary(active),
              "heat_paired_duration_metrics": {k: metrics(scored(paired, "rate", "line", "active_rate", "active_line"), k)
                                               for k in ("rate", "line", "active_rate", "active_line")},
              "resting_hr": {k: v for k, v in rest.items() if k != "rows"}, "chronos": chronos}
    render(run, report, history, heart, heat, rest, sleep)
    info["ended_at"] = datetime.now(build.TZ).isoformat()
    build.dump(run / "metrics.json", report)
    build.dump(run / "manifest.json", {"status": "completed", "argv": sys.argv, "info": info,
              "artifact_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}})
    print(run, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=ROOT / "data/健康数据.xlsx")
    parser.add_argument("--chronos", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models-dir", type=Path, default=ROOT.parent / "DT-modules")
    options = parser.parse_args()
    started = datetime.now(build.TZ)
    run = ROOT / "runs" / ("v2-" + started.strftime("%Y%m%d-%H%M%S-%f"))
    run.mkdir(parents=True)
    config = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    try:
        run_experiment(run, options, config, started)
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
