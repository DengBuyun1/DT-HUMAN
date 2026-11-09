"""Render the recorded DT experiment without refitting any models."""
import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from swim import build_swims, cases, read_workbook

ROOT = Path(__file__).resolve().parent
NAVY, BLUE, ORANGE, GREEN, GRAY = "#192f3e", "#0072b2", "#c6501a", "#008064", "#657681"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "text.color": NAVY, "axes.labelcolor": NAVY, "axes.titlecolor": NAVY})


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def values(rows, field):
    return np.array([float(row[field]) if row[field] not in (None, "") else np.nan for row in rows])


def save(fig, out, name):
    fig.savefig(out / name, dpi=155, facecolor="white")
    plt.close(fig)


def overview(out, bundle, train):
    history = bundle["history"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.8, 6.3), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("实验数据用量与心率预测覆盖", fontsize=20, fontweight="bold")
    dates = sorted({d for series in history["series"].values() for d, _, _ in series["points"]})
    labels = ["实时心率", "静息心率", "步数", "日活动热量", "血氧", "压力", "主睡眠时长", "小睡时长", "体重", "体脂率"]
    observed = [{d for d, _, _ in history["series"][name]["points"]} for name in labels]
    from matplotlib.colors import ListedColormap
    ax.imshow([[d in days for d in dates] for days in observed], aspect="auto",
              cmap=ListedColormap(["#edf2f3", GREEN]), vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)), [f"{label} · {len(days)} 天" for label, days in zip(labels, observed)])
    ticks = list(range(0, len(dates), 5))
    ax.set_xticks(ticks, [dates[i][5:] for i in ticks])
    ax.set_title("A  原始记录覆盖日", loc="left", fontweight="bold")
    ax.set_xlabel("绿格表示当天有记录，灰格表示当天缺测")
    horizons = [bundle["heart"]["metrics"][str(h)] for h in (30, 120)]
    scored = [item["metrics"][item["model"]]["n"] for item in horizons]
    unscored = [item["issued"] - count for item, count in zip(horizons, scored)]
    unavailable = [item["eligible"] - item["issued"] for item in horizons]
    positions = [1, 0]
    for counts, left, color, label in [(scored, [0, 0], BLUE, "已评分"), (unscored, scored, ORANGE, "已预测但目标缺测"),
                                      (unavailable, np.add(scored, unscored), "#dbe5e8", "输入不足")]:
        bars = bx.barh(positions, counts, left=left, color=color, height=.44, label=label)
        bx.bar_label(bars, label_type="center", color="white" if color != "#dbe5e8" else NAVY)
    bx.set_yticks(positions, ["提前 30 秒", "提前 2 分钟"])
    bx.set_ylim(-1.45, 1.45)
    bx.set_xlim(0, 270)
    bx.xaxis.set_major_locator(MaxNLocator(integer=True))
    bx.set_title("B  六场留出游泳的候选窗口", loc="left", fontweight="bold")
    bx.text(0, -.52, f"开发期 8 场、8 个日期，训练窗分别为 {train[30]} 和 {train[120]}。\n"
                     "热量 14 场，回测 10 场、7 个日期。\n静息心率 29 个日对，滚动评分 15 天。\n睡眠探索 10 晚，两种终点分别配对 10 晚和 9 晚。",
            va="top", fontsize=10, linespacing=1.45)
    bx.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=2, frameon=False, fontsize=9)
    save(fig, out, "00_实验样本与覆盖.png")


def heart_figure(out, bundle, rows, horizon):
    tests = [session for session in bundle["heart"]["sessions"] if session["held_out"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharey=True)
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.13, hspace=.42, wspace=.16)
    fig.suptitle(f"游泳心率留出回放 · 提前 {horizon} 秒", fontsize=21, fontweight="bold", y=.965)
    model = "persistence" if horizon == 30 else "ridge"
    used = [r for r in rows if int(r["horizon"]) == horizon]
    for axis, session in zip(axes.flat, tests):
        selected = [r for r in used if r["session"] == session["id"]]
        seconds = values(selected, "target_s")
        actual, predicted = values(selected, "observed"), values(selected, model)
        error = np.abs(actual - predicted)
        eligible = max(0, len(session["windows"]) - 2 - horizon // 30)
        observed = np.array([[x, y if y is not None else np.nan] for x, y in session["windows"]])
        axis.plot(observed[:, 0] / 60, observed[:, 1], color=NAVY, lw=1.4, marker=".", ms=4)
        for key, color, line in [("persistence", GREEN, ":"), ("ridge", BLUE, "--")]:
            time_grid = observed[:, 0]
            predictions = dict(zip(seconds, values(selected, key)))
            axis.plot(time_grid / 60, [predictions.get(t, np.nan) for t in time_grid],
                      linestyle=line, color=color, lw=1.45, marker=".", ms=3.8)
        misses = error > 10
        axis.scatter(seconds[misses] / 60, actual[misses], marker="o", s=32,
                     facecolors="none", edgecolors=ORANGE, zorder=4)
        missing = np.isnan(actual)
        axis.scatter(seconds[missing] / 60, predicted[missing], marker="x", s=23, color=GRAY, zorder=5)
        count = int(np.isfinite(error).sum())
        title = f"{session['date'][5:]}   {session['id']}   {count}/{eligible} 窗可评分"
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("距场次开始 / 分钟")
        axis.grid(axis="y", alpha=.20)
        axis.text(.98, .04, f"默认模型 MAE {np.nanmean(error):.1f}" if count else "本场无配对点",
                  transform=axis.transAxes, ha="right", color=GRAY, fontsize=9,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86})
    for axis in axes[:, 0]:
        axis.set_ylabel("心率 / 次每分钟")
    fig.legend(handles=[Line2D([0], [0], color=NAVY, label="30 秒窗实测"),
                        Line2D([0], [0], color=GREEN, ls=":", label="最近心率"),
                        Line2D([0], [0], color=BLUE, ls="--", label="个人岭回归"),
                        Line2D([0], [0], color=ORANGE, marker="o", markerfacecolor="none", ls="", label="默认模型误差超过 10"),
                        Line2D([0], [0], color=GRAY, marker="x", ls="", label="已预测但目标缺测")],
               loc="upper center", bbox_to_anchor=(.5, .915), ncol=5, frameon=False)
    fig.text(.07, .035, "所有小图共用纵轴。各场横轴随场次长度变化。默认模型为" + ("最近心率。" if horizon == 30 else "个人岭回归。"), color=GRAY)
    save(fig, out, f"01_游泳心率_{horizon}秒.png")


def training_figure(out, training):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), sharey=True, layout="constrained")
    fig.suptitle("游泳心率开发期 · 8 场真实观测", fontsize=20, fontweight="bold")
    for axis, session in zip(axes.flat, training):
        n30 = len(cases([session], 30))
        n120 = len(cases([session], 120))
        axis.plot(session.seconds / 60, session.heart, color=NAVY, lw=1.35, marker=".", ms=3)
        axis.set_title(f"{session.date[5:]}   {session.key}\n训练窗 30 秒 {n30} · 2 分钟 {n120}",
                       loc="left", fontsize=10)
        axis.set_xlabel("距场次开始 / 分钟")
        axis.grid(axis="y", alpha=.18)
    for axis in axes[:, 0]:
        axis.set_ylabel("心率 / 次每分钟")
    save(fig, out, "01_游泳心率_开发期.png")


def heat_figure(out, bundle, rows):
    heat = bundle["heat"]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0), layout="constrained")
    fig.suptitle("游泳时长与设备整场热量", fontsize=20, fontweight="bold")
    ax, bx, cx, dx = axes.flat
    samples = heat["samples"]
    for row in samples:
        ax.scatter(row["minutes"], row["observed"], color=NAVY, s=29)
    intercept, slope = heat["coefficients"]
    x = np.linspace(*heat["support"], 100)
    ax.plot(x, np.maximum(0, intercept + slope*x), color=BLUE, label="全 14 场拟合曲线")
    plans = [25, 40, 55]
    ax.scatter(plans, np.maximum(0, intercept + slope*np.array(plans)), marker="D", color=ORANGE, label="三档演示输入")
    ax.set(xlabel="整场时长 / 分钟", ylabel="设备热量 / 千卡", title="A  全部 14 场与演示曲线")
    ax.legend(frameon=False, fontsize=9)
    x = np.arange(len(rows))
    names = [row["id"] for row in rows]
    models = [("historical", "历史中位数", GRAY, "v"), ("rate", "每分钟热量", GREEN, "s"),
              ("line", "约束线性回归", BLUE, "^"), ("selected", "滚动选用", ORANGE, "D")]
    bx.plot(x, values(rows, "observed"), "o-", color=NAVY, label="实测", lw=1.1)
    for key, label, color, marker in models:
        bx.scatter(x, values(rows, key), color=color, marker=marker, s=22, label=label)
    bx.set(xticks=x, xticklabels=names, ylabel="设备热量 / 千卡", title="B  10 场滚动留出 · 实测与全部候选")
    bx.legend(ncol=3, frameon=False, fontsize=8)
    in_scope = [row["in_support"] == "True" for row in rows]
    errors = values(rows, "selected") - values(rows, "observed")
    cx.bar(x, errors, color=[BLUE if v else ORANGE for v in in_scope])
    cx.axhline(0, color=NAVY, lw=.8)
    cx.set(xticks=x, xticklabels=names, ylabel="预测减实测 / 千卡", title="C  逐场有符号误差")
    cx.legend(handles=[Line2D([0], [0], color=BLUE, lw=6, label="当折范围内 · 6 场"),
                       Line2D([0], [0], color=ORANGE, lw=6, label="当折范围外 · 4 场")], frameon=False, fontsize=9)
    positions = np.arange(len(models))
    for offset, group, color, label in [(-.16, "in", BLUE, "范围内 · 5 日期"), (.16, "out", ORANGE, "范围外 · 2 日期")]:
        vals = [heat["subgroups"][group][key]["date_mae"] for key, *_ in models]
        bars = dx.barh(positions + offset, vals, .29, color=color, label=label)
        dx.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    dx.set(yticks=positions, yticklabels=[label for _, label, *_ in models], xlim=(0, 235),
           xlabel="日期等权 MAE / 千卡", title="D  按当折训练时长范围分组")
    dx.invert_yaxis()
    dx.legend(frameon=False, fontsize=9, loc="lower right")
    for axis in axes.flat:
        axis.grid(axis="y" if axis is not dx else "x", alpha=.18)
    save(fig, out, "02_运动热量回测.png")


def rest_and_sleep(out, rest, sleep):
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 10.5), sharex=True, layout="constrained",
                            gridspec_kw={"height_ratios": [2, 2, 1, 1]})
    fig.suptitle("次日设备静息心率 · 15 天滚动留出", fontsize=20, fontweight="bold")
    x = np.arange(len(rest))
    actual = values(rest, "observed")
    axes[0].plot(x, actual, "o-", color=NAVY, label="设备实测")
    for key, name, color, line in [("historical", "历史中位数", GRAY, "--"), ("last", "最近值", GREEN, ":"),
                                    ("minutes", "运动分钟", BLUE, "-."), ("steps", "步数", ORANGE, "--")]:
        y = values(rest, key)
        axes[0].plot(x, y, linestyle=line, marker=".", color=color, label=name, lw=1.45)
        axes[1].plot(x, np.abs(y-actual), linestyle=line, marker=".", color=color, lw=1.45,
                     label=f"{name}   MAE {mean(np.abs(y-actual)):.2f}")
    axes[0].set(ylabel="心率 / 次每分钟", title="A  实测与四条预测")
    axes[1].set(ylabel="绝对误差 / 次每分钟", title="B  同日配对误差", ylim=(0, None))
    for axis in axes[:2]:
        axis.legend(ncol=3, frameon=False, fontsize=9)
    for axis, key, label, title, color in [(axes[2], "input_minutes", "分钟", "C  前一日运动时间", BLUE),
                                           (axes[3], "input_steps", "步", "D  前一日步数", ORANGE)]:
        axis.bar(x, values(rest, key), color=color, width=.65)
        axis.set(ylabel=label, title=title, ylim=(0, None))
    axes[-1].set(xticks=x, xticklabels=[r["date"][5:] for r in rest], xlabel="静息心率目标日期")
    for axis in axes:
        axis.grid(axis="y", alpha=.19)
    save(fig, out, "03_静息心率回测.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    fig.suptitle("主睡眠时长与后续记录 · 10 晚探索", fontsize=19, fontweight="bold")
    for axis, key, label, color in [(axes[0], "next_rest", "次日静息心率 / 次每分钟", BLUE),
                                    (axes[1], "afternoon_pressure", "同日午后压力 / 分", GREEN)]:
        x, y = values(sleep, "minutes"), values(sleep, key)
        axis.scatter(x, y, c=color, s=35, alpha=.8)
        axis.set(xlabel="主睡眠时长 / 分钟", ylabel=label,
                 title=f"{label.split(' / ')[0]} · {np.isfinite(y).sum()} 晚")
        axis.grid(alpha=.18)
    axes[2].barh(np.arange(len(sleep)), values(sleep, "pressure_hours"), color=[GREEN if r["afternoon_pressure"] else GRAY for r in sleep])
    axes[2].axvline(3, color=ORANGE, ls="--", label="至少 3 小时")
    axes[2].set(yticks=np.arange(len(sleep)), yticklabels=[r["date"][5:] for r in sleep], xlim=(0, 5.4),
                xlabel="满足每小时至少两条的小时数", title="13:00–18:00 压力覆盖")
    axes[2].invert_yaxis()
    axes[2].legend(frameon=False, fontsize=9)
    save(fig, out, "04_睡眠配对与覆盖.png")


def gallery(out, run, train):
    analysis = ROOT.parent / "测试数据" / "分析"
    prior = analysis / "孪生验收总览_2026-11-06"
    earlier = [prior / name for name in ("00_验收总览.png", "01_游泳心率逐场.png", "02_小时指标对照.png",
                                        "03_日活动与睡眠.png", "04_体成分与条件情景.png")]
    earlier.append(analysis / "多模型孪生测试_2026-11-06" / "多模型效果总览.png")
    current = [out / name for name in ("00_实验样本与覆盖.png", "01_游泳心率_开发期.png", "01_游泳心率_30秒.png", "01_游泳心率_120秒.png",
                                       "02_运动热量回测.png", "03_静息心率回测.png", "04_睡眠配对与覆盖.png")]

    def link(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        return escape(Path(os.path.relpath(path, out)).as_posix(), quote=True)

    def figures(paths):
        return "\n".join(f'<figure><figcaption>{escape(p.stem)}</figcaption><a href="{link(p)}">'
                         f'<img src="{link(p)}" alt="{escape(p.stem)}" loading="lazy"></a></figure>' for p in paths)

    files = [run / name for name in ("metrics.json", "swim_heart.csv", "swim_energy_conditional.csv",
                                      "resting_hr.csv", "sleep_exploration.csv", "predictions.sqlite")]
    downloads = " · ".join(f'<a href="{link(p)}">{escape(p.name)}</a>' for p in files)
    (out / "index.html").write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>孪生实验图册</title>
<style>body{{font:16px/1.65 "Microsoft YaHei",sans-serif;color:#192f3e;background:#edf3f4;margin:0}}main{{max-width:1180px;margin:auto;padding:20px}}figure{{background:white;margin:18px 0;padding:15px}}img{{width:100%;height:auto}}figcaption{{font-weight:bold}}a{{color:#006294}}nav{{line-height:2.2}}</style>
<main><h1>个人活动数字孪生实验图册</h1><p>单个账号，2026 年 9 月 29 日至 10 月 29 日。原始 Excel 共 15 张表，10 张非空表、30,940 行。各任务按场次、日期或睡眠事件计数。</p>
<p>游泳心率开发期 8 场，提前 30 秒和 120 秒分别使用 {train[30]} 和 {train[120]} 个训练窗。热量从 4 场起滚动拟合，留出 10 场。静息心率从 14 个日对起滚动拟合，留出 15 天。</p>
<nav><a href="#current">本轮图表</a> · <a href="#earlier">前轮指标与模型对照</a> · <a href="{link(run / 'index.html')}">全量记录与交互回放</a></nav>
<p>{downloads}</p><h2 id="current">11 月 9 日原型回测</h2>{figures(current)}
<h2 id="earlier">11 月 6 日前轮结果</h2><p>这些图保留当时的模型、数据切分与验收口径。体重、体脂和光照曲线为条件情景。</p>{figures(earlier)}
</main></html>''', encoding="utf-8")


def main(run):
    out = run / "可视化"
    out.mkdir(exist_ok=True)
    page = (run / "index.html").read_text(encoding="utf-8")
    bundle = json.JSONDecoder().raw_decode(page.split("const data = ", 1)[1])[0]
    heart, heat, rest, sleep = [load_rows(run / name) for name in (
        "swim_heart.csv", "swim_energy_conditional.csv", "resting_hr.csv", "sleep_exploration.csv")]
    workbook = read_workbook((ROOT / "data" / "健康数据.xlsx").read_bytes())
    swims, _ = build_swims(workbook)
    cutoff = max(row["fit_through"] for row in heart)
    training = [session for session in swims if session.date <= cutoff]
    train = {h: len(cases(training, h)) for h in (30, 120)}
    assert len(heart) + len(heat) + len(rest) == bundle["info"]["ledger"]["issued"]
    assert sum(bool(r["observed"]) for r in heart + heat + rest) == bundle["info"]["ledger"]["scored"]
    print(f"HR training: {len(training)} sessions, {len({s.date for s in training})} dates, windows {train}")
    print(f"Heat fit sizes: {sorted({r['train_n'] for r in heat}, key=int)}")
    print(f"Rest fit sizes: {min(int(r['train_n']) for r in rest)} to {max(int(r['train_n']) for r in rest)}")
    overview(out, bundle, train)
    training_figure(out, training)
    for horizon in (30, 120):
        heart_figure(out, bundle, heart, horizon)
    heat_figure(out, bundle, heat)
    rest_and_sleep(out, rest, sleep)
    gallery(out, run, train)
    print(out / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing DTs/runs run directory")
    main(parser.parse_args().run.resolve())
