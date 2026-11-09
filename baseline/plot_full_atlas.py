"""Generate a local atlas of the source workbook and its recorded V2 experiments."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from statistics import mean, median
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
from matplotlib import dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

import swim


ROOT = Path(__file__).resolve().parent
PROFILE = ROOT.parent / "测试数据/分析/全量数据核查_2026-11-09/workbook_profile.json"
INK, BLUE, TEAL, AMBER, PURPLE, GRAY = "#182f3d", "#0072B2", "#008269", "#B74619", "#745495", "#778a96"
MODELS = {"persistence": ("最近值", GRAY, ":"), "ridge_v1": ("原版 Ridge", BLUE, "-"),
          "ridge": ("加权 Ridge", TEAL, "--"), "kalman": ("Kalman", AMBER, "-."),
          "historical": ("历史中位数", GRAY, ":"), "rate": ("每分钟换算", BLUE, "-"),
          "line": ("约束直线", AMBER, "--"), "legacy_selected": ("旧滚动选择", PURPLE, "-."),
          "last": ("最近静息心率", PURPLE, ":"), "minutes": ("运动分钟修正", BLUE, "-"),
          "steps": ("步数修正", TEAL, "--")}
TIME = {"心率": "H", "预警数据": "S", "睡眠": "G", "血氧饱和度": "G", "日常活动": "I",
        "单次运动": "G", "压力": "G", "体重": "U", "活动统计": "N", "日常活动状态": "H"}
STAGES = {"浅睡": "#5886b7", "深睡": "#2b4979", "快速眼动": "#9471a2",
          "清醒": "#ce7646", "零星小睡": "#439b89"}
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
                     "pdf.fonttype": 42, "axes.unicode_minus": False})


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key):
    val = row.get(key)
    return float(val) if val not in ("", None) else np.nan


def numbers(rows, key):
    return np.array([number(r, key) for r in rows])


def panel(ax, name, unit=None, grid="y"):
    ax.set_title(name, loc="left", fontweight="bold", fontsize=11, pad=8)
    if unit:
        ax.set_ylabel(unit)
    if grid:
        ax.grid(axis=grid, alpha=.16, color=GRAY)


def dates(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))


def frame(title, note, rows, cols, size=None, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=size or (14, 3.0 * rows + 1.0),
                             layout="constrained", squeeze=False, **kwargs)
    fig.suptitle(title + "\n" + note, ha="left", x=.025, fontsize=15, fontweight="bold")
    return fig, axes.flat


class Atlas:
    def __init__(self, out, pdf):
        self.out, self.pdf = out, pdf
        self.items = []

    def add(self, fig, group, title, note):
        name = f"{len(self.items) + 1:02d}_{title.replace(' ', '_')}.png"
        path = self.out / name
        fig.savefig(path, dpi=155, facecolor="white")
        self.pdf.savefig(fig, facecolor="white")
        plt.close(fig)
        self.items.append({"group": group, "title": title, "note": note, "png": name})


def overview(atlas, data, start, end):
    names = list(data)
    day_range = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    daily = {s: Counter(date.fromisoformat(r[TIME[s]][:10]) for r in data[s]) if s in TIME else Counter()
             for s in names}
    fig, (a, b) = frame("原始工作簿覆盖总览", "颜色表示当天有原始记录。不同表的行数代表不同记录单位。",
                        1, 2, (15, 7.2), gridspec_kw={"width_ratios": [1.4, 1]})
    a.imshow([[bool(daily[s][d]) for d in day_range] for s in names], aspect="auto",
             cmap=ListedColormap(["#edf1f3", TEAL]), vmin=0, vmax=1)
    a.set_yticks(range(len(names)), [f"{s} · {len(daily[s])}天" for s in names])
    ticks = range(0, len(day_range), 5)
    a.set_xticks(list(ticks), [day_range[i].strftime("%m/%d") for i in ticks])
    panel(a, "A  每日记录是否存在", grid=None)
    a.set_xlabel("日期")
    n = [len(data[s]) for s in names]
    b.barh(range(len(names)), n, color=[BLUE if v else "#d6e1e6" for v in n])
    b.set_xscale("symlog", linthresh=1)
    for i, v in enumerate(n):
        b.annotate(f"{v:,}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    b.set_ylim(len(names) - .5, -.5)
    b.set_yticks(range(len(names)), names)
    b.set_xlim(0, max(n) * 4)
    panel(b, "B  原始表记录量", grid="x")
    b.set_xlabel("行数（大于 1 的刻度按对数排列）")
    atlas.add(fig, "原始观测", "工作簿覆盖与行数", f"{len(names)} 张表，{sum(bool(data[s]) for s in names)} 张有记录，共 {sum(n):,} 行。")


def plot_column(ax, sheet, col, spec, rows):
    label = spec["label"].strip()
    subset = [r for r in rows if col in r]
    if not subset:
        panel(ax, label, grid=None)
        ax.text(.5, .5, "无记录", transform=ax.transAxes, ha="center", color=GRAY)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    x = [datetime.fromisoformat(r[TIME[sheet]]) for r in subset]
    if col in spec["time_cols"]:
        t = [datetime.fromisoformat(r[col]) for r in subset]
        count = Counter(v.date() for v in t)
        days = sorted(count)
        ax.bar(days, [count[d] for d in days], width=.72, color=TEAL)
        panel(ax, f"{label} · {len(subset):,} 条", "每天记录数")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        dates(ax)
    elif "min" in spec:
        y = numbers(subset, col)
        unit = "原表数值"
        if sheet == "体重":
            unit = {"F": "kg", "I": "%", "M": "%", "P": "%"}.get(col, unit)
        if sheet == "心率":
            for kind, color in [("实时心率", BLUE), ("静息心率", AMBER)]:
                pairs = [(a, b) for a, b, r in zip(x, y, subset) if r.get("G") == kind]
                if pairs:
                    xx, yy = zip(*pairs)
                    ax.scatter(xx, yy, s=6 if len(pairs) > 100 else 18, alpha=.42 if len(pairs) > 100 else .9,
                               color=color, label=kind, linewidths=0)
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.scatter(x, y, s=5 if len(x) > 300 else 19, alpha=.42 if len(x) > 300 else .87,
                       linewidths=0, color=BLUE)
        suffix = " · 全零" if np.all(y == 0) else (" · 恒定" if np.ptp(y) == 0 else "")
        panel(ax, f"{label} · {len(subset):,} 条{suffix}", unit)
        if suffix:
            ax.set_yticks([y[0]])
        dates(ax)
    else:
        counts = Counter(r[col] for r in subset)
        cats, values = zip(*counts.most_common())
        ax.barh(range(len(cats)), values, color=BLUE, height=.6)
        ax.set_yticks(range(len(cats)), cats, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.27)
        for i, n in enumerate(values):
            ax.annotate(str(n), (n, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
        panel(ax, f"{label} · {len(subset):,} 条", grid="x")
        ax.set_xlabel("记录数")


def sheet_pages(atlas, data, profile):
    catalog = []
    for sheet in profile["sheets"]:
        name, rows = sheet["name"], data[sheet["name"]]
        assert len(rows) == sheet["rows"], name
        fields = list(sheet["columns"].items())
        for key, spec in fields:
            n = sum(key in r for r in rows)
            assert n == spec["nonempty"], (name, key)
            kind = ("无记录" if not n else "时间" if key in sheet["times"] else
                    "全零" if "min" in spec and spec["min"] == spec["max"] == 0 else
                    "数值恒定" if "min" in spec and spec["min"] == spec["max"] else
                    "数值" if "min" in spec else "类别")
            catalog.append({"工作表": name, "列": key, "字段": spec["label"].strip(),
                            "总行数": len(rows), "非空": n, "缺测": len(rows) - n, "类型": kind})
        if not rows:
            continue
        batches = [fields[i:i+12] for i in range(0, len(fields), 12)]
        for k, group in enumerate(batches, start=1):
            cols = min(3, len(group))
            title = name if len(batches) == 1 else f"{name} {k}"
            fig, axes = frame(f"{name}  ·  {len(rows):,} 条原始记录",
                              "散点为原始数值。全零和恒定字段按设备原记录绘制。",
                              (len(group) + cols - 1) // cols, cols)
            axs = list(axes)
            for ax, (col, spec) in zip(axs, group):
                spec = {**spec, "time_cols": sheet["times"]}
                if name == "睡眠" and col in ("K", "L"):
                    spec["label"] = f"{spec['label']}（{col} 列）"
                plot_column(ax, name, col, spec, rows)
            for ax in axs[len(group):]:
                ax.set_visible(False)
            atlas.add(fig, "原始观测", title, "含数值、时间计数、分类和无记录字段。" if len(batches) == 1 else
                      f"第 {k} 组字段。各面板单位沿用原表，缺测留空。")
    with (atlas.out / "字段覆盖.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)
    return catalog


def sleep_stages(atlas, data):
    events = sorted(data["睡眠"], key=lambda r: r["F"])
    states = [(datetime.fromisoformat(r["H"]), r["F"], float(r["G"]))
              for r in data["日常活动状态"] if r["F"] in STAGES]
    fig, (a, b) = frame("睡眠时间与设备分期", "每行对应一个睡眠事件。时间为事件内相对分钟。",
                        1, 2, (15, 8), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, r in enumerate(events):
        start, end = map(datetime.fromisoformat, [r["F"], r["G"]])
        a.barh(i, (end - start).total_seconds() / 60, height=.65, color="#e4eaed")
        by_stage = defaultdict(list)
        for t, stage, minutes in states:
            if start <= t < end:
                by_stage[stage].append(((t - start).total_seconds()/60, minutes))
        for stage, segments in by_stage.items():
            a.broken_barh(segments, (i - .325, .65), facecolors=STAGES[stage], linewidth=0)
    a.set_yticks(range(len(events)), [f"{r['F'][5:16]}  {r['O']}" for r in events], fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("入睡后分钟")
    a.legend(handles=[Line2D([0], [0], lw=5, color=c, label=k) for k, c in STAGES.items()],
             frameon=False, ncol=3, loc="lower right")
    panel(a, "A  事件与逐段状态", grid="x")
    for kind, color in [("科学睡眠", BLUE), ("零星小睡", TEAL)]:
        subset = [r for r in events if r["O"] == kind]
        b.scatter([datetime.fromisoformat(r["G"]) for r in subset], numbers(subset, "H"), s=35,
                  color=color, label=kind, linewidths=0)
    dates(b)
    b.legend(frameon=False)
    panel(b, "B  睡眠时长与结束日期", "分钟")
    atlas.add(fig, "原始观测", "睡眠事件与分期", f"共 {len(events)} 个睡眠事件，灰色为事件跨度内未显示分期的时间。")


def swim_source(atlas, sessions):
    fig, axes = frame("14 场游泳的原始心率窗口", "每点为 30 秒内至少两条有效心率的中位数，缺窗保留空档。",
                      4, 4, (16, 11), sharey=True)
    axs = list(axes)
    for ax, session in zip(axs, sessions):
        ax.plot(session.seconds/60, session.heart, color=BLUE, lw=1.3, marker=".", ms=2.5)
        panel(ax, f"{session.key}  {session.date[5:]} · {session.duration/60:.0f}分")
        ax.set_xlabel("运动分钟")
    for ax in axs[len(sessions):]:
        ax.set_visible(False)
    fig.supylabel("心率（次/分）", fontsize=10)
    atlas.add(fig, "原始观测", "游泳心率原始窗口", "全部14场，统一纵轴，显示训练与评估的原始心率窗。")


def line(ax, x, y, key):
    label, color, style = MODELS[key]
    ax.plot(x, y, linestyle=style, color=color, linewidth=1.55, label=label)


def fig_legend(fig, keys, actual=True):
    handles = [Line2D([0], [0], color=INK, lw=2.3, label="设备实测")] if actual else []
    handles.extend(Line2D([0], [0], color=MODELS[k][1], linestyle=MODELS[k][2],
                          lw=1.6, label=MODELS[k][0]) for k in keys)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)


def v2_overview(atlas, report):
    fig, axes = frame("第二版模型误差与覆盖", "每个时距在相同目标窗上比较，日期等权 MAE 先按场次汇总再按日期汇总。",
                      2, 2, (13.5, 7.6))
    max_mae = max(model["date_mae"] for h in ("30", "120") for model in report["heart"][h]["metrics"].values())
    max_count = max(report["heart"][h]["eligible"] for h in ("30", "120"))
    for h in ("30", "120"):
        ax, bx = next(axes), next(axes)
        summary = report["heart"][h]
        models = summary["metrics"]
        x = np.arange(len(models))
        vals = [m["date_mae"] for m in models.values()]
        bars = ax.bar(x, vals, color=[MODELS[k][1] for k in models], width=.63)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylim(0, max_mae * 1.18)
        ax.set_xticks(x, [MODELS[k][0] for k in models], fontsize=9)
        panel(ax, f"提前 {h} 秒 · 日期等权 MAE", "次/分")
        score = next(iter(models.values()))["n"]
        counts = [score, summary["missing_target"], summary["eligible"] - summary["issued"]]
        labels = ["已评分", "目标缺测", "输入不足"]
        bars = bx.barh(labels[::-1], counts[::-1], color=["#cdd8de", AMBER, BLUE])
        bx.bar_label(bars, padding=4)
        bx.set_xlim(0, max_count)
        bx.set_xlabel("候选窗数")
        panel(bx, f"提前 {h} 秒 · {summary['eligible']} 个记录范围内窗口", grid="x")
    atlas.add(fig, "第二版实验", "心率模型与预测覆盖", "同时显示模型误差、可评分数、目标缺测和输入不足。")


def heart_replay(atlas, rows, sessions, h):
    selected = [r for r in rows if int(r["horizon"]) == h and r["in_record"] == "True"]
    keys = ("persistence", "ridge_v1", "ridge", "kalman")
    test = [s for s in sessions if any(r["session"] == s.key for r in selected)]
    fig, axes = frame(f"提前 {h} 秒心率 · 逐场实测与四种模型",
                      "横轴为预测目标时刻。缺失观测和未发出预测保留空档。",
                      (len(test) + 3) // 4, 4, (16, 7.2), sharey=True)
    axs = list(axes)
    for ax, session in zip(axs, test):
        r = sorted([r for r in selected if r["session"] == session.key], key=lambda r: int(r["target_s"]))
        x = numbers(r, "target_s")/60
        ax.plot(session.seconds/60, session.heart, color=INK, linewidth=2.1, alpha=.82)
        for k in keys:
            line(ax, x, numbers(r, k), k)
        panel(ax, f"{session.key}  {session.date[5:]} · {sum(t['status']=='scored' for t in r)}窗")
        ax.set_xlabel("运动分钟")
    for ax in axs[len(test):]:
        ax.set_visible(False)
    fig.supylabel("心率（次/分）", fontsize=10)
    fig_legend(fig, keys)
    atlas.add(fig, "第二版实验", f"游泳心率逐场 {h}秒", f"{len(test)} 场、{len({r['date'] for r in selected})} 个日期的逐窗曲线。")


def date_errors(rows, key):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        y, yhat = number(r, "observed"), number(r, key)
        if np.isfinite(y) and np.isfinite(yhat):
            grouped[r["date"]][r["session"]].append(abs(y - yhat))
    return {d: mean(map(mean, s.values())) for d, s in grouped.items()}


def heart_errors(atlas, rows, report):
    fig, axes = frame("心率逐日误差与误差参考", "箱线图描述逐窗误差。橙色区间取自此前已评分日期。",
                      2, 3, (16, 8.2))
    keys = ("persistence", "ridge_v1", "ridge", "kalman")
    for h in ("30", "120"):
        a, b, c = next(axes), next(axes), next(axes)
        rr = [r for r in rows if r["horizon"] == h and r["status"] == "scored"]
        for key in keys:
            d = date_errors(rr, key)
            line(a, [day[5:] for day in d], list(d.values()), key)
        a.tick_params(axis="x", labelrotation=27)
        panel(a, f"{h} 秒 · 日期等权误差", "次/分")
        box = b.boxplot([np.abs(numbers(rr, k) - numbers(rr, "observed")) for k in keys], patch_artist=True,
                        tick_labels=[MODELS[k][0] for k in keys], widths=.57, flierprops={"markersize": 3})
        for patch, k in zip(box["boxes"], keys):
            patch.set_facecolor(MODELS[k][1])
            patch.set_alpha(.46)
        b.tick_params(axis="x", labelrotation=20)
        panel(b, f"{h} 秒 · 逐窗绝对误差", "次/分")
        band = [r for r in rr if r["ref_days"] != "0"]
        c.axhline(0, color=GRAY, lw=.8)
        days = sorted({r["date"] for r in band})
        for i, d in enumerate(days):
            g = [r for r in band if r["date"] == d]
            y = numbers(g, "observed") - numbers(g, "selected")
            lo = number(g[0], "ref_low") - number(g[0], "selected")
            hi = number(g[0], "ref_high") - number(g[0], "selected")
            c.fill_between([i - .26, i + .26], lo, hi, color=AMBER, alpha=.20)
            jitter = np.random.default_rng(i).uniform(-.20, .20, size=len(g))
            c.scatter(i + jitter, y, s=10, alpha=.46, color=BLUE)
        c.set_xticks(range(len(days)), [d[5:] for d in days])
        if days:
            c.set_xlim(-.48, len(days) - .52)
        m = report["heart"][h]["empirical_band"]
        panel(c, f"{h} 秒 · 参考范围命中 {m['hits']}/{m['n']}", "实测 − 预测（次/分）")
    fig_legend(fig, keys, actual=False)
    atlas.add(fig, "第二版实验", "心率逐日误差与历史参考", "含逐日 MAE、逐窗绝对误差分布，以及历史误差参考范围与实际误差。")


def heat_page(atlas, rows, summary, active):
    label = "活动时间（运动后回顾）" if active else "总时长（计划输入）"
    fig, (a, b, c, d) = frame(f"游泳设备热量 · {label}",
                              "每个点对应一场。采用当时训练时长范围区分范围内与范围外。",
                              2, 2, (14, 8.4))
    keys = ("historical", "rate", "line", "legacy_selected")
    x = np.arange(len(rows))
    y = numbers(rows, "observed")
    a.plot(x, y, marker="o", color=INK, lw=2, ms=4)
    for key in keys:
        line(a, x, numbers(rows, key), key)
    a.set_xticks(x, [r["id"] for r in rows])
    panel(a, "A  逐场实测与估算", "千卡")
    for axis, key in [(b, "rate"), (c, "line")]:
        for scope, color, marker in [("True", BLUE, "o"), ("False", AMBER, "^")]:
            g = [r for r in rows if r["in_support"] == scope]
            axis.scatter(numbers(g, "minutes"), numbers(g, key) - numbers(g, "observed"),
                         color=color, marker=marker, s=40, label="范围内" if scope == "True" else "范围外")
        axis.axhline(0, color=GRAY, lw=.9)
        axis.set_xlabel("分钟")
    panel(b, "B  每分钟换算偏差", "估算 − 实测（千卡）")
    b.legend(frameon=False)
    panel(c, "C  约束直线偏差", "估算 − 实测（千卡）")
    positions = np.arange(len(keys))
    for offset, scope, color, name in [(-.23, "all", GRAY, "全部"), (0, "in", BLUE, "范围内"),
                                        (.23, "out", AMBER, "范围外")]:
        vals = [summary["subgroups"][scope][k]["date_mae"] for k in keys]
        d.bar(positions+offset, vals, width=.22, color=color, label=name)
    d.set_xticks(positions, [MODELS[k][0] for k in keys], fontsize=9)
    d.legend(frameon=False, ncol=3, fontsize=8)
    panel(d, "D  日期等权误差分组", "MAE（千卡）")
    fig_legend(fig, keys)
    atlas.add(fig, "第二版实验", "热量回测 活动时间" if active else "热量回测 总时长",
              f"同一批 {len(rows)} 场，展示不同模型和适用范围的误差。")


def duration_sensitivity(atlas, rows, active, report):
    match = {r["id"]: r for r in active}
    rows = [r for r in rows if r["id"] in match]
    fig, (a, b) = frame("游泳时长定义与热量误差", "相同场次对照。活动时间字段在运动结束后获得。",
                        1, 2, (14, 5.3))
    total = numbers(rows, "minutes")
    act = numbers([match[r["id"]] for r in rows], "minutes")
    a.scatter(total, act, s=40, color=BLUE)
    high = max(total.max(), act.max()) * 1.08
    a.plot([0, high], [0, high], color=GRAY, linestyle="--", lw=1)
    a.set(xlim=(0, high), ylim=(0, high), xlabel="总时长（分钟）", ylabel="活动时间（分钟）")
    panel(a, "A  同场两种时长")
    keys = ("rate", "active_rate", "line", "active_line")
    m = report["heat_paired_duration_metrics"]
    values = [m[k]["date_mae"] for k in keys]
    bars = b.barh(["总时长换算", "活动时间换算", "总时长直线", "活动时间直线"][::-1],
                  values[::-1], color=[TEAL, BLUE, TEAL, BLUE])
    b.bar_label(bars, fmt="%.2f", padding=4)
    b.set_xlim(0, max(values) * 1.23)
    b.set_xlabel("日期等权 MAE（千卡）")
    panel(b, "B  对应热量估算误差", grid="x")
    atlas.add(fig, "第二版实验", "两种时长的同场比较", "总时长与活动时间在共同样本上比较。")


def chronos_pages(atlas, rows, report):
    context = report["chronos"]["evaluation"]
    fig, (a, b) = frame("Chronos-2 历史长度对照", "每档误差与该档可评分窗口上的原版 Ridge 成对比较。",
                        1, 2, (14, 5.5))
    keys = ("context_3", "context_10", "context_20", "selected")
    labels = ["90秒", "5分钟", "10分钟", "历史选择"]
    pos = np.arange(len(keys))
    for shift, key, color, name in [(-.17, "candidate_metrics", TEAL, "Chronos-2"),
                                    (.17, "reference_metrics", BLUE, "同窗原版 Ridge")]:
        vals = [context[k][key]["date_mae"] for k in keys]
        bars = a.bar(pos + shift, vals, width=.33, color=color, label=name)
        a.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    a.set_xticks(pos, labels)
    a.set_ylim(0, max(context[k]["candidate_metrics"]["date_mae"] for k in keys) * 1.29)
    a.legend(frameon=False)
    panel(a, "A  提前 120 秒的误差", "日期等权 MAE（次/分）")
    bars = b.barh(labels[::-1], [context[k]["candidate_metrics"]["n"] for k in keys][::-1], color=BLUE)
    b.bar_label(bars, padding=4)
    b.set_xlim(0, max(context[k]["reference_metrics"]["n"] for k in keys) * 1.2)
    panel(b, "B  可评分窗口", grid="x")
    b.set_xlabel("窗口数")
    atlas.add(fig, "第二版实验", "Chronos-2 对照与覆盖", "各历史长度的同样本误差与覆盖。开发门槛未达到，未进行 LoRA。")

    groups = sorted({(r["date"], r["session"], r["phase"]) for r in rows})
    fig, axes = frame("Chronos-2 逐场预测与观测", "各曲线按目标时间显示。较长上下文缺失的区段留空。",
                      (len(groups) + 3) // 4, 4, (16, 9.5), sharey=True)
    axs = list(axes)
    style = [("ridge_v1", BLUE, "-", "原版 Ridge"), ("context_3", GRAY, ":", "90秒"),
             ("context_10", TEAL, "--", "5分钟"), ("context_20", AMBER, "-.", "10分钟")]
    for ax, (day, session, phase) in zip(axs, groups):
        by_time = {int(r["target_s"]): r for r in rows if r["session"] == session}
        seconds = range(min(by_time), max(by_time) + 1, 30)
        rr = [by_time.get(t, {}) for t in seconds]
        x = np.array(seconds)/60
        ax.plot(x, numbers(rr, "observed"), color=INK, lw=2)
        for key, color, ls, _ in style:
            ax.plot(x, numbers(rr, key), color=color, linestyle=ls, linewidth=1.45)
        panel(ax, f"{session}  {day[5:]}  {'开发' if phase=='development' else '评估'}")
        ax.set_xlabel("运动分钟")
    for ax in axs[len(groups):]:
        ax.set_visible(False)
    fig.supylabel("心率（次/分）", fontsize=10)
    fig.legend(handles=[Line2D([0], [0], color=INK, lw=2, label="设备实测")] +
               [Line2D([0], [0], color=c, linestyle=ls, label=n) for _, c, ls, n in style],
               loc="outside lower center", ncol=5, frameon=False)
    atlas.add(fig, "第二版实验", "Chronos-2 逐场曲线", "开发日期与评估日期分别标明。")


def rest_page(atlas, rows):
    fig, (a, b, c, d) = frame("次日设备静息心率回测", "前日活动模型、两条历史基线和实际测量逐日比较。",
                              2, 2, (14, 8))
    x = [date.fromisoformat(r["date"]) for r in rows]
    y = numbers(rows, "observed")
    keys = ("historical", "last", "minutes", "steps")
    a.plot(x, y, color=INK, marker="o", lw=2, ms=4)
    for k in keys:
        line(a, x, numbers(rows, k), k)
        line(b, x, np.abs(numbers(rows, k) - y), k)
    panel(a, "A  实测与预测", "次/分")
    panel(b, "B  日期绝对误差", "次/分")
    dates(a)
    dates(b)
    for ax, field, title, unit in [(c, "input_minutes", "C  前日运动时间", "分钟"),
                                    (d, "input_steps", "D  前日步数", "步")]:
        ax.scatter(numbers(rows, field), y, color=BLUE, s=32)
        panel(ax, title, "次日静息心率（次/分）")
        ax.set_xlabel(unit)
    fig_legend(fig, keys)
    atlas.add(fig, "第二版实验", "静息心率逐日回测", f"评估 {len(rows)} 天。活动散点用于观察，图中没有拟合因果关系。")


def sleep_pairs(atlas, rows):
    fig, (a, b, c) = frame("主睡眠与后续观测的配对", "一个点为一晚主睡眠。缺测终点留空。",
                           1, 3, (15, 5.1))
    for ax, key, title, unit in [(a, "next_rest", "A  次日静息心率", "次/分"),
                                   (b, "afternoon_pressure", "B  同日午后压力", "设备分数")]:
        g = [r for r in rows if np.isfinite(number(r, key))]
        ax.scatter(numbers(g, "minutes"), numbers(g, key), color=BLUE, s=37)
        panel(ax, f"{title} · {len(g)} 晚", unit)
        ax.set_xlabel("主睡眠时长（分钟）")
    days = [r["date"][5:] for r in rows]
    hour = numbers(rows, "pressure_hours")
    c.bar(days, hour, color=[TEAL if h >= 3 else AMBER for h in hour])
    c.axhline(3, ls="--", lw=1, color=GRAY)
    c.tick_params(axis="x", labelrotation=48)
    panel(c, "C  午后压力有效小时数", "小时")
    atlas.add(fig, "第二版实验", "睡眠与静息心率及压力", "显示两种配对终点，以及下午压力的可用小时。")


def gallery(atlas, workbook, run, catalog):
    nrows = sum(s["rows"] for s in json.loads(PROFILE.read_text(encoding="utf-8"))["sheets"])
    charts = []
    for group in ("原始观测", "第二版实验"):
        charts.append(f"<h2>{group}</h2>")
        for item in (i for i in atlas.items if i["group"] == group):
            src = quote(item["png"])
            title = escape(item["title"])
            charts.append(f'<article><h3>{title}</h3><p>{escape(item["note"])}</p>'
                          f'<a href="{src}"><img src="{src}" alt="{title}" loading="lazy"></a></article>')
    prior = ROOT / "runs/20261109-110040-088694/可视化/index.html"
    prior_html = f'<a href="{prior.as_uri()}">旧版与早期实验图册</a>' if prior.exists() else ""
    raw_links = " · ".join(f'<a href="{(run / filename).as_uri()}">{escape(filename)}</a>'
                           for filename in ["metrics.json", "swim_heart.csv", "swim_energy_total.csv",
                                            "swim_energy_active.csv", "chronos_120s.csv",
                                            "resting_hr.csv", "sleep_exploration.csv"])
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>个人活动数字孪生 · 全量数据图册</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;color:#172e3e;background:#eff4f4;font:16px/1.6 'Microsoft YaHei',sans-serif}}
header{{background:#173a40;color:#fff;padding:30px max(18px,calc((100vw - 1140px)/2))}}
header h1{{margin:0;font-size:clamp(23px,5vw,37px)}} main{{max-width:1180px;margin:auto;padding:18px}}
a{{color:#00628a}} .resources,article{{background:#fff;padding:17px 20px;border-radius:8px;margin:12px 0;border:1px solid #d7e4e5}}
article img{{width:100%;height:auto;display:block}}h3{{margin:0}} article p{{color:#4c6470;margin:5px 0 10px}}
</style></head><body><header><h1>个人活动数字孪生 · 全量数据图册</h1>
<p>{nrows:,} 行原始记录 · {len(catalog)} 个非身份字段 · {len(atlas.items)} 张组图</p></header><main>
<div class="resources"><a href="{quote('全量数据图册.pdf')}">完整 PDF</a> ·
<a href="{quote('字段覆盖.csv')}">字段覆盖清单</a> · <a href="{(run / 'index.html').as_uri()}">第二版离线回放</a> · {prior_html}
<p>原始表以实测记录为准。灰色格表示缺测，恒定字段标注为恒定或全零。A–E 身份与基础信息列不展开。</p>
<p>数据来源：<a href="{workbook.resolve().as_uri()}">本地工作簿</a> · {raw_links}</p></div>
{''.join(charts)}</main></body></html>"""
    (atlas.out / "index.html").write_text(body, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "runs/v2-20261109-170819-386454")
    p.add_argument("--workbook", type=Path, default=ROOT / "data/健康数据.xlsx")
    args = p.parse_args()
    run, workbook = args.run.resolve(), args.workbook.resolve()
    out = run / "全量数据图册"
    out.mkdir(exist_ok=True)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    payload = workbook.read_bytes()
    digest = sha256(payload).hexdigest()
    assert digest == report["info"]["source_sha256"] == profile["sha256"], "Source workbook mismatch"
    data = swim.read_workbook(payload)
    sessions, _ = swim.build_swims(data)
    start = min(date.fromisoformat(row[TIME[s]][:10]) for s, rows in data.items() if s in TIME for row in rows)
    end = max(date.fromisoformat(row[TIME[s]][:10]) for s, rows in data.items() if s in TIME for row in rows)
    with PdfPages(out / "全量数据图册.pdf", metadata={"Title": "个人活动数字孪生 · 全量数据图册"}) as pdf:
        atlas = Atlas(out, pdf)
        overview(atlas, data, start, end)
        catalog = sheet_pages(atlas, data, profile)
        sleep_stages(atlas, data)
        swim_source(atlas, sessions)
        print(f"Source figures: {len(atlas.items)}", flush=True)
        heart = read_csv(run / "swim_heart.csv")
        heat = read_csv(run / "swim_energy_total.csv")
        active = read_csv(run / "swim_energy_active.csv")
        v2_overview(atlas, report)
        heart_replay(atlas, heart, sessions, 30)
        heart_replay(atlas, heart, sessions, 120)
        heart_errors(atlas, heart, report)
        heat_page(atlas, heat, report["heat_total"], False)
        heat_page(atlas, active, report["heat_active"], True)
        duration_sensitivity(atlas, heat, active, report)
        if report["chronos"]["status"] == "completed":
            chronos_pages(atlas, read_csv(run / "chronos_120s.csv"), report)
        rest_page(atlas, read_csv(run / "resting_hr.csv"))
        sleep_pairs(atlas, read_csv(run / "sleep_exploration.csv"))
    gallery(atlas, workbook, run, catalog)
    (out / "图册索引.json").write_text(json.dumps({"source_sha256": digest, "figures": atlas.items,
              "rows": sum(map(len, data.values())), "fields": len(catalog)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(atlas.items)} figures at {out}", flush=True)


if __name__ == "__main__":
    main()
