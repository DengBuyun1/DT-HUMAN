"""Draw fixed-split coverage and predictions from completed experiment records."""
from datetime import datetime, timedelta
from html import escape
from urllib.parse import quote

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from plot_full_atlas import AMBER, BLUE, GRAY, INK, TEAL, dates, frame, panel


MODEL_COLORS = {"historical": ("训练段中位数", GRAY, ":"),
                "last": ("上一个实测", "#795797", "--"),
                "ridge_one_step": ("固定 Ridge · 逐步", BLUE, "-"),
                "ridge_open_loop": ("固定 Ridge · 一次性推演", AMBER, "--"),
                "persistence": ("最近心率", GRAY, ":"),
                "ridge_v1": ("原版 Ridge", BLUE, "-"),
                "ridge_weighted": ("加权 Ridge", TEAL, "--"),
                "kalman": ("Kalman", AMBER, "-."),
                "rate": ("每分钟换算", BLUE, "-"), "line": ("约束直线", AMBER, "--")}


def vals(rows, key):
    return [r[key] if r.get(key) is not None else np.nan for r in rows]


def lines(ax, x, rows, names):
    for key in names:
        if not any(r.get(key) is not None for r in rows):
            continue
        label, color, style = MODEL_COLORS[key]
        ax.plot(x, vals(rows, key), color=color, ls=style, lw=1.45, label=label, marker=".", ms=3)


def legend(fig, names):
    handles = [Line2D([0], [0], color=INK, lw=2, label="实测")]
    handles += [Line2D([0], [0], color=MODEL_COLORS[k][1], ls=MODEL_COLORS[k][2], lw=1.6,
                       label=MODEL_COLORS[k][0]) for k in names]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)


def render(out, report, daily, hdata, records, swim_counts, heart, heat):
    figures = []

    def save(fig, title):
        name = f"{len(figures)+1:02}_{title}.png"
        fig.savefig(out / name, dpi=155, facecolor="white")
        plt.close(fig)
        figures.append((title, name))

    cmap = ListedColormap(["#e9edf0", "#8bb9d3", TEAL])
    start = datetime.fromisoformat(report["start"])
    n = report["calendar_days"]
    ticks = [start + timedelta(days=i) for i in range(n)]
    info = report["daily_hourly"]
    ditems = [item for item in info if item["resolution"] == "day"]
    fig, (ax,) = frame("日级记录连续性", "深绿色为最长连续段，浅蓝色为该指标的其他有记录日期。", 1, 1, (14, 9))
    matrix = []
    for item in ditems:
        series = daily[item["metric"]]["points"]
        lo, hi = item["longest_start"], item["longest_end"]
        matrix.append([2 if t in series and lo <= t.isoformat() <= hi else int(t in series) for t in ticks])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(range(len(ditems)), [f"{item['metric']} · {item['longest_slots']}天" for item in ditems])
    ax.set_xticks(range(0, n, 2), [t.strftime("%m/%d") for t in ticks[::2]])
    ax.axvline(n//2-.5, color=AMBER, lw=1.5, ls="--")
    ax.set_xlabel("日期  ·  虚线为完整31日序列的训练／测试分界")
    save(fig, "日级连续性与分界")

    fig, axes = frame("小时级采样覆盖", "每小时至少 3 条无冲突读数。深绿色标出每项指标各自最长的连续段。",
                      len(hdata), 1, (14, 10))
    for ax, item in zip(axes, (i for i in info if i["resolution"] == "hour")):
        raw = hdata[item["metric"]]["points"]
        lo, hi = item["longest_start"], item["longest_end"]
        grid = np.zeros((24, n), dtype=int)
        for j, day in enumerate(ticks):
            for hr in range(24):
                t = day + timedelta(hours=hr)
                if raw.get(t, (None, 0))[1] >= 3:
                    grid[hr, j] = 2 if lo <= t.isoformat() <= hi else 1
        ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2)
        ax.set_xticks(range(0, n, 2), [t.strftime("%m/%d") for t in ticks[::2]])
        ax.set_yticks(range(0, 24, 4))
        panel(ax, f"{item['metric']} · {item['observed_slots']} 个合格小时 · 最长 {item['longest_slots']} 小时",
              "当天小时", grid=None)
    save(fig, "小时采样覆盖与最长段")

    by_metric = {(r["resolution"], r["metric"]) for r in records}
    groups = [[i for i in info if i["resolution"] == "day" and i["longest_slots"] == n],
              [i for i in info if (i["resolution"], i["metric"]) in by_metric
               and (i["resolution"] == "hour" or i["longest_slots"] < n)]]
    model_keys = ("historical", "last", "ridge_one_step", "ridge_open_loop")
    for items, title in zip(groups, ("31日完整序列训练与测试", "短连续段与小时序列训练与测试")):
        fig, axes = frame(title, "橙色虚线右侧为测试段。逐步预测使用上一时点已到达的实测，一次性推演只使用前半段。",
                          (len(items)+1)//2, 2, (14, 3.1*((len(items)+1)//2)+1.2))
        axes = list(axes)
        for ax, item in zip(axes, items):
            rr = [r for r in records if (r["metric"], r["resolution"]) == (item["metric"], item["resolution"])]
            x = [datetime.fromisoformat(r["slot"]) for r in rr]
            mid = item["test"]["train_slots"]
            boundary = x[mid-1] + (x[mid]-x[mid-1])/2
            ax.axvspan(x[0], boundary, color="#e8f2f5", alpha=.65)
            ax.axvline(boundary, color=AMBER, lw=1.25, ls="--")
            ax.plot(x, vals(rr, "observed"), color=INK, lw=1.9, marker="o", ms=3.2)
            lines(ax, x, rr, model_keys)
            if item["resolution"] == "hour":
                from matplotlib import dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H时"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
            else:
                dates(ax)
            unit = "小时" if item["resolution"] == "hour" else "天"
            panel(ax, f"{item['metric']}  ·  {mid} + {len(x)-mid} {unit}", item["unit"])
        for ax in axes[len(items):]:
            ax.set_visible(False)
        legend(fig, model_keys)
        save(fig, title)

    fig, (ax,) = frame("游泳场内连续性与日期半分", "最长连续段按实际有有效中位数的 30 秒窗口计算。",
                       1, 1, (14, 5))
    names = [r["session"] for r in swim_counts]
    ax.bar(names, [r["total_windows"]/2 for r in swim_counts], color="#e4ebee", label="全场窗口时长")
    ax.bar(names, [r["longest_continuous_minutes"] for r in swim_counts], width=.62,
           color=[BLUE if r["split"] == "train" else TEAL for r in swim_counts])
    ntrain = report["swim"]["train_swims"]
    ax.axvline(ntrain-.5, color=AMBER, lw=1.3, ls="--")
    panel(ax, f"{ntrain} 场训练 · {report['swim']['test_swims']} 场测试", "分钟")
    ax.legend(handles=[Line2D([0], [0], lw=6, color="#e4ebee", label="全场窗口时长"),
                       Line2D([0], [0], lw=6, color=BLUE, label="训练场最长连续段"),
                       Line2D([0], [0], lw=6, color=TEAL, label="测试场最长连续段")],
              frameon=False, ncol=3)
    save(fig, "游泳场次连续性")

    keys = ("persistence", "ridge_v1", "ridge_weighted", "kalman")
    for h in (30, 120):
        title = f"游泳心率固定半分_{h}秒"
        test = [r for r in swim_counts if r["split"] == "test"]
        fig, axes = frame(title.replace("_"," · "), "只显示从输入到目标全程连续的评分窗。其余候选窗保存在逐窗表。",
                          (len(test)+3)//4, 4, (16, 9), sharey=True)
        axes = list(axes)
        for ax, session in zip(axes, test):
            rr = [r for r in heart if r["session"] == session["session"] and r["horizon"] == h]
            by_time = {r["target_min"]: r for r in rr if r["continuous"]}
            xmin = min((r["target_min"] for r in rr), default=0.)
            xmax = max((r["target_min"] for r in rr), default=xmin)
            x = np.arange(xmin, xmax+.25, .5)
            scored = [by_time.get(t, {}) for t in x]
            ax.plot(x, vals(scored, "observed"), color=INK, lw=2.1, marker=".", ms=4)
            lines(ax, x, scored, keys)
            panel(ax, f"{session['session']} · {session['date'][5:]} · {len(by_time)}窗")
            ax.set_xlabel("目标分钟")
        for ax in axes[len(test):]:
            ax.set_visible(False)
        fig.supylabel("心率（次/分）")
        legend(fig, keys)
        save(fig, title)

    fig, axes = frame("游泳热量固定半分对照", "场次事件按日历分半。实际总时长与活动时间分别作为条件。",
                      1, 2, (14, 5.1))
    keys = ("historical", "rate", "line")
    for ax, label in zip(axes, report["heat"]):
        rr = [r for r in heat if r["duration_kind"] == label]
        names = [r["id"] for r in rr]
        ax.plot(names, vals(rr, "observed"), color=INK, lw=1.9, marker="o", ms=4)
        lines(ax, names, rr, keys)
        panel(ax, f"{label} · {len(rr)} 场测试", "设备热量（千卡）")
    legend(fig, keys)
    save(fig, "游泳热量固定半分")

    table = "".join(f'<article><h2>{escape(title)}</h2><a href="{quote(name)}">'
                    f'<img src="{quote(name)}" alt="{escape(title)}" loading="lazy"></a></article>'
                    for title, name in figures)
    links = " · ".join(f'<a href="{quote(name)}">{escape(name)}</a>' for name in
                      ("metrics.json", "连续性概览.csv", "逐时间点预测.csv",
                       "逐游泳场次连续性.csv", "游泳心率预测.csv", "热量估算.csv"))
    (out / "index.html").write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>连续数据前后半段实验</title>
<style>body{{margin:0;background:#eef3f4;color:#18333e;font:16px/1.6 'Microsoft YaHei',sans-serif}}
main{{max-width:1250px;margin:auto;padding:18px}} article{{background:#fff;padding:16px;margin:14px 0;border-radius:7px}}
article img{{display:block;width:100%;height:auto}}a{{color:#00679b}}h2{{font-size:18px}}</style>
<main><h1>连续数据前后半段实验</h1><p>每个连续段以前半训练、后半固定测试。所有图均保存在本地。</p>
<p>{links}</p>{table}</main></html>""", encoding="utf-8")
