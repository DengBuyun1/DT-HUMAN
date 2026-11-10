"""Offline V3 figures from the same saved targets and prediction rows."""
from collections import defaultdict
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from plot_full_atlas import AMBER,BLUE,GRAY,INK,TEAL
from v3.aggregate import DAY,HOUR

COLORS = dict(zip(("D-B0","D-B1","D-B2","D-B3","D-R0","HR-B0","HR-B1","HR-R0","HR-R1",
                   "E-B0","E-R0","E-R1","E-R2"),
                  (GRAY,TEAL,AMBER,"#7858a7",BLUE,GRAY,AMBER,BLUE,TEAL,GRAY,BLUE,AMBER,TEAL)))


def chart(ax,title,unit=None):
    ax.set_title(title,loc="left",fontsize=10.5,fontweight="bold")
    if unit:
        ax.set_ylabel(unit)
    ax.grid(alpha=.24)
    ax.spines[["top","right"]].set_visible(False)


def calendar_axis(ax,hour=False):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %Hh" if hour else "%m/%d"))
    ax.tick_params(axis="x",labelrotation=18 if hour else 0)


def render(out,days,cutoff,targets,cov,diag,predictions,result,cal):
    pages=[]
    with PdfPages(out/"V3.0_实验图册.pdf") as pdf:
        def save(fig,name,title,note):
            fig.suptitle(title+"\n"+note,x=.04,ha="left",fontsize=13,fontweight="bold")
            path=name+".png"
            fig.savefig(out/path,dpi=140,facecolor="white")
            pdf.savefig(fig,facecolor="white")
            plt.close(fig)
            pages.append((path,title,note))
        fig,axes=plt.subplots(3,1,figsize=(12,8),layout="constrained")
        for ax,metric in zip(axes,("实时心率","血氧","压力")):
            hour=[r for r in cov if r["resolution"]=="hour" and r["metric"]==metric]
            matrix=np.array([r["valid_5min_bins"] for r in hour]).reshape(len(days),24).T
            heatmap=ax.imshow(matrix,cmap="Blues",aspect="auto",origin="lower",vmin=0,vmax=12)
            ax.set_xticks(range(0,len(days),3),[d.strftime("%m/%d") for d in days[::3]])
            ax.set_yticks((0,6,12,18,23))
            ax.axvline(days.index(cutoff)-.5,color=AMBER,lw=1.5)
            chart(ax,metric+" · 每小时有效 5 分钟格数","小时")
        fig.colorbar(heatmap,ax=list(axes),shrink=.8,label="有效 5 分钟格数")
        save(fig,"01_覆盖热图","时段覆盖","橙线分隔日历前后半段。小时至少 3 个有效格才合格。")

        fig,axes=plt.subplots(3,1,figsize=(12,8.5),layout="constrained")
        for ax,metric,unit in zip(axes,("实时心率","血氧","压力"),("次/分","%","分")):
            rows=[r for r in diag if r["metric"]==metric]
            ax.plot(days,[r["old_raw_median"] for r in rows],".-",color=GRAY,label="旧版原始读数中位数")
            ax.plot(days,[r["new_hour_balanced"] for r in rows],"o-",ms=3,color=BLUE,label="新版观测小时等权摘要")
            ax.scatter([d for d,r in zip(days,rows) if r["strict_eligible"]],
                       [r["new_hour_balanced"] for r in rows if r["strict_eligible"]],s=13,color=TEAL,label="12 小时敏感性")
            ax.axvline(cutoff,color=AMBER,alpha=.8)
            chart(ax,metric,unit);calendar_axis(ax);ax.legend(loc="best",fontsize=8,frameon=False)
        save(fig,"02_聚合口径","新旧日目标比较","缺测保持空白。新旧目标差异反映统计口径变化。")

        fig,axes=plt.subplots(2,1,figsize=(12,5.8),layout="constrained")
        hr=[r for r in diag if r["metric"]=="实时心率"]
        axes[0].bar(days,[r["swim_fraction"] or 0 for r in hr],color=BLUE)
        axes[0].set_ylim(0,1)
        chart(axes[0],"游泳时段读数占当日实时心率记录的比例","占比")
        for tid,label,_,_,points,_ in targets:
            if tid in ("day:小睡总时长","day:单次小睡时长中位数"):
                axes[1].scatter(list(points),[p.value for p in points.values()],label=label,s=27,alpha=.85)
        chart(axes[1],"小睡总量与单次中位数","分钟")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.axvline(cutoff,color=AMBER)
            calendar_axis(ax)
        save(fig,"02b_采样比例与小睡","采样密度和睡眠口径","每个点对应有观测的日期。小睡没有记录时保留缺失。")

        records=defaultdict(list)
        for row in predictions:
            records[row["protocol_id"],row["target_id"],row["information_mode"]].append(row)
        day_targets=[t for t in targets if t[3]=="day" and t[-1]=="model"]
        for page,i in enumerate(range(0,len(day_targets),4),1):
            ts=day_targets[i:i+4]
            fig,axes=plt.subplots(len(ts),2,figsize=(14,2.5*len(ts)+1.3),squeeze=False,layout="constrained")
            for pair,(tid,label,unit,_,points,_) in zip(axes,ts):
                for ax,mode in zip(pair,("online_one_step","open_loop")):
                    rows=records["V3-F50",tid,mode]
                    ax.plot(days,[points[d].value if d in points else np.nan for d in days],"o-",ms=3,color=INK,label="实测")
                    ax.axvspan(days[0],cutoff,alpha=.07,color=TEAL)
                    ax.axvline(cutoff,color=AMBER)
                    for model in ("D-B0","D-B1","D-B2","D-B3","D-R0"):
                        rr=[r for r in rows if r["model_id"]==model]
                        if rr and any(r["prediction"] is not None for r in rr):
                            ax.plot([r["target_start"] for r in rr],[r["prediction"] for r in rr],".-",
                                    lw=1.2,color=COLORS[model],label=model,alpha=.86)
                    chart(ax,label+" · "+("在线一步" if mode=="online_one_step" else "开环"),unit)
                    calendar_axis(ax)
                    ax.legend(loc="best",ncol=3,fontsize=7,frameon=False)
            save(fig,f"03_{page}_日级固定半段","日级固定前后半段","左列接收前一日观测，右列仅从训练末端推演。")

        fig,axes=plt.subplots(3,1,figsize=(13,8),layout="constrained")
        for ax,(tid,label,unit,_,points,_) in zip(axes,[t for t in targets if t[3]=="hour"]):
            rows=records["V3-C50",tid,"online_one_step"]
            ticks=sorted({r["target_start"] for r in rows})
            if ticks:
                start=ticks[0]-len(ticks)*HOUR
                chosen=[d for d in sorted(points) if start<=d<=ticks[-1]]
                grid=[chosen[0]+i*HOUR for i in range(int((ticks[-1]-chosen[0])/HOUR)+1)]
                ax.plot(grid,[points[d].value if d in points else np.nan for d in grid],".-",color=INK,label="实测")
                ax.axvline(ticks[0],color=AMBER)
                for key in ("D-B0","D-B1","D-B2","D-R0"):
                    rr=[r for r in rows if r["model_id"]==key]
                    if any(r["prediction"] is not None for r in rr):
                        ax.plot([r["target_start"] for r in rr],[r["prediction"] for r in rr],".-",color=COLORS[key],label=key)
                calendar_axis(ax,True);ax.legend(loc="best",frameon=False,ncol=4,fontsize=8)
            else:
                ax.text(.03,.5,"没有符合规则的连续段",transform=ax.transAxes)
            chart(ax,label,unit)
        save(fig,"04_小时最长连续段","小时级连续段半分","按各指标自己的最长合格段划分，横坐标为北京时间。")

        for horizon in (30,120):
            rows=records["V3-F50",f"swim_hr_{horizon}","online_one_step"]
            keys=sorted({r["session"] for r in rows})
            fig,axes=plt.subplots((len(keys)+1)//2,2,figsize=(14,2.25*((len(keys)+1)//2)+1.5),
                                  squeeze=False,layout="constrained")
            for ax,key in zip(axes.flat,keys):
                session=[r for r in rows if r["session"]==key and r["in_record"]]
                base=[r for r in session if r["model_id"]=="HR-B0"]
                first=min(r["origin"] for r in base)
                def minutes(rr):
                    return [(r["target_end"]-first).total_seconds()/60 for r in rr]
                ax.plot(minutes(base),[r["observed"] for r in base],".-",color=INK,label="实测")
                for model in ("HR-B0","HR-R0","HR-R1"):
                    rr=[r for r in session if r["model_id"]==model]
                    ax.plot(minutes(rr),[r["prediction"] for r in rr],"-",lw=1,color=COLORS[model],label=model)
                chart(ax,key+" · "+base[0]["date"],"次/分")
                ax.set_xlabel("距该场首个候选发出时点的分钟")
                ax.legend(loc="best",frameon=False,fontsize=7,ncol=4)
            for ax in list(axes.flat)[len(keys):]:
                ax.axis("off")
            save(fig,f"05_游泳心率_{horizon}秒",f"游泳心率 · {horizon} 秒 · 全部测试场次","固定参数预测。观测或输入缺失处断开。")

        fixed_heat=records["V3-F50","swim_heat_total","conditional_duration"]
        fig,axes=plt.subplots(1,2,figsize=(14,5.3),layout="constrained")
        base=[r for r in fixed_heat if r["model_id"]=="E-R0"]
        support=base[0]["support_min"],base[0]["support_max"]
        axes[0].axvspan(*support,color=TEAL,alpha=.15,label="训练时长范围")
        axes[0].scatter([r["minutes"] for r in base],[r["observed"] for r in base],c=INK,label="设备热量",zorder=5)
        for key in ("E-B0","E-R0","E-R1","E-R2"):
            rr=sorted([r for r in fixed_heat if r["model_id"]==key],key=lambda r:r["minutes"])
            axes[0].plot([r["minutes"] for r in rr],[r["prediction"] for r in rr],color=COLORS[key],label=key)
        for i,key in enumerate(("E-R0","E-R1","E-R2")):
            rr=[r for r in fixed_heat if r["model_id"]==key]
            axes[1].bar(np.arange(len(rr))+(i-1)*.24,[r["prediction"]-r["observed"] for r in rr],
                        width=.23,color=COLORS[key],label=key)
        axes[1].set_xticks(range(len(base)),[r["session"]+"\n"+f'{r["minutes"]:g}m' for r in base],fontsize=8)
        axes[1].axhline(0,color=INK,lw=1)
        for ax,title in zip(axes,("总时长和热量","逐场误差 · 正值为高估")):
            chart(ax,title,"千卡");ax.legend(frameon=False,fontsize=8)
        axes[0].set_xlabel("实际总时长（分钟）")
        save(fig,"06_热量支持范围","游泳热量条件估算","按实际总时长回测。绿色区间为训练时长支持范围。")

        fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
        lookup={(r["protocol_id"],r["target_id"],r["model_id"]):r for r in result["results"] if r["cohort"]=="all"}
        for horizon,ax in zip((30,120),axes[0]):
            models=("HR-B0","HR-B1","HR-R0","HR-R1")
            for i,proto in enumerate(("V3-F50","V3-ROLL")):
                rr=[lookup[proto,f"swim_hr_{horizon}",k] for k in models]
                ax.bar(np.arange(len(models))+(i-.5)*.34,[r["scores"]["date_mae"] for r in rr],
                       width=.33,color=(BLUE,TEAL)[i],label=proto)
            ax.set_xticks(range(len(models)),models)
            chart(ax,f"{horizon} 秒 · 固定 vs 滚动","日期均衡 MAE")
            ax.legend(frameon=False)
        ax=axes[1,0]
        for j,h in enumerate((30,120)):
            m=lookup["V3-F50",f"swim_hr_{h}","HR-R0"]
            ax.bar(np.arange(3)+(j-.5)*.33,[m["candidates"],m["issued"],m["n_scored"]],
                   width=.32,label=f"{h} 秒",color=(BLUE,TEAL)[j])
        ax.set_xticks(range(3),["记录内候选","已发出","可评分"])
        chart(ax,"固定半段 · 发出与评分","窗口数");ax.legend(frameon=False)
        ax=axes[1,1]
        bands=[b for b in result["bands"] if b["model_id"]=="HR-R0"]
        for i,b in enumerate(bands):
            ax.bar(i-.16,b["enabled"]/b["issued"],.3,color=BLUE,label="启用比例" if i==0 else None)
            if b["scored"]:
                ax.bar(i+.16,b["hits"]/b["scored"],.3,color=TEAL,label="启用后覆盖" if i==0 else None)
        ax.set_xticks(range(len(bands)),[b["target_id"].split("_")[-1]+" 秒" for b in bands])
        ax.set_ylim(0,1)
        chart(ax,"固定点预测的在线经验误差带","比例");ax.legend(frameon=False)
        save(fig,"07_稳定性与覆盖","稳定性、覆盖与经验误差带","点预测和在线校准独立登记。经验覆盖依赖本次历史日期。")

        fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
        ax=axes[0]
        decisions=[d for d in result["decisions"] if d["protocol_id"]=="V3-F50" and d["target_id"]=="swim_hr_120"]
        for gate in decisions[0]["comparisons"]:
            dropped=gate.get("leave_one_date_out",{})
            ax.plot(list(dropped),list(dropped.values()),".-",label=gate["candidate"]+" 对 "+gate["reference"])
        ax.axhline(0,color=GRAY);ax.tick_params(axis="x",labelrotation=25,labelsize=8)
        chart(ax,"120 秒预测 · 每次去掉一个日期","其余日期平均 MAE 改善")
        ax.legend(frameon=False,fontsize=8)
        ax=axes[1]
        band=[r for r in cal if r["target_id"]=="swim_hr_120" and r["model_id"]=="HR-R0" and r["lower"] is not None]
        if band:
            day=band[0]["date"]
            selected=[r for r in band if r["date"]==day]
            ax.fill_between(range(len(selected)),[r["lower"] for r in selected],[r["upper"] for r in selected],
                            color=BLUE,alpha=.15,label="经验误差带")
            ax.plot([r["prediction"] for r in selected],color=BLUE,label="点预测")
            ax.plot([r["observed"] for r in selected],".",color=INK,ms=3,label="实测")
            ax.legend(frameon=False,fontsize=8)
        else:
            ax.text(.03,.5,"既往可评分日期不足",transform=ax.transAxes)
        chart(ax,"首个启用日的经验误差参考","次/分")
        save(fig,"08_删日与经验区间","日期敏感性与在线区间","历史误差带至少需要 5 个已完成的前序评分日期。")
        observed=[t for t in targets if t[-1]=="observed_only"]
        fig,axes=plt.subplots((len(observed)+2)//3,3,figsize=(14,10),layout="constrained")
        for ax,(_,label,unit,_,points,_) in zip(axes.flat,observed):
            ax.scatter(list(points),[p.value for p in points.values()],color=BLUE,s=22)
            chart(ax,label+" · "+str(len(points))+" 天",unit)
            calendar_axis(ax)
        for ax in list(axes.flat)[len(observed):]:
            ax.axis("off")
        save(fig,"09_体重和体成分","体重与体成分","按日显示已有测量摘要，未测日期留空。")
    decision_rows="".join("<tr>"+"".join(f"<td>{escape(str(r.get(k) or '—'))}</td>" for k in
                          ("protocol_id","target_id","information_mode","status","selected"))+"</tr>" for r in result["decisions"])
    images="".join(f'<figure><h2>{escape(title)}</h2><a href="{path}"><img src="{path}" alt="{escape(title)}" loading="lazy"></a><figcaption>{escape(note)}</figcaption></figure>'
                   for path,title,note in pages)
    links=" · ".join(f'<a href="{name}">{label}</a>' for name,label in
                    (("V3.0_实验图册.pdf","PDF 图册"),("results_analysis.md","结果分析"),("metrics.json","完整指标"),
                     ("decision_table.csv","模块决定"),("predictions.csv","逐条预测"),("verification_report.md","验证报告")))
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3.0 实验图册</title><style>body{{margin:0;background:#eef3f5;color:#19333e;font:16px/1.55 'Microsoft YaHei',sans-serif}}
header,main{{max-width:1220px;margin:auto;padding:20px}}header{{background:#113c4b;color:white}}
a{{color:#086986}}header a{{color:#9be3ee}}figure,section{{background:white;margin:18px 0;padding:16px;border-radius:10px}}
img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}td,th{{padding:7px;text-align:left;border-bottom:1px solid #dbe6eb}}
.scroll{{overflow:auto;max-height:520px}}</style>
<header><h1>V3.0 数据口径修正与受控预测实验</h1><p>单人回顾数据 · 固定半段、连续段、滚动与在线误差参考分开报告</p><p>{links}</p></header>
<main><section><h2>模块决定</h2><div class="scroll"><table><thead><tr><th>协议</th><th>目标</th><th>信息条件</th><th>状态</th><th>采用或回退</th></tr></thead>
<tbody>{decision_rows}</tbody></table></div></section>{images}</main></html>"""
    (out/"index.html").write_text(html,encoding="utf-8")


def write_reports(out,result):
    scores=result["results"]
    def find(proto,tid,model,cohort="all"):
        return next(r for r in scores if r["protocol_id"]==proto and r["target_id"]==tid and
                    r["model_id"]==model and r["cohort"]==cohort)
    def adopted(tid):
        r=next(x for x in result["decisions"] if x["protocol_id"]=="V3-F50" and x["target_id"]==tid)
        return r["selected"],r["status"]
    lines=["# V3.0 实验结果","", "所有结果来自同一人的既有设备记录，按冻结方案回顾复测。日期和场次用于加权汇总。","",
           "## 观测质量","",f"- 原始数据 {result['source_rows']:,} 行，{result['calendar_days']} 天。",
           f"- 3 个传感器同小时合格 {result['quality']['hourly_joint_count']} 小时，最长共同连续段 {result['quality']['hourly_joint_longest']} 小时。",
           "- 12 小时规则、最长连续段与固定日历结果分别列在 coverage、continuity 和 metrics 文件。","",
           "## 游泳心率固定半段","","| 提前量 | 模型 | 日期均衡 MAE | P90 | 发出 / 记录内候选 | 可评分 |","|---|---|---:|---:|---:|---:|"]
    for h in (30,120):
        for model in ("HR-B0","HR-B1","HR-R0","HR-R1"):
            r=find("V3-F50",f"swim_hr_{h}",model)
            m=r["scores"]
            lines.append(f"| {h} 秒 | {model} | {m['date_mae']:.3f} | {m['p90_abs']:.3f} | {r['issued']} / {r['candidates']} | {r['n_scored']} |")
    lines += ["",*(f"- {h} 秒采用状态为 {adopted(f'swim_hr_{h}')[1]}，采用或回退至 {adopted(f'swim_hr_{h}')[0]}。" for h in (30,120)),
              "- 日期、场次、严格连续、误差分层和删日结果可在 CSV 和指标 JSON 中复核。","",
              "## 游泳热量固定半段","","| 模型 | 全部测试日期均衡 MAE | 逐场 MAE | 支持范围内可评分场次 |","|---|---:|---:|---:|"]
    for key in ("E-B0","E-R0","E-R1","E-R2"):
        a=find("V3-F50","swim_heat_total",key)
        b=find("V3-F50","swim_heat_total",key,"supported")
        lines.append(f"| {key} | {a['scores']['date_mae']:.3f} | {a['scores']['mae']:.3f} | {b['n_scored']} |")
    lines += ["",f"- 默认方法为 {adopted('swim_heat_total')[0]}，采用状态为 {adopted('swim_heat_total')[1]}。",
              "- 以实际总时长为条件，输出尺度为设备记录的运动热量。","",
              "## 日级和小时级","","逐指标、逐协议的决定见 [decision_table.csv](decision_table.csv)。",
              "指标 JSON 含训练常量与相邻值对照、RMSE、P90、R² 及预测与观测波动比。","",
              "| 日级摘要 | 主覆盖日 | 12 小时规则覆盖日 |","|---|---:|---:|"]
    for target in ("hr","spo2","pressure"):
        rr=[r for r in result["inventory"] if r["target_id"]==target+"_observed_hour_balanced_v3"]
        lines.append(f"| {rr[0]['label']} | {rr[0]['observed_slots']} | {rr[1]['observed_slots']} |")
    calendar_decisions=[r for r in result["decisions"] if not r["target_id"].startswith("swim") and r["status"]=="research_candidate"]
    lines.extend(["",f"日级及小时级学习模型达到采用规则的协议和指标组合共有 {len(calendar_decisions)} 项。","",
                  "## 固定与滚动","", "| 目标 | 模型 | 固定 MAE | 滚动 MAE | 共同样本 |",
                  "|---|---|---:|---:|---:|"])
    for r in result["rolling_comparisons"]:
        if r["target_id"].startswith("swim") and r["model_id"] in ("HR-R0","HR-R1","E-R0") and r["n_shared"]:
            lines.append(f"| {r['target_id']} | {r['model_id']} | {r['reference_metrics']['date_mae']:.3f} | "
                         f"{r['candidate_metrics']['date_mae']:.3f} | {r['n_shared']} |")
    (out/"results_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    checked=result["legacy"].get("checked",{})
    checks=["# V3.0 验证记录","",f"- P0 固定半段历史基准逐文件一致检查 {sum(checked.values())}/{len(checked)}。",
            f"- 核验了 {result['prediction_rows']:,} 条预测或拒答记录，校验唯一键、时间顺序和评分状态。",
            "- 运行清单保存数据、方案、代码、环境和产物指纹。",
            "- 冻结划分及标签扰动的独立行为检查见 tests/v3/test_contract.py。",
            "- S06 搜索按配置关闭。固定半段仅有四个训练日期。"]
    (out/"verification_report.md").write_text("\n".join(checks)+"\n",encoding="utf-8")
    dictionary="""# V3.0 数据字典

- 日期均按北京时间。日目标在次日零点可用，小时与游泳窗在其右端可用。
- 日心率、血氧和压力的新摘要取合格小时中位数。每小时至少三个有效五分钟格，日须覆盖至少八小时和三个六小时时段。
- 小睡总时长和单次中位数分开。无记录表示缺失。主睡眠按事件结束日期索引。
- 活动统计使用工作簿日值，体成分只做观测展示。
- 游泳目标为发出时刻加提前量后结束的 30 秒心率窗，单位次/分。热量为设备整场记录，单位千卡。
- 逐条预测的 issued_at 为空表示未发出。outside_record 表示已发出但目标超出完整记录窗口。
- source_ids 连接清理记录或源表记录序号。汇总时间及信息可用时间记录在 aggregate_lineage.csv。
- date_mae 对日等权，有场次时先在场次内平均再在日内等权。
- 在线一步输入为已到达真实值，开环输入为训练末端状态或此前生成的估计。
- 热量 formal_output 受训练时长支持范围约束。超范围的研究误差仍保存。
"""
    (out/"data_dictionary.md").write_text(dictionary,encoding="utf-8")
