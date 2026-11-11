"""Apply V3 evaluation to independent workbook-level subjects without changing V3.0."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from math import isfinite
from statistics import median
import argparse
import hashlib
import json
import platform
import traceback
from html import escape
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from threadpoolctl import threadpool_limits

import swim
from v3 import daily, heat, heart
from v3.aggregate import DAY, HOUR, Point, day_values, floor_day, sensor_grid, sleep_days
from v3.data_quality import TZ, SENSORS, clean_sensors, local_time
from v3.evaluate import stats
from v3.run import (ROOT, CONFIG, PLAN, ACTIVITY, BODY, SENSOR_IDS, aggregates_and_splits,
                    observed_fields, output_csv, output_json, score_all)
from v3.splits import runs, windows, prior_sessions
from v3.verify import digest, verify_rows

SOURCE = ROOT.parent/"测试数据/所有数据"
DEST = ROOT.parent/"测试数据/分析/V3多对象孪生测试"
MINUTE = timedelta(minutes=1)
TIMES = {"心率":"H","血氧饱和度":"G","压力":"G","活动统计":"N","睡眠":"G",
         "单次运动":"G","体重":"U","日常活动状态":"H"}
TYPES = {"户外跑步","户外骑行","户外步行","走路","游泳"}
METRIC_IDS = ("hr_observed_hour_balanced_v3","spo2_observed_hour_balanced_v3","pressure_observed_hour_balanced_v3",
              "day:静息心率","day:主睡眠时长","day:小睡总时长","day:步数","day:日活动热量","day:运动分钟","day:活动小时")
COLORS = {"D-B0":"#475569","D-B1":"#9d6b12","D-B2":"#c66c15","D-B3":"#7b55a0","D-R0":"#037692",
          "HR-B0":"#475569","HR-B1":"#9d6b12","HR-R0":"#037692","HR-R1":"#188568",
          "E-B0":"#475569","E-R0":"#037692","E-R1":"#c66c15","E-R2":"#188568"}
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False})


def calendar(data):
    dates=set()
    for sheet,col in TIMES.items():
        for row in data.get(sheet,[]):
            try:
                dates.add(floor_day(local_time(row[col])))
            except (KeyError,ValueError):
                pass
    if len(dates)<2:
        raise ValueError("No usable multi-day timeline")
    start,end=min(dates),max(dates)
    days=[start+DAY*i for i in range((end-start).days+1)]
    return days,days[len(days)//2]


def gather(data,days):
    clean,audit=clean_sensors(data)
    targets,strict,coverage=[],[],[]
    for name,slug in SENSOR_IDS.items():
        unit=SENSORS[name][2]
        h,d,c=sensor_grid(clean[name],days,CONFIG["hour_min_5min_bins"],CONFIG["day_min_hours"],
                          CONFIG["day_min_6h_blocks"])
        _,s,_=sensor_grid(clean[name],days,CONFIG["hour_min_5min_bins"],CONFIG["strict_day_min_hours"],
                          CONFIG["day_min_6h_blocks"])
        tid=f"{slug}_observed_hour_balanced_v3"
        targets.extend([(tid,name+"日摘要",unit,"day",d,"model"),(f"hour:{name}",name,unit,"hour",h,"model")])
        strict.append((tid,name+"日摘要（严格）",unit,"day",s,"model"))
        coverage.extend(dict(r,metric=name,strict_day_eligible=r["slot"] in s if r["resolution"]=="day" else None) for r in c)
    targets.append(("day:静息心率","静息心率","次/分","day",day_values(
        (t,v,rid) for t,(v,rid) in clean["静息心率"].items()),"model"))
    sleep,events,sleep_log=sleep_days(data["睡眠"])
    audit.extend(dict(r,metric="睡眠事件") for r in sleep_log)
    targets.extend((f"day:{label}",label,"分钟","day",p,"model") for label,p in sleep.items())
    targets+=observed_fields(data,"活动统计",ACTIVITY,"N","model")
    targets+=observed_fields(data,"体重",BODY,"U","observed_only")
    return clean,audit,targets,strict,coverage,events


def minute_experiment(points):
    groups=defaultdict(list)
    for t,(value,raw_id) in points.items():
        groups[t.replace(second=0,microsecond=0)].append((value,raw_id))
    series={t:Point(float(median(v for v,_ in group)),len(group),tuple(i for _,i in group))
            for t,group in sorted(groups.items())}
    tid="minute:实时心率"
    window=windows(tid,series,[],MINUTE,None,False)[0]
    window.protocol_id="V3-C50-MIN"
    rows,fits=daily.evaluate_series(tid,window,series,MINUTE,CONFIG)
    lineage=[dict(target_id=tid,aggregate_id=f"{tid}:{t.isoformat()}",slot=t,available_at=t+MINUTE,
                  rule="minute_median",value=p.value,count=p.count,source_ids=p.sources) for t,p in series.items()]
    split=[dict(protocol_id=window.protocol_id,segment_id=window.segment_id,target_id=tid,phase=phase,slot=t,
                observed=True,aliases=()) for phase,part in (("train",window.train),("test",window.test)) for t in part]
    return series,window,rows,fits,lineage,split


def exercise_inputs(data,points,cutoff):
    states=defaultdict(set)
    for r in data.get("日常活动状态",[]):
        try:states[local_time(r["H"])].add(r["F"])
        except (KeyError,ValueError):continue
    events,sessions,observed=[],[],{}
    times=sorted(points)
    import bisect
    for index,r in sorted(enumerate(data.get("单次运动",[]),1),key=lambda x:x[1].get("G","")):
        try:start,end=local_time(r["G"]),local_time(r["H"])
        except (KeyError,ValueError):continue
        candidates=states[start]
        label=r.get("F") or (next(iter(candidates)) if len(candidates)==1 else "未知")
        phase="train" if end<=cutoff and start<cutoff else "test" if start>=cutoff else "excluded_cross_boundary"
        if end<=start:phase="invalid_time"
        key=f"A{index:03}"
        row=dict(session=key,source_id=f"单次运动:{index}",activity_type=label,start=start,end=end,phase=phase,
                 confirmed=label in TYPES)
        events.append(row)
        if phase in ("excluded_cross_boundary","invalid_time") or label not in TYPES:
            continue
        duration=(end-start).total_seconds()
        length=int(duration//swim.BIN)
        values=[[] for _ in range(length)]
        source=[[] for _ in range(length)]
        for stamp in times[bisect.bisect_left(times,start):bisect.bisect_left(times,end)]:
            index=int((stamp-start).total_seconds()//swim.BIN)
            if index<length:
                value,rid=points[stamp]
                values[index].append(value)
                source[index].append(rid)
        y=np.array([float(median(v)) if len(v)>=2 else np.nan for v in values])
        s=swim.Swim(key,start,duration,np.arange(1,length+1)*swim.BIN,y,sum(map(len,values)),0)
        row["eligible_hr_windows"]=int(np.isfinite(y).sum())
        sessions.append(s)
        observed[key]=(r,source,row)
    return events,sessions,observed


def exercise_experiment(data,points,cutoff):
    events,sessions,observed=exercise_inputs(data,points,cutoff)
    rows,fits,lineage,split=[],[],[],[]
    for s in sessions:
        raw,source,event=observed[s.key]
        end=event["end"]
        split.append(dict(protocol_id="V3-F50-ACT",segment_id="confirmed_activity",
                          target_id="exercise_session",phase=event["phase"],slot=s.start,session=s.key,
                          observed=True,activity_type=event["activity_type"]))
        lineage.append(dict(target_id="exercise",aggregate_id="workout:"+s.key,slot=s.start,
                            available_at=end,source_ids=[event["source_id"]]))
        lineage.extend(dict(target_id="exercise_hr_window",aggregate_id=f"{s.key}:{j}",slot=s.start+j*heart.BIN,
                            available_at=s.start+(j+1)*heart.BIN,value=float(v),count=len(source[j]),
                            source_ids=source[j]) for j,v in enumerate(s.heart) if np.isfinite(v))
    if not sessions:
        return events,rows,fits,lineage,split
    # Shared V3 heart model is run on confirmed activities as a pooled exploratory cohort.
    # The swim target ID is retained internally for the existing scorer, then relabeled on export.
    train=prior_sessions(sessions,cutoff)
    test=[s for s in sessions if s.start>=cutoff]
    for h in CONFIG["heart_horizons"]:
        models,fit=heart.make_predictors(train,h,CONFIG["heart_min_train_dates"],CONFIG["heart_alpha"])
        meta=dict(protocol_id="V3-F50-ACT",segment_id="confirmed_activity",train_end=cutoff,
                  model_version=f"V3-F50-ACT:HR:{h}:{cutoff.isoformat()}")
        fits.append(dict(meta,target_id=f"swim_hr_{h}",train_sessions=[s.key for s in train],**fit))
        for s in test:
            rows.extend(dict(r,activity_type=observed[s.key][2]["activity_type"]) for r in heart.replay_swim(s,h,models,meta))
    # The existing conditional energy estimator uses actual total duration and device calories.
    # Each sample must have a valid positive duration and finite label.
    safe=[]
    for s in sessions:
        raw=dict(observed[s.key][0])
        try:
            if not (isfinite(float(raw["I"])) and float(raw["I"])>0 and isfinite(float(raw["O"]))):continue
        except (KeyError,ValueError):continue
        safe.append(s)
    if safe:
        heat_rows,params=heat.experiment(data,safe,cutoff)
        for r in [*heat_rows,*params]:
            r["protocol_id"]="V3-F50-ACT"
            r["model_version"]=r["model_version"].replace("V3-F50:","V3-F50-ACT:",1)
        for r in heat_rows:
            r["prediction_id"]=r["prediction_id"].replace("V3-F50:","V3-F50-ACT:",1)
            r["activity_type"]=observed[r["session"]][2]["activity_type"]
        rows.extend(heat_rows)
        fits.extend(params)
    return events,rows,fits,lineage,split


def rename_exercise(value):
    if isinstance(value,dict):
        return {k:rename_exercise(v) for k,v in value.items()}
    if isinstance(value,list):
        return [rename_exercise(v) for v in value]
    if isinstance(value,str):
        return value.replace("swim_hr_","exercise_hr_").replace("swim_heat_","exercise_heat_")
    return value


def visible_scores(results,protocol,target,mode="online_one_step"):
    return [r for r in results if r["protocol_id"]==protocol and r["target_id"]==target
            and r["cohort"]=="all" and r["information_mode"]==mode and r["scores"]]


def report(out,obj,result,predictions,targets,minute,window,events):
    days=result["calendar_days"]
    result_rows=result["results"]
    target_map={t[0]:t for t in targets}
    title=f"对象 {obj:02d} · V3 孪生算法测试"
    fmt=lambda x:f"{x:,.2f}" if x is not None else "—"
    table=[]
    base={"D-B0":"训练中位数","D-B1":"训练均值","D-B2":"上一时点实测","D-B3":"训练末值","D-R0":"Ridge"}
    for proto,tid,mode in [
            *(( "V3-F50",t,"online_one_step") for t in METRIC_IDS),
            *(( "V3-F50",t,"open_loop") for t in METRIC_IDS),
            *(( "V3-C50",t,mode) for t in METRIC_IDS for mode in ("online_one_step","open_loop")
              if any(r["protocol_id"]=="V3-C50" and r["target_id"]==t for r in result_rows)),
            *(( "V3-C50",f"hour:{n}","online_one_step") for n in SENSOR_IDS),
            ("V3-C50-MIN","minute:实时心率","online_one_step"),
            ("V3-C50-MIN","minute:实时心率","open_loop")]:
        label=target_map[tid][1] if tid in target_map else "分钟实时心率"
        options=visible_scores(result_rows,proto,tid,mode)
        ds=next((r for r in result["decisions"] if r["protocol_id"]==proto and r["target_id"]==tid
                 and r.get("information_mode")==mode),{})
        model=next((r for r in options if r["model_id"]=="D-R0"),None)
        candidate=model or next((r for r in options if r["model_id"]==ds.get("selected")),None) or next(iter(options),None)
        if candidate is None:
            table.append([proto,label,mode,"证据不足","—","—","—","—","—",ds.get("status","insufficient_evidence")])
            continue
        scores=candidate["scores"]
        ref=[c for c in ds.get("comparisons",[]) if c.get("reference_metrics")]
        best=min(ref,key=lambda c:c["reference_metrics"]["date_mae"]) if ref else None
        table.append([proto,label,mode,base.get(candidate["model_id"],candidate["model_id"]),scores["n"],fmt(scores["mae"]),
                      fmt(scores["rmse"]),fmt(scores["p90_abs"]),
                      f"{base.get(best['reference'],best['reference'])} {fmt(best['reference_metrics']['date_mae'])}（共同样本日期 MAE）" if best else "—",
                      ds.get("status","insufficient_evidence")])
    for tid,mode in [("exercise_hr_30","online_one_step"),("exercise_hr_120","online_one_step"),
                     ("exercise_heat_total","conditional_duration")]:
        for r in visible_scores(result_rows,"V3-F50-ACT",tid,mode):
            x=r["scores"]
            table.append(["V3-F50-ACT",tid,mode,r["model_id"],x["n"],fmt(x["mae"]),fmt(x["rmse"]),
                          fmt(x["p90_abs"]),f"日期均衡 MAE {fmt(x['date_mae'])}","探索性"])
    columns=("协议","目标","信息条件","模型","评分数","MAE","RMSE","P90","基线或补充","采用状态")
    output_csv(out/"关键结果.csv",[dict(zip(columns,r)) for r in table],columns)
    lines=[f"# {title}","",f"源文件为健康数据{obj}.xlsx。日历共 {days} 日，固定切分为前 {days//2} 日训练、后 {days-days//2} 日测试。",
           f"源表数据 {result['source_rows']:,} 行。测量日覆盖与实际测试点数见连续性和划分文件。",
           "游泳任务按可确认场次计数。确认数为零时，不报告游泳心率与游泳热量误差。",
           "分钟补充实验使用至少一条清理后的实时心率计算分钟中位数，对最长连续分钟段执行相同单滞后 Ridge 和基线。",
           f"分钟最长连续段为 {len(window.train)+len(window.test)} 分钟，前 {len(window.train)} 分钟训练，后 {len(window.test)} 分钟测试。",
           "V3-F50-ACT 为确认运动的混合场次补充实验，沿用 V3 预测器和同样的时间分界。步行、跑步、骑行不视作游泳。","",
           "## 通过采用门槛的主要结果",""]
    passed=[d for d in result["decisions"] if d["status"]=="research_candidate"
            and d["protocol_id"] in ("V3-F50","V3-C50","V3-C50-MIN","V3-F50-ACT")]
    for d in passed:
        c=min(d["comparisons"],key=lambda v:v["reference_metrics"]["date_mae"])
        target=target_map.get(d["target_id"],(None,d["target_id"],"原指标单位"))
        a,b=c["candidate_metrics"],c["reference_metrics"]
        lines.append(f"- {target[1]}（{d['protocol_id']}，{d['information_mode']}）"
                     f"在共同 {a['n_dates']} 个日期的日期均衡 MAE 为 {fmt(a['date_mae'])} {target[2]}，"
                     f"最强对照 {base.get(c['reference'],c['reference'])} 为 {fmt(b['date_mae'])} {target[2]}，"
                     f"下降 {100*c['relative_gain']:.1f}%。模型 RMSE 为 {fmt(a['rmse'])} {target[2]}，"
                     f"测试 R² 为 {a['r2']:.3f}。" if a["r2"] is not None else
                     f"- {target[1]}（{d['protocol_id']}，{d['information_mode']}）通过本轮采用门槛。")
    if not passed:
        lines.append("主要协议中没有学习模型达到采用门槛。")
    lines+=["","## 各指标误差","","| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    lines.extend("| "+" | ".join(map(str,r))+" |" for r in table)
    (out/"实验报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    figs=[]
    focus=[t for t in targets if t[0] in METRIC_IDS]
    fig,axes=plt.subplots((len(focus)+1)//2,2,figsize=(13,11),layout="constrained")
    first,last=result["calendar_start"],result["calendar_end"]
    grid=[first+i*DAY for i in range(days)]
    for ax,(tid,label,unit,_,points,_) in zip(axes.flat,focus):
        ax.plot(grid,[points[t].value if t in points else np.nan for t in grid],".-",color="#172f43",label="实测")
        for model in ("D-B0","D-B2","D-R0"):
            rs=[r for r in predictions if r["protocol_id"]=="V3-F50" and r["target_id"]==tid
                and r["information_mode"]=="online_one_step" and r["model_id"]==model]
            if any(r["prediction"] is not None for r in rs):
                ax.plot([r["target_start"] for r in rs],[r["prediction"] for r in rs],lw=1,color=COLORS[model],label=model)
        ax.axvline(result["cutoff"],color="#c66c15",alpha=.6)
        ax.set_title(label);ax.set_ylabel(unit);ax.grid(alpha=.2);ax.legend(fontsize=7,loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.suptitle(title+" · 日级固定半段在线一步")
    fig.savefig(out/"01_日级固定半段.png",dpi=135);plt.close(fig);figs.append("01_日级固定半段.png")
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout="constrained")
    for ax,tid in zip(axes.flat,("hour:实时心率","hour:血氧","hour:压力","minute:实时心率")):
        points=minute if tid.startswith("minute:") else target_map[tid][4]
        proto="V3-C50-MIN" if tid.startswith("minute:") else "V3-C50"
        rr=[r for r in predictions if r["protocol_id"]==proto and r["target_id"]==tid
            and r["information_mode"]=="online_one_step"]
        if rr:
            slots=sorted({r["target_start"] for r in rr})
            ax.plot(slots,[points[t].value if t in points else np.nan for t in slots],".-",color="#172f43",label="实测")
            for model in ("D-B0","D-B2","D-R0"):
                ss=[r for r in rr if r["model_id"]==model]
                if any(r["prediction"] is not None for r in ss):
                    ax.plot([r["target_start"] for r in ss],[r["prediction"] for r in ss],color=COLORS[model],lw=1,label=model)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            ax.tick_params(axis="x",rotation=20,labelsize=8);ax.legend(fontsize=8)
        else:
            ax.text(.2,.5,"无可评估连续段",transform=ax.transAxes)
        ax.set_title(tid.replace("hour:","小时 · ").replace("minute:","分钟 · "));ax.grid(alpha=.2)
    fig.suptitle(title+" · 最长连续段测试")
    fig.savefig(out/"02_小时与分钟连续段.png",dpi=135);plt.close(fig);figs.append("02_小时与分钟连续段.png")
    if any(r["protocol_id"]=="V3-F50-ACT" for r in result_rows):
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
        for ax,(tid,unit) in zip(axes,(("exercise_hr_30","次/分"),("exercise_hr_120","次/分"),("exercise_heat_total","千卡"))):
            rr=visible_scores(result_rows,"V3-F50-ACT",tid,"conditional_duration" if "heat" in tid else "online_one_step")
            if rr:
                ax.bar([r["model_id"] for r in rr],[r["scores"]["date_mae"] for r in rr],
                       color=[COLORS[r["model_id"]] for r in rr])
            else:ax.text(.2,.5,"可评分样本不足",transform=ax.transAxes)
            ax.set_title(tid);ax.set_ylabel("日期均衡 MAE（"+unit+"）")
        fig.suptitle(title+" · 确认运动场次探索")
        fig.savefig(out/"03_确认运动对照.png",dpi=135);plt.close(fig);figs.append("03_确认运动对照.png")
    links=[("实验报告.md","文字报告"),("关键结果.csv","关键结果"),("metrics.json","完整指标"),
           ("predictions.csv","逐条预测"),("continuity.csv","连续性"),("decision_table.csv","采用状态"),
           ("manifest.json","运行清单")]
    body=" · ".join(f'<a href="{quote(f)}">{label}</a>' for f,label in links)
    body+="".join(f'<figure><img src="{quote(f)}" alt="{escape(f)}"></figure>' for f in figs)
    page=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title><style>body{{max-width:1220px;margin:2rem auto;font:16px/1.6 sans-serif;color:#173847}}img{{width:100%}}</style><h1>{title}</h1>{body}</html>'
    (out/"index.html").write_text(page,encoding="utf-8")


def run_subject(out,obj,source):
    started=datetime.now(TZ)
    payload=source.read_bytes()
    data=swim.read_workbook(payload)
    days,cutoff=calendar(data)
    clean,audit,targets,strict,coverage,sleep=gather(data,days)
    inv,agg,split,rows,fits=aggregates_and_splits(targets,strict,days,cutoff)
    minute,mw,mp,mf,ma,ms=minute_experiment(clean["实时心率"])
    events,ep,ef,ea,es=exercise_experiment(data,clean["实时心率"],cutoff)
    rows.extend(mp+ep);fits.extend(mf+ef);agg.extend(ma+ea);split.extend(ms+es)
    counts=verify_rows(rows)
    ids={r["aggregate_id"] for r in agg}
    assert len(ids)==len(agg)
    assert all(not r.get("observed_id") or r["observed_id"] in ids for r in rows)
    assert all(not r.get("input_id") or set(r["input_id"].split(";"))<=ids for r in rows)
    assert len({r["model_version"] for r in fits})==len(fits)
    verified_versions={r["model_version"]:r["train_end"] for r in fits}
    assert all(verified_versions[r["model_version"]]==r["train_end"] for r in rows)
    result,per_date,per_session,sensitivity=score_all(rows,[],inv)
    n_swims=sum(r["activity_type"]=="游泳" and r["confirmed"] for r in events)
    if not n_swims:
        result["decisions"].extend(dict(protocol_id="V3-F50",target_id=t,information_mode=mode,
                                        status="insufficient_evidence",selected=None,reason="no_confirmed_swimming")
                                   for t,mode in (("swim_hr_30","online_one_step"),("swim_hr_120","online_one_step"),
                                                  ("swim_heat_total","conditional_duration")))
    result.update(subject=obj,workbook=source.name,source_rows=sum(map(len,data.values())),
                  sheet_counts={k:len(v) for k,v in data.items()},calendar_days=len(days),
                  calendar_start=days[0],calendar_end=days[-1],cutoff=cutoff,
                  source_sha256=hashlib.sha256(payload).hexdigest(),inventory=inv,
                  minute_test=dict(eligible_slots=len(minute),longest_slots=len(mw.train)+len(mw.test),
                                   train=len(mw.train),test=len(mw.test)),
                  exercise=dict(count=len(events),confirmed_swims=n_swims,
                                types=Counter(r["activity_type"] for r in events),
                                confirmed_train=sum(r["confirmed"] and r["phase"]=="train" for r in events),
                                confirmed_test=sum(r["confirmed"] and r["phase"]=="test" for r in events)),
                  verified_prediction_rows=counts,quality=dict(
                      clean_counts={k:len(v) for k,v in clean.items()},
                      measurement_statuses=Counter(r["status"] for r in audit)))
    # Convert only the internally tagged confirmed-activity rows, never missing swim tasks.
    for container in (result["results"],result["decisions"],rows,fits,per_date,per_session,sensitivity):
        for j,r in enumerate(container):
            if r.get("protocol_id")=="V3-F50-ACT":
                container[j]=rename_exercise(r)
    output_json(out/"metrics.json",result)
    output_json(out/"model_parameters.json",fits)
    for file,values in (("cleaning_log.csv",audit),("coverage.csv",coverage),("continuity.csv",inv),
                        ("aggregate_lineage.csv",agg),("split_manifest.csv",split),("predictions.csv",rows),
                        ("decision_table.csv",result["decisions"]),("per_date_metrics.csv",per_date),
                        ("per_session_metrics.csv",per_session),("sensitivity_results.csv",sensitivity),
                        ("sleep_events.csv",sleep),("exercise_inventory.csv",events)):
        output_csv(out/file,values)
    extras=[dict(sheet=k,record=i,fields=r) for k in ("体温","血压","血糖","血脂","ecg心电图")
            for i,r in enumerate(data.get(k,[]),1)]
    output_csv(out/"低频实测记录.csv",extras,("sheet","record","fields"))
    snapshot={k:v for k,v in CONFIG.items() if k not in ("source_sha256","calendar_start","calendar_end","fixed_cutoff")}
    snapshot.update(source_sha256=result["source_sha256"],calendar_start=days[0],calendar_end=days[-1],fixed_cutoff=cutoff,
                    scope="independent_workbook",supplementary=["V3-C50-MIN","V3-F50-ACT"],
                    calibration="not_run; this batch covers point forecasts and conditional estimates")
    output_json(out/"protocol.json",snapshot)
    report(out,obj,result,rows,targets,minute,mw,events)
    dependencies=[Path(__file__),ROOT/"swim.py",ROOT/"build.py",ROOT/"v2/evaluate.py",
                  ROOT/"v3/protocol.json",*sorted((ROOT/"v3").glob("*.py"))]
    output_json(out/"manifest.json",dict(subject=obj,run_id=out.parent.name,status="completed",
                source_file=source.name,source_sha256=result["source_sha256"],plan_sha256=digest(PLAN),
                started=started.isoformat(),ended=datetime.now(TZ).isoformat(),python=platform.python_version(),
                code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in dependencies},
                artifact_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    return result


def summarize(dest,completed,failed):
    rows=[]
    for obj,result in completed.items():
        inv={r["target_id"]:r for r in result["inventory"] if r["rule"]=="primary"}
        model=visible_scores(result["results"],"V3-C50-MIN","minute:实时心率")
        rm=lambda key:next((r["scores"]["rmse"] for r in model if r["model_id"]==key),None)
        rows.append(dict(对象=f"对象{obj:02d}",行数=result["source_rows"],日历天=result["calendar_days"],
                         心率有效日=inv["hr_observed_hour_balanced_v3"]["observed_slots"],
                         血氧有效日=inv["spo2_observed_hour_balanced_v3"]["observed_slots"],
                         压力有效日=inv["pressure_observed_hour_balanced_v3"]["observed_slots"],
                         最长心率分钟=result["minute_test"]["longest_slots"],分钟Ridge_RMSE=rm("D-R0"),
                         分钟上一值_RMSE=rm("D-B2"),确认游泳=result["exercise"]["confirmed_swims"],
                         确认运动训练=result["exercise"]["confirmed_train"],确认运动测试=result["exercise"]["confirmed_test"],
                         预测记录=result["verified_prediction_rows"]))
    output_csv(dest/"多对象概览.csv",rows)
    lines=["# 多对象独立孪生测试","",
           "对象编号对应工作簿编号。训练、标准化、测试、模型决定与图表均逐工作簿独立完成。",
           "分钟级误差用于同一对象同一测试段的对照，测试时间和覆盖不同，不能作人员排名。","",
           "| 对象 | 日历天 | 心率有效日 | 血氧有效日 | 压力有效日 | 最长连续心率分钟 | 分钟 Ridge RMSE | 分钟上一值 RMSE |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        nums=[f"{r[k]:.2f}" if r[k] is not None else "—" for k in ("分钟Ridge_RMSE","分钟上一值_RMSE")]
        lines.append("| "+" | ".join(str(r[k]) for k in ("对象","日历天","心率有效日","血氧有效日","压力有效日","最长心率分钟"))+" | "+" | ".join(nums)+" |")
    lines+=["","分钟误差单位为次/分。有效日采用 V3 固定覆盖门槛。",""]
    for obj in sorted(completed):
        folder=f"对象{obj:02d}_健康数据{obj}"
        lines.append(f"- [对象 {obj:02d} 离线图册]({folder}/index.html) · [实验报告]({folder}/实验报告.md)")
    for obj,err in failed.items():lines.append(f"- 对象 {obj:02d} 执行失败。{err}")
    (dest/"批次说明.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(output=str(dest),results=rows,failed=failed),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path,default=DEST)
    parser.add_argument("--subjects",type=int,nargs="+",default=list(range(2,7)))
    args=parser.parse_args()
    dest=args.output/("批次-"+datetime.now(TZ).strftime("%Y%m%d-%H%M%S-%f"))
    dest.mkdir(parents=True)
    completed,failed={},{}
    for obj in sorted(set(args.subjects)):
        out=dest/f"对象{obj:02d}_健康数据{obj}"
        out.mkdir()
        try:
            with threadpool_limits(limits=1):
                completed[obj]=run_subject(out,obj,args.source/f"健康数据{obj}.xlsx")
            print(f"对象 {obj:02d} 完成",flush=True)
        except Exception as err:
            failed[obj]=str(err)
            (out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
            print(f"对象 {obj:02d} 失败: {err}",flush=True)
    summarize(dest,completed,failed)
    if failed:
        raise SystemExit(1)


if __name__=="__main__":
    main()
