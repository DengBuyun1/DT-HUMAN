"""Local-only V3.0 experiment entry point."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import csv
from math import isfinite
from pathlib import Path
import json
import platform
import shutil
import sys
import traceback

import numpy as np
import sklearn
import threadpoolctl

import build
import swim
from v3.aggregate import DAY,HOUR,day_values,floor_day,sensor_grid,sleep_days
from v3.data_quality import SENSORS,TZ,clean_sensors,local_time
from v3 import daily,heart,heat
from v3.calibration import calibrate
from v3.evaluate import compare,decision,stats,valid
from v3.splits import Window,runs,windows
from v3.verify import digest,verify_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT/"v3/protocol.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PLAN = ROOT/"docs/plans/04_V3.0_数据口径修正与受控预测优化实施方案_2026-11-09.md"
LEGACY = ("20261109-110040-088694","v2-20261109-170819-386454","连续半段-20261109-192322-716804")
ACTIVITY = {"步数":("F","步"),"日活动热量":("G","千卡"),"运动分钟":("H","分钟"),"活动小时":("I","小时")}
BODY = {"体重":("F","kg"),"BMI":("G","kg/m²"),"体脂量":("H","kg"),"体脂率":("I","%"),
        "肌肉量":("J","kg"),"基础代谢":("K","千卡/天"),"水分量":("L","kg"),"水分率":("M","%"),
        "骨盐量":("O","kg"),"蛋白质率":("P","%"),"身体年龄":("Q","岁"),"身体得分":("R","分"),"骨骼肌量":("S","kg")}
SENSOR_IDS = {"实时心率":"hr","血氧":"spo2","压力":"pressure"}


def encoded(value):
    if isinstance(value,datetime):
        return value.replace(tzinfo=TZ).isoformat()
    raise TypeError(type(value))


def output_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=encoded)+"\n",encoding="utf-8")


def csv_cell(value):
    if isinstance(value,datetime):
        return encoded(value)
    if isinstance(value,(dict,list,tuple)):
        return json.dumps(value,ensure_ascii=False,allow_nan=False,default=encoded)
    return value


def output_csv(path,rows,columns=()):
    fields = dict.fromkeys(columns)
    for row in rows:
        fields.update(dict.fromkeys(row))
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({k:csv_cell(v) for k,v in row.items()} for row in rows)


def calendar():
    first=datetime.fromisoformat(CONFIG["calendar_start"])
    last=datetime.fromisoformat(CONFIG["calendar_end"])
    days=[first+i*DAY for i in range((last-first).days+1)]
    cutoff=datetime.fromisoformat(CONFIG["fixed_cutoff"])
    assert cutoff==days[len(days)//2]
    return days,cutoff


def legacy_evidence(repeat):
    refs={name:{p.name:digest(p) for p in sorted((ROOT/"runs"/name).iterdir())
               if p.is_file() and p.suffix in (".json",".csv")} for name in LEGACY}
    fixed=refs[LEGACY[-1]]
    assert len(fixed)==6 and all(refs.values())
    if repeat is None:
        return dict(reference_files=refs, status="reference_hashes_only")
    equality={name:digest(repeat/name)==checksum for name,checksum in fixed.items()}
    assert all(equality.values()),equality
    return dict(reference_files=refs,status="byte_identical",replay=repeat.name,checked=equality)


def observed_fields(data,sheet,specs,time_column,prefix):
    targets=[]
    for label,(column,unit) in specs.items():
        records=[]
        for i,row in enumerate(data[sheet],1):
            try:
                stamp,value=local_time(row[time_column]),float(row[column])
            except (KeyError,ValueError):
                continue
            if isfinite(value):
                records.append((stamp,value,f"{sheet}:{i}"))
        point=day_values(records)
        if sheet=="活动统计" and any(p.count!=1 for p in point.values()):
            raise ValueError(f"Duplicate activity totals for {label}")
        targets.append((f"day:{label}",label,unit,"day",point,prefix))
    return targets


def prepare(data,days):
    clean,audit=clean_sensors(data)
    current=[{"F":str(value),"H":t.isoformat(),"G":"实时心率"} for t,(value,_) in clean["实时心率"].items()]
    sessions,labels=swim.build_swims(dict(data,心率=current))
    sensor_targets,strict_targets,coverage,diagnostics = [],[],[],[]
    old=build.observations(data)["series"]
    raw_counts=Counter()
    for row in audit:
        if row["timestamp"] is not None:
            raw_counts[(row["metric"],floor_day(row["timestamp"]),row["status"])]+=1
    swim_spans=[(s.start,s.start+timedelta(seconds=s.duration)) for s in sessions]
    swim_readings=Counter(floor_day(t) for t in clean["实时心率"]
                         if any(a<=t<b for a,b in swim_spans))
    for name,slug in SENSOR_IDS.items():
        unit=SENSORS[name][2]
        hour,day,cov=sensor_grid(clean[name],days,CONFIG["hour_min_5min_bins"],
                                 CONFIG["day_min_hours"],CONFIG["day_min_6h_blocks"])
        _,strict,_=sensor_grid(clean[name],days,CONFIG["hour_min_5min_bins"],
                              CONFIG["strict_day_min_hours"],CONFIG["day_min_6h_blocks"])
        tid=f"{slug}_observed_hour_balanced_v3"
        sensor_targets.extend([(tid,name+"观测小时等权摘要",unit,"day",day,"model"),
                               (f"hour:{name}",name,unit,"hour",hour,"model")])
        strict_targets.append((tid,name+"观测小时等权摘要（严格）",unit,"day",strict,"model"))
        for row in cov:
            item=dict(row,metric=name)
            if row["resolution"]=="day":
                t=row["slot"]
                item.update(strict_day_eligible=t in strict, conflict_rows=raw_counts[name,t,"conflict"],
                            duplicate_rows=raw_counts[name,t,"duplicate"])
            item["old_hour_eligible"]=(row["raw_unique"]>=3) if row["resolution"]=="hour" else None
            coverage.append(item)
        by_day={local_time(t):v for t,v,_ in old[name]["points"]}
        day_cov={r["slot"]:r for r in cov if r["resolution"]=="day"}
        for t in days:
            n=day_cov[t]["raw_unique"]
            diagnostics.append(dict(metric=name,date=t.date().isoformat(),old_raw_median=by_day.get(t),
                                    new_hour_balanced=day[t].value if t in day else None,
                                    strict_eligible=t in strict,eligible_hours=day_cov[t]["valid_hours"],
                                    swim_fraction=swim_readings[t]/n if name=="实时心率" and n else None))
    rest=day_values((t,value,rid) for t,(value,rid) in clean["静息心率"].items())
    sleep,events,sleep_log=sleep_days(data["睡眠"])
    audit.extend(dict(row,metric="睡眠事件") for row in sleep_log)
    targets=sensor_targets+[
        ("day:静息心率","静息心率","次/分","day",rest,"model"),
        *((f"day:{label}",label,"分钟","day",pts,"model") for label,pts in sleep.items())
    ]
    targets+=observed_fields(data,"活动统计",ACTIVITY,"N","model")
    targets+=observed_fields(data,"体重",BODY,"U","observed_only")
    joint=set.intersection(*(set(pts) for _,_,_,res,pts,_ in targets if res=="hour"))
    joint_runs=runs(joint,HOUR)
    counts=dict(labels=labels, cleaned_records={k:len(v) for k,v in clean.items()},
                statuses=Counter((r["status"]) for r in audit), hourly_joint_count=len(joint),
                hourly_joint_longest=len(joint_runs[0]) if joint_runs else 0)
    return targets,strict_targets,coverage,diagnostics,sessions,clean,audit,events,counts


def aggregates_and_splits(targets,strict_targets,days,cutoff):
    inventory,aggregates,splits,predictions,fits=[],[],[],[],[]
    grid_by_res={"day":days,"hour":[day+i*HOUR for day in days for i in range(24)]}
    steps={"day":DAY,"hour":HOUR}
    for spec,group in (("primary",targets),("strict_12h",strict_targets)):
        for tid,label,unit,res,pts,role in group:
            step=steps[res]
            segments=runs(pts,step)
            inventory.append(dict(target_id=tid,label=label,unit=unit,resolution=res,rule=spec,role=role,
                                  observed_slots=len(pts),calendar_slots=len(grid_by_res[res]),
                                  longest_slots=len(segments[0]) if segments else 0,
                                  longest_start=segments[0][0] if segments else None,
                                  longest_end=segments[0][-1] if segments else None,
                                  all_segments=[dict(start=x[0],end=x[-1],slots=len(x)) for x in segments]))
            if spec=="primary":
                aggregates.extend(dict(target_id=tid,aggregate_id=f"{tid}:{t.isoformat()}",
                                       slot=t,available_at=t+step,rule=spec,value=p.value,count=p.count,
                                       source_ids=p.sources) for t,p in sorted(pts.items()))
            if role=="observed_only":
                continue
            specs=windows(tid,pts,grid_by_res[res],step,cutoff,res=="day")
            if spec=="strict_12h":
                specs=[Window("V3-SENS",w.segment_id+":12h:"+w.protocol_id,w.train,w.test,w.aliases) for w in specs]
            elif res=="day":
                fixed=specs[0]
                specs.append(Window("V3-ROLL",tid+":full",fixed.train,fixed.test))
            for window in specs:
                for phase,slots in (("train",window.train),("test",window.test)):
                    splits.extend(dict(protocol_id=window.protocol_id,segment_id=window.segment_id,target_id=tid,
                                       phase=phase,slot=t,observed=t in pts,aliases=window.aliases) for t in slots)
                if window.test:
                    rows,models=daily.evaluate_series(tid,window,pts,step,CONFIG,window.protocol_id=="V3-ROLL")
                    predictions.extend(rows)
                    fits.extend(models)
    return inventory,aggregates,splits,predictions,fits


def annotate_workouts(data,sessions,clean,cutoff):
    workout_ids={local_time(r["G"]):f"单次运动:{i}" for i,r in enumerate(data["单次运动"],1)}
    rows,split=[],[]
    for s in sessions:
        before=s.start.date()<cutoff.date()
        end=s.start+timedelta(seconds=s.duration)
        phase="excluded_cross_boundary" if before and end>cutoff else "train" if before else "test"
        split.append(dict(protocol_id="V3-F50",segment_id="swim",target_id="swim_session",phase=phase,
                          slot=s.start,observed=True,session=s.key))
        rows.append(dict(target_id="workout",aggregate_id=f"workout:{s.key}",slot=s.start,available_at=end,
                         source_ids=[workout_ids[s.start]]))
        groups=defaultdict(list)
        for t,(_,rid) in clean["实时心率"].items():
            index=int((t-s.start).total_seconds()//swim.BIN)
            if 0<=index<len(s.heart):
                groups[index].append(rid)
        for j,value in enumerate(s.heart):
            if np.isfinite(value):
                edge=s.start+(j+1)*heart.BIN
                rows.append(dict(target_id="swim_hr_window",aggregate_id=f"{s.key}:{j}",slot=edge-heart.BIN,
                                 available_at=edge,value=float(value),count=len(groups[j]),source_ids=groups[j]))
    return rows,split


def score_all(predictions,calibration,inventory):
    grouped=defaultdict(list)
    for row in predictions:
        group=(row["protocol_id"],row["target_id"],row["segment_id"],row["information_mode"])
        grouped[group].append(row)
    summaries,decisions,by_date_rows,by_swim_rows,sensitivity=[],[],[],[],[]
    for (protocol,tid,segment,mode),rows in sorted(grouped.items()):
        meta=dict(protocol_id=protocol,target_id=tid,segment_id=segment,information_mode=mode)
        kind="heart" if tid.startswith("swim_hr") else "heat" if tid.startswith("swim_heat") else "daily"
        groups={"all":rows}
        if kind=="heart":
            groups={"all":[r for r in rows if r["in_record"]],
                    "strict":[r for r in rows if r["strict"]]}
        elif kind=="heat":
            groups.update(supported=[r for r in rows if r["in_train_support"]],
                          outside_support=[r for r in rows if not r["in_train_support"]])
        tol=CONFIG["heart_tolerance_bpm"] if kind=="heart" else None
        for model in sorted({r["model_id"] for r in rows}):
            candidates=[r for r in rows if r["model_id"]==model]
            for slice_name,sub in groups.items():
                selected=[r for r in sub if r["model_id"]==model]
                n=len(valid(selected))
                scores=stats(selected,tol) if n>=CONFIG["min_test_points"] else None
                summary=dict(meta,model_id=model,cohort=slice_name,candidates=len(selected),
                             issued=sum(r["issued_at"] is not None for r in selected),n_scored=n,
                             observed_targets=sum(r["observed"] is not None for r in selected),
                             n_test_calendar_dates=len({r["date"] for r in selected}),scores=scores)
                if kind=="heart":
                    summary.update(issued_outside_record=sum(r["status"]=="outside_record" for r in candidates),
                                   in_record_eligible=sum(r["input_valid"] for r in selected),
                                   issued_target_missing=sum(r["status"]=="target_missing" for r in selected))
                if kind=="heat":
                    summary.update(formal_output=sum(r["formal_output"] for r in selected),
                                   reference_output=sum(r["reference_output"] for r in selected))
                summaries.append(summary)
                if slice_name!="all" or protocol=="V3-SENS":
                    sensitivity.append(dict({k:v for k,v in summary.items() if k!="scores"},**(scores or {})))
                if slice_name=="all":
                    dates=defaultdict(list)
                    sessions=defaultdict(list)
                    for r in valid(selected):
                        dates[r["date"]].append(r)
                        if r.get("session"):
                            sessions[(r["date"],r["session"])].append(r)
                    by_date_rows.extend(dict(meta,model_id=model,date=k,**stats(v,tol)) for k,v in sorted(dates.items()))
                    by_swim_rows.extend(dict(meta,model_id=model,date=k[0],session=k[1],**stats(v,tol)) for k,v in sorted(sessions.items()))
        result=decision(groups["all"] if groups["all"] else rows,CONFIG,kind,mode=="open_loop")
        comparisons=result.pop("comparisons")
        for comp in comparisons:
            sensitivity.extend(dict(meta,analysis="S03_delete_date",candidate=comp["candidate"],
                                    reference=comp["reference"],removed_date=d,delta_mae=v)
                               for d,v in comp.get("leave_one_date_out",{}).items())
        if kind=="heart":
            for model in sorted({r["model_id"] for r in rows}):
                strata=defaultdict(list)
                for r in valid([x for x in groups["all"] if x["model_id"]==model]):
                    y=r["observed"]
                    strata["<120" if y<120 else "120–<140" if y<140 else "140–<160" if y<160 else ">=160"].append(r)
                for stratum,sub in sorted(strata.items()):
                    sensitivity.append(dict(meta,analysis="S05_target_stratum",model_id=model,stratum=stratum,
                                            n_sessions=len({r["session"] for r in sub}),**stats(sub,tol)))
        result["comparisons"]=comparisons
        decisions.append(dict(meta,**result))
    stability=[]
    for (protocol,tid,segment,mode),updated in sorted(grouped.items()):
        if protocol!="V3-ROLL":
            continue
        fixed=grouped["V3-F50",tid,segment,mode]
        if not fixed:
            continue
        for model in sorted({r["model_id"] for r in updated}):
            matched=[dict(r,model_id=label) for label,rows in (("rolling",updated),("fixed",fixed))
                     for r in rows if r["model_id"]==model and r.get("in_record",True)]
            comparison=compare(matched,"rolling","fixed",CONFIG)
            result_row=dict(analysis="X06_fixed_vs_rolling",target_id=tid,model_id=model,
                            information_mode=mode,segment_id=segment,**comparison)
            stability.append(result_row)
            sensitivity.append(result_row)
    for row in inventory:
        if row["role"]=="observed_only":
            decisions.append(dict(protocol_id="observed",target_id=row["target_id"],segment_id=None,
                                  information_mode=None,status="observed_only",selected=None))
        elif row["longest_slots"]==0:
            decisions.append(dict(protocol_id="V3-C50" if row["rule"]=="primary" else "V3-SENS",
                                  target_id=row["target_id"],segment_id=row["rule"],information_mode=None,
                                  status="insufficient_evidence",selected=None,reason="no_eligible_segment"))
    band_metrics=[]
    for target,model in sorted({(r["target_id"],r["model_id"]) for r in calibration}):
        items=[r for r in calibration if r["target_id"]==target and r["model_id"]==model]
        enabled=[r for r in items if r["status"]=="enabled"]
        scored=[r for r in enabled if r["observed"] is not None]
        band_metrics.append(dict(target_id=target,model_id=model,issued=len(items),enabled=len(enabled),
                                 enabled_dates=len({r["date"] for r in enabled}),scored=len(scored),
                                 hits=sum(r["lower"]<=r["observed"]<=r["upper"] for r in scored),
                                 mean_width=float(np.mean([r["upper"]-r["lower"] for r in enabled])) if enabled else None))
    return dict(results=summaries,decisions=decisions,bands=band_metrics,rolling_comparisons=stability),by_date_rows,by_swim_rows,sensitivity


def run(out,workbook,legacy_repeat):
    started=datetime.now(TZ)
    config=CONFIG
    plan_hash=digest(PLAN)
    payload=workbook.read_bytes()
    source_hash=digest(workbook)
    if source_hash!=config["source_sha256"] or plan_hash!=config["plan_sha256"]:
        raise ValueError("Frozen plan/workbook SHA-256 mismatch")
    legacy=legacy_evidence(legacy_repeat)
    shutil.copyfile(PLAN,out/"plan_snapshot.md")
    shutil.copyfile(CONFIG_PATH,out/"protocol.json")
    data=swim.read_workbook(payload)
    days,cutoff=calendar()
    assert sum(map(len,data.values()))==30940
    targets,strict,cov,diag,sessions,clean,audit,sleep_events,quality=prepare(data,days)
    inventory,aggregates,splits,predictions,fits=aggregates_and_splits(targets,strict,days,cutoff)
    swim_aggregates,swim_splits=annotate_workouts(data,sessions,clean,cutoff)
    aggregates+=swim_aggregates
    splits+=swim_splits
    hr_fixed=None
    for evaluate,options in ((heart.experiment,{}),(heart.experiment,{"rolling":True}),
                             (heat.experiment,{}),(heat.experiment,{"rolling":True}),(heat.experiment,{"active":True})):
        args=(sessions,cutoff,config) if evaluate is heart.experiment else (data,sessions,cutoff)
        rows,params=evaluate(*args,**options)
        predictions.extend(rows)
        fits.extend(params)
        if evaluate is heart.experiment and not options:
            hr_fixed=rows
    cal=calibrate(hr_fixed,config)
    n_verified=verify_rows(predictions)
    aggregate_ids={row["aggregate_id"] for row in aggregates}
    assert len(aggregate_ids)==len(aggregates)
    for row in predictions:
        if row.get("observed_id"):
            assert row["observed_id"] in aggregate_ids, row["observed_id"]
        if row.get("input_id"):
            assert set(row["input_id"].split(";"))<=aggregate_ids, row["input_id"]
    model_versions={row["model_version"]:row["train_end"] for row in fits}
    assert len(model_versions)==len(fits)
    assert all(model_versions[row["model_version"]]==row["train_end"] for row in predictions)
    result,dates,per_swim,sensitivity=score_all(predictions,cal,inventory)
    result.update(quality=quality,source_rows=sum(map(len,data.values())),calendar_days=len(days),
                  cutoff=cutoff,inventory=inventory,legacy=legacy,prediction_rows=n_verified)
    for file,rows in (("cleaning_log.csv",audit),("coverage.csv",cov),("continuity.csv",inventory),
                      ("aggregation_comparison.csv",diag),("aggregate_lineage.csv",aggregates),
                      ("sleep_events.csv",sleep_events),("split_manifest.csv",splits),
                      ("predictions.csv",predictions),("per_date_metrics.csv",dates),
                      ("per_session_metrics.csv",per_swim),("sensitivity_results.csv",sensitivity),
                      ("decision_table.csv",result["decisions"]),("calibration.csv",cal),
                      ("platform_events.csv",data["预警数据"])):
        output_csv(out/file,rows)
    output_json(out/"model_parameters.json",fits)
    output_json(out/"metrics.json",result)
    from v3.figures import render, write_reports
    render(out,days,cutoff,targets,cov,diag,predictions,result,cal)
    write_reports(out,result)
    import matplotlib
    environment=dict(python=sys.version.split()[0],platform=platform.platform(),
                     numpy=np.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,
                     threadpoolctl=threadpoolctl.__version__,thread_limit=1,device="CPU",dtype="float64")
    paths=[ROOT/p for p in ("build.py","swim.py","v2/evaluate.py","plot_full_atlas.py","requirements.txt",
                           "vendor/garmin_lab/registry/store.py","tests/v3/test_contract.py")]
    paths+=sorted((ROOT/"v3").glob("*.py"))
    output_json(out/"manifest.json",dict(run_id=out.name,stage="04",version="V3.0",status="completed",
                source_sha256=source_hash,plan_sha256=plan_hash,parents=LEGACY,command=[sys.executable,*sys.argv],
                started_at=started.isoformat(),ended_at=datetime.now(TZ).isoformat(),
                source_checkpoint="9bbcf572b56fa6ca7920c3cd13adb93f05905697",local_code_hashes_define_run=True,
                availability_assumption="Asia/Shanghai; observations become available at window/event/day end; upload times unknown",
                matrix={f"X{i:02}":"executed" for i in range(9)},predictions_verified=n_verified,environment=environment,
                code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
                artifact_sha256={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook",type=Path,default=ROOT/"data/健康数据.xlsx")
    parser.add_argument("--legacy-repeat",type=Path)
    args=parser.parse_args()
    if args.legacy_repeat is None:
        parser.error("Run the P0 replay and provide its directory with --legacy-repeat.")
    name="v3-"+datetime.now(TZ).strftime("%Y%m%d-%H%M%S-%f")
    out=ROOT/"runs"/name
    out.mkdir()
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            run(out,args.workbook,args.legacy_repeat)
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    print(out)


if __name__=="__main__":
    main()
