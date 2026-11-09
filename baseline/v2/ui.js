Object.assign(names, {ridge_v1:"原版 Ridge",ridge:"加权 Ridge",kalman:"Kalman 局部趋势",legacy_selected:"原滚动选择"});

function renderV2() {
  const r=data.report;
  const heart=Object.entries(r.heart).map(([h,v])=> {
    const rows=Object.entries(v.metrics).map(([key,m])=>[
      names[key],m.days,m.n,num(m.date_mae,2),num(m.p90_abs,2),num(m.max_abs,2),
      `${m.within_tolerance}/${m.n}`,v.comparisons[key]?.passed?"达到增量筛选":key===v.model?"回放基线":"未达到增量筛选"
    ]);
    const band=v.empirical_band;
    const sorted=data.heart.rows.filter(x=>x.horizon===Number(h)&&x.status==="scored")
      .sort((a,b)=>Math.abs(a.selected-a.observed)-Math.abs(b.selected-b.observed));
    const examples=[sorted[0],sorted.at(-1)].filter(Boolean).map(x=>[
      x.session,`${num(x.origin_s/60)} → ${num(x.target_s/60)} 分`,
      num(x.selected),num(x.observed),num(Math.abs(x.selected-x.observed))
    ]);
    return `<h3>提前 ${h} 秒心率</h3><p>后续日期建议采用 ${names[v.recommended_next_model]}。历史回放登记的是预先指定的 ${names[v.model]}。</p>
    <p>有效范围内发出 ${v.issued}/${v.eligible} 条，目标缺测 ${v.missing_target} 条，另有 ${v.outside_recording_issued} 条在记录结束时撤回。</p>
    ${table(["模型","日期","可评分窗","日期等权 MAE","P90误差","最大误差","±10 次/分","筛选"],rows)}
    <p class="meta">历史误差参考在 ${band.days} 个后续日期可评分，命中 ${band.hits}/${band.n}，平均宽度 ${num(band.mean_width)} 次/分。</p>
    ${table(["回放案例","发出 → 目标","预测","实测","绝对误差"],examples)}`;
  }).join("");
  const heat=["all","in","out"].flatMap(scope=>Object.entries(r.heat_total.subgroups[scope]).map(([key,m])=>[
    {all:"全部",in:"范围内",out:"范围外"}[scope],names[key],m?.n??0,num(m?.date_mae,2),num(m?.p90_abs,2)
  ]));
  const duration=Object.entries(r.heat_paired_duration_metrics).map(([key,m])=>[{
    rate:"总时长 × 每分钟热量",line:"总时长约束直线",active_rate:"活动时间 × 每分钟热量",active_line:"活动时间约束直线"
  }[key],m?.n??0,num(m?.date_mae,2)]);
  const c=r.chronos;
  const chronos=c.status==="completed"?table(["历史窗口","同批评分窗","Chronos 日期 MAE","同批 Ridge 日期 MAE","发出 / 参考"],
    Object.entries(c.evaluation).map(([k,v])=>[k==="selected"?"此前日期选择":k.replace("context_",""),v.candidate_metrics?.n??0,
      num(v.candidate_metrics?.date_mae,2),num(v.reference_metrics?.date_mae,2),`${v.issued_candidate} / ${v.issued_reference}`])):`<p>${c.status==="failed"?"推理失败，详见本地运行日志。":"本次运行未请求预训练模型对照。"}</p>`;
  el("v2Results").innerHTML=`<p>单人探索性回顾评估。心率按场次、日期等权；下列误差用相同目标口径比较。</p>${heart}
    <h3>热量 · 总时长条件</h3>${table(["适用范围","模型","场次","日期等权 MAE（千卡）","P90误差"],heat)}
    <h3>时长字段对照</h3><p>活动时间是运动结束后的回顾信息。</p>${table(["时长定义和模型","同批场次","日期等权 MAE（千卡）"],duration)}
    <h3>Chronos-2 · 提前 120 秒</h3>${chronos}${c.status==="completed"?`<p class="meta">${c.lora_trigger?"开发门槛已达到，微调另行评估。":"开发门槛未达到，未启动微调。"}</p>`:""}`;
}
