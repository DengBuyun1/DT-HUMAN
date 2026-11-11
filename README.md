# DT-HUMAN

个人健康数字孪生实验原型。当前研究范围是选定指标的局部孪生，结合个人基线、趋势提取和条件推算。已归档运动心率、时长条件下的设备热量、日级与小时级指标，以及多对象独立回测。

- [2026-11-11 讨论与后续范围](baseline/docs/plans/2026-11-11_任务范围修订.md)
- [V3.0 实验与算法说明](baseline/docs/experiments/V3.0_阶段04实验与算法说明.md) · [全指标汇总](baseline/v3/results/aggregate_metrics.csv) · [模型决定](baseline/v3/results/decisions.csv)
- [多对象实验与运行方式](baseline/v3/cohort/README.md) · [批次汇总](baseline/v3/cohort/results/summary.json)

在 `baseline/` 安装依赖并运行合成行为测试。

```powershell
py -3.10 -m pip install -r requirements.txt matplotlib==3.10.6
py -3.10 -X utf8 -m unittest discover -s tests/v3 -v
```

公开仓库收录源码、公开方案、合成测试、汇总指标和复现说明。完整工作簿、逐点结果、台账和会议录音由授权数据持有方本地保管。真实数据复跑使用相应的本地归档。[V3.0 复跑说明](baseline/v3/README.md)

## 阶段回溯

| 阶段 | 快照 |
|---|---|
| 01 · V1 基线 | [初版归档提交](https://github.com/DengBuyun1/DT-HUMAN/commit/1709d4b21d564c40b020bdb635f99719410b4172) |
| 02 · V2 对照与回放 | [第二版归档提交](https://github.com/DengBuyun1/DT-HUMAN/commit/2578f2bd21b0bbc2f082757af5dd42283ba91566) |
| 03 · 连续性与固定半段实验 | [阶段 03 提交](https://github.com/DengBuyun1/DT-HUMAN/commit/9b1fde03fa9b8962dff80bf21e870bb223ce909f) · [实验说明与复跑](baseline/v2/results/continuous_half.md) · [汇总结果](baseline/v2/results/continuous_half_summary.json) · [文件指纹](baseline/v2/results/continuous_half_manifest.json) |
| 04 · V3.0 方案定稿 | [阶段定稿提交](https://github.com/DengBuyun1/DT-HUMAN/commit/9bbcf572b56fa6ca7920c3cd13adb93f05905697) · [版本索引](baseline/docs/plans/00_方案版本与执行顺序索引.md) · [详细方案](baseline/docs/plans/04_V3.0_数据口径修正与受控预测优化实施方案_2026-11-09.md) · [方案指纹](baseline/docs/plans/plan_manifest.json) |
| 04 · V3.0 本地实施与实验 | [复跑说明](baseline/v3/README.md) · [修订算法说明](baseline/docs/experiments/V3.0_阶段04实验与算法说明.md) · [全指标汇总](baseline/v3/results/aggregate_metrics.csv) · [模型决定](baseline/v3/results/decisions.csv) · [运行指纹](baseline/v3/results/provenance.json) |
