# DT-HUMAN

单人可穿戴活动数字孪生教学原型。基线包括游泳短时心率、时长条件下的设备热量、次日静息心率的对照实验。

公开归档包含源码、实验配置、汇总结果和样本覆盖。原始健康工作簿、逐条结果和含个人时序的生成页面由数据持有方在本地保存。复跑时，将已有的授权工作簿置于 `baseline/data/健康数据.xlsx`，在 `baseline` 目录运行 `py -3.10 -m pip install -r requirements.txt` 和 `py -3.10 -X utf8 build.py`。

基线结果是单人历史回顾评估。热量目标为设备记录，心率目标为后续30秒窗的中位数。

第二版的复跑入口和实验口径见 [实施说明](baseline/v2/README.md)，效果见 [第二版汇总](baseline/v2/results/summary.json)。另附 [旧版汇总](baseline/results/summary.json)、[样本覆盖](baseline/data/coverage.csv) 和 [早期模型对照](baseline/results/previous_model_comparisons.json)。

## 阶段回溯

| 阶段 | 快照 |
|---|---|
| 01 · V1 基线 | [初版归档提交](https://github.com/DengBuyun1/DT-HUMAN/commit/1709d4b21d564c40b020bdb635f99719410b4172) |
| 02 · V2 对照与回放 | [第二版归档提交](https://github.com/DengBuyun1/DT-HUMAN/commit/2578f2bd21b0bbc2f082757af5dd42283ba91566) |
| 03 · 连续性与固定半段实验 | [阶段 03 提交](https://github.com/DengBuyun1/DT-HUMAN/commit/9b1fde03fa9b8962dff80bf21e870bb223ce909f) · [实验说明与复跑](baseline/v2/results/continuous_half.md) · [汇总结果](baseline/v2/results/continuous_half_summary.json) · [文件指纹](baseline/v2/results/continuous_half_manifest.json) |
| 04 · V3.0 方案定稿 | [阶段定稿提交](https://github.com/DengBuyun1/DT-HUMAN/commit/9bbcf572b56fa6ca7920c3cd13adb93f05905697) · [版本索引](baseline/docs/plans/00_方案版本与执行顺序索引.md) · [详细方案](baseline/docs/plans/04_V3.0_数据口径修正与受控预测优化实施方案_2026-11-09.md) · [方案指纹](baseline/docs/plans/plan_manifest.json) |
| 04 · V3.0 本地实施与实验 | [复跑说明](baseline/v3/README.md) · [修订算法说明](baseline/docs/experiments/V3.0_阶段04实验与算法说明.md) · [全指标汇总](baseline/v3/results/aggregate_metrics.csv) · [模型决定](baseline/v3/results/decisions.csv) · [运行指纹](baseline/v3/results/provenance.json) |

本分支归档阶段 04 的 V3.0 实施和实验结果。公开版保留可审阅源码、合成行为测试、汇总结果和指纹。完整数据复跑使用授权的本地归档。
