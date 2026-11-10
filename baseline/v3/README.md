# V3.0 阶段 04 实验归档

正式运行编号为 `v3-20261110-141526-382818`，独立复跑编号为 `v3-20261110-141526-390329`。

- [实验与算法说明](../docs/experiments/V3.0_阶段04实验与算法说明.md)
- [全指标汇总](results/aggregate_metrics.csv) · [模型决定](results/decisions.csv) · [覆盖统计](results/coverage.csv)
- [核心结果](results/summary.json) · [复跑与源码指纹](results/provenance.json)

在 `baseline/` 运行合成行为测试。

```powershell
py -3.10 -m pip install -r requirements.txt matplotlib==3.10.6
py -3.10 -X utf8 -m unittest discover -s tests/v3 -v
```

公开源码的训练、预测和评分逻辑与本地运行一致。公开版用相对日序说明切分，并从入口和测试移除真实小睡样例。具体改动和本地原源码指纹见运行来源文件。

真实数据复跑使用授权本地完整归档，包括工作簿、冻结方案原件、原始协议、原始源码及历史父运行目录。解压后在其 `DTs/` 目录运行 `py -3.10 -X utf8 -m v3.run --legacy-repeat "runs/连续半段-20261110-140414-192689"`。公开分支主要用于审阅算法和汇总结果。
