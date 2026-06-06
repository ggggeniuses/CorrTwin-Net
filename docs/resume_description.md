# Resume Description

## 中文简历版

CorrTwin-Net: 基于 Python/PyTorch 构建无线信道统计特性代理建模流程，生成 Rayleigh/Rician clean-channel ensemble `|ACF|/|CCF|/|FCF|` 标签，并训练 MLP/ResMLP 进行快速曲线预测。

## 推免简历要点

- 设计多 realization 信道统计标签生成流程，修正 Rician 功率归一化、PDP 采样和 ensemble 相关函数聚合定义。
- 实现 MLP/ResMLP 与 Mean、Ridge、RandomForest、KNN baseline 对比，在 formal P0 设置下完成精度、速度和 OOD 泛化评估。
- 增加 actual-generator 理论校验、OOD 数据审计、artifact 哈希清单和 GitHub Actions CI，提高项目可复现性。

## English Version

CorrTwin-Net: Built a Python/PyTorch wireless-channel statistics surrogate pipeline that generates clean Rayleigh/Rician ensemble `|ACF|/|CCF|/|FCF|` labels, trains MLP/ResMLP predictors, and evaluates accuracy, speed, OOD behavior, theory sanity checks, and artifact reproducibility.
