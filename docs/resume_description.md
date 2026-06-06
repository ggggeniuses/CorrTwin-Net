# Resume Description

## Short Version

CorrTwin-Net: 面向无线信道归一化 ensemble 相关幅值的深度学习代理建模项目。基于 Python/PyTorch 构建 Rayleigh/Rician 多径信道统计标签生成流程，使用多个独立信道 realization 估计 clean-channel `|ACF|/|CCF|/|FCF|`，并训练 MLP/ResMLP 模型进行快速预测。

## Three Resume Bullets

- 构建 clean-channel 无线信道统计数据生成流程，显式区分 Rayleigh/Rician 信道，修正 PDP、Rician K 因子功率归一化和 ensemble 相关函数估计。
- 基于 PyTorch 实现 MLP/ResMLP 代理模型，并加入 Mean Curve、Channel-Type Mean、Ridge、RandomForest、KNN 等 baseline 进行系统对比。
- 设计 actual-generator 理论校验、OOD 数据审计、完整 ensemble 仿真速度对比和 artifact 哈希校验，提升项目可复现性与物理可信度。

## Interview Explanation

这个项目不是做常见的 CSI 信道估计，而是关注无线信道统计特性的代理建模。传统参数扫描需要反复运行多 realization 信道仿真来估计 ACF/CCF/FCF 等统计曲线。我用深度学习模型学习场景参数到 clean-channel ensemble 相关幅值曲线的映射，在离线训练后用于快速参数扫描和敏感性分析。

## English Version

CorrTwin-Net: Built a development-preview wireless channel statistics surrogate-modeling pipeline in Python/PyTorch. The project generates clean Rayleigh/Rician ensemble `|ACF|/|CCF|/|FCF|` labels, trains MLP/ResMLP surrogate models, compares them with mean, Ridge, RandomForest, and KNN baselines, and evaluates theory validation, OOD audits, inference speed, and artifact reproducibility.
