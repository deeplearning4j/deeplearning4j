---
title: Deeplearning4j路线图
layout: cn-default
---

# Deeplearning4j路线图

基于对客户和开源社区成员需求的认识，Skymind设定了下列开发重点。我们欢迎贡献者按自定的优先顺序添加功能。 

优先级最高：

* 为ND4J重写CUDA（进行中）
* CPU优化（C++后端）
* 超参数优化（进行中，基础工作已完成：[Arbiter](https://github.com/deeplearning4j/Arbiter)）
* 参数服务器
* ND4J的稀疏支持
* 网络定型表现测试：对比其他平台（必要之处进行优化）
* Spark性能测试：对比本地（同上）
* 大规模构建示例

优先级中等：

* ND4J的OpenCL
* CTC循环神经网络（用于语音等）

锦上添花：

* 自动微分
* ND4J的复数支持（+优化）
* 强化学习
* Python支持/接口
* 集成（ensemble）学习支持
* 变分自动编码器
* 生成式对抗模型

优先级较低：

* 无Hessian矩阵（Hessian-free）优化算法
* 其他类型的循环神经网络：多维度；注意力模型、神经图灵机等
* 3D卷积神经网络

本页内容将持续更新。上次更新日期为2016年2月27日。
