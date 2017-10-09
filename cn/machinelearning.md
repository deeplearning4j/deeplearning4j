---
title: 基于Deeplearning4j的机器学习
layout: cn-default
---

# 基于Deeplearning4j的机器学习

Deeplearning4j及其配套开源库（[ND4J](http://nd4j.org/)、DataVec、Arbiter等）主要用于实现可扩展的深度人工神经网络，但开发者也可以用我们的框架来构建更传统的机器学习算法。

## DL4J提供的算法

* 线性回归
* 逻辑回归
* [K均值聚类](https://deeplearning4j.org/doc/org/deeplearning4j/clustering/kmeans/package-tree.html )
* K最近邻（k-NN）
* 基于[VP树](https://en.wikipedia.org/wiki/Vantage-point_tree)、[t-SNE](https://lvdmaaten.github.io/tsne/)和四叉树的K最近邻优化

## ND4J可实现的算法

ND4J是一个通用张量库，因此在可实现的算法种类上没有任何限制。 

我们正在与李海峰的SMILE（[Statistical Machine Intelligence and Learning Engine](http://haifengl.github.io/smile/)）进行集成。SMILE可实现包括随机森林和GBM在内的一百多种统计和机器学习算法，是我们所知的性能表现最好的开源JVM机器学习库。 
