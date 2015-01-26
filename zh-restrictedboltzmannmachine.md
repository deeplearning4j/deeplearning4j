---
title: 
layout: zh-default
---

# 受限玻尔兹曼机(RMB)

引用Geoff Hinton(一个谷歌研究员,也是一名大学教授),玻尔兹曼机是“一个对称连接,利用类似神经元作单位来随机决定开关的网络”。(随机的意思是“[随机确定的](http://deeplearning4j.org/glossary.html#stochasticgradientdescent)” )

受限玻尔兹曼机“拥有一层可见的单位和一层隐藏单元,它们是无明显可见的或隐藏的隐藏连接的。”这个“受限”来自它的节点连接的强加限制:层内连接是不允许的,但一个层的每一个节点会连接到的下一个节点,这就是“对称”。

所以, RBM的'节点必须形成一个对称的二分图,数据将从底部的可视层( V0 -V3 )到顶部的隐藏层(H0 -H 2),如下:

![Alt text](../img/bipartite_graph.png)

一个训练有素的限制波尔兹曼机将通过可视层学习输入的数据的结构;它通过一次又一次的数据重建,每一次的重建都会增加它们的相似性度(与原始数据作为基准来比较)。这由RBM重组的数据与原数据的相似性度是使用损失函数来衡量。

RBM对于维度(dimensionality),降维(reduction),分类(classification),协同(collaborative),过滤(filtering),特征(feature),学习(learning)和课题建模(topic modeling)都非常有用。因为RBM的操作非常简单,使用受限玻尔兹曼机为神经网络是我们的第一个选择。

## 参数和K

请参考所有单层网络的共同参数。

变量k是运行对比分歧的次数。每一次的对比分歧运行就如马尔可夫链构(Markov chain)构成限制波尔兹曼机。通常它的值是1 。

## 在Iris启动RBM

