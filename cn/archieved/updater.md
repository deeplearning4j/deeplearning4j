---
layout: default
title: Deeplearning4j更新器介绍
---

# Deeplearning4j更新器介绍

本页内容主要面向已经了解[随机梯度下降](./glossary.html#stochasticgradientdescent)原理的读者。

下文介绍的各种更新器之间最主要的区别是对于学习速率的处理方式。 

## 随机梯度下降

![Alt text](../img/updater_math1.png)

`Theta`（θ）是权重，每个theta按其对应的损失函数的梯度进行调整。

`Alpha`（α）是学习速率。如果alpha值很小，那么向最小误差收敛的过程会比较缓慢。如果alpha值很大，模型会偏离最小误差，学习将会停止。

由于定型样例之间的差异，损失函数(L)的梯度在每次迭代后变化很快。请看下图中的收敛路径。更新的步幅很小，在向最小误差逼近的过程中会来回振荡。

![Alt text](../img/updater_1.png)

* [Deeplearning4j中的SGDUpdater](https://github.com/deeplearning4j/deeplearning4j/blob/b585d6c1ae75e48e06db86880a5acd22593d3889/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/SgdUpdater.java)

## 动量

我们用*动量（momentum）*来减少振荡。动量会根据之前更新步骤的情况来调整更新器的运动方向。我们用一个新的超参数`μ`（mu）来表示动量。

![Alt text](../img/updater_math2.png)

后面还会再用到动量的概念。（注意不要将动量与下文中介绍的矩（moment）混淆。）

![Alt text](../img/updater_2.png)

上图为使用了动量的SGD算法。

* [Deeplearnign4j中的Nesterov动量更新器](https://github.com/deeplearning4j/deeplearning4j/blob/b585d6c1ae75e48e06db86880a5acd22593d3889/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/NesterovsUpdater.java)

## Adagrad

Adagrad会根据每个参数对应的历史梯度（之前更新步骤中的情况）来调整该参数的alpha。具体方法是将更新规则中的当前梯度除以历史梯度之和。其结果是，梯度很大时，alpha会减小，反之则alpha增大。

![Alt text](../img/updater_math3.png)

* [Deeplearning4j中的AdaGradUpdater](http://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/AdaGradUpdater.html)

## RMSProp

RMSProp和Adagrad的唯一区别在于`g_t`项的计算方式是对梯度的平均值而非总和进行指数衰减。

![Alt text](../img/updater_math4.png)

此处的`g_t`称为`δL`的二阶矩。此外，还可以引入一阶矩`m_t`。

![Alt text](../img/updater_math5.png)

像第一个例子中那样加入动量……

![Alt text](../img/updater_math6.png)

……最后像第一个例子中一样得到新的`theta`。

![Alt text](../img/updater_math7.png)

* [Deeplearning4j中的RMSPropUpdater](https://github.com/deeplearning4j/deeplearning4j/blob/b585d6c1ae75e48e06db86880a5acd22593d3889/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/updater/RmsPropUpdater.java)

## AdaDelta

AdaDelta同样采用指数衰减的`g_t`平均值，也就是梯度的二阶矩。但它不采用通常作为学习速率的alpha，而是引入`x_t`，即`v_t`的二阶矩。 

![Alt text](../img/updater_math8.png)

* [Deepelearning4j中的AdaDeltaUpdater](http://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/AdaDeltaUpdater.html)

## ADAM

ADAM同时使用一阶矩`m_t`和二阶矩`g_t`，但二者均会随时间衰减。步幅约为`±α`。当我们不断逼近最小误差时，步幅会逐渐缩小。

![Alt text](../img/updater_math9.png)

* [Deeplearning4j中的AdamUpdater](http://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/AdamUpdater.html)
