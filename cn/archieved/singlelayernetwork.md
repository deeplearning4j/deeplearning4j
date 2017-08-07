---
title: 单层神经网络
layout: cn-default
---

# 单层神经网络

深度学习中的单层神经网络是一种由一个可见的输入层和一个隐藏的输出层组成的网络。 

单层网络的目的（亦即[目标函数](./glossary.html#objectivefunction)）是通过最小化[重构熵](./glossary.html#reconstructionentropy)来学习特征。

这让网络能自动学习输入的特征，进而能更好地找出关联，更加准确地识别判别性特征。在此基础之上，多层网络可以准确地对数据进行分类。单层网络的学习是预定型步骤。

每个单层网络都具备下列属性：

* 隐藏偏差：输出的偏差
* 可见偏差：输入的偏差
* 权重矩阵：网络的权重 

### 单层网络的定型

要定型网络，首先将输入向量导入输入层。再为输入加入一些高斯噪声。噪声函数的类型取决于神经网络的具体情况。然后通过预定型使重构熵最小化，直至网络学会最适合用于重构输入数据的特征。

### 学习速率

学习速率一般在0.001到0.1之间。学习速率又称步幅，是函数在搜索空间中移动的速率。学习速率越低，定型时间越长，但有可能带来更精确的结果。

### 动量

动量是另一项决定优化算法收敛速度的因素。

### L2正则化常数

L2是[这一等式](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm)中的λ。

单层网络包括以下类型：

* [受限玻尔兹曼机](./restrictedboltzmannmachine.html)
* [连续受限玻尔兹曼机](./continuousrestrictedboltzmannmachine.html)
* [降噪自动编码器](./denoisingautoencoder.html)
