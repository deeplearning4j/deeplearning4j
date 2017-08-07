---
title: 创建深度学习网络
layout: cn-default
---

# 创建深度学习网络

多层网络由[单层神经网络](./singlelayernetwork.html)堆叠形成。输入层与第一层神经网络和一个[前馈网络](./glossary.html#feedforward)相连。输入层之后的每一个层都将前一层的输出作为输入。

多层网络可接受与单层网络同样类型的输入。多层网络的参数通常也与对应的单层网络相同。

多层网络的输出层通常是一个[逻辑回归分类器](http://en.wikipedia.org/wiki/Multinomial_logistic_regression)，将结果分为零和一。这是一个用于依据深度网络最后一个隐藏层来对输入特征进行分类的判别层。 

多层网络包括以下类型的层：

* *K*个单层网络 

* 一个softmax回归输出层。

### 参数

以下是您在定型一个网络时需要考虑的参数。

### 学习速率 

学习速率又称步幅，是函数在搜索空间中移动的速率。学习速率的值通常在0.001到0.1之间。步幅越小，定型时间越长，但有可能带来更精确的结果。 

### 动量 

动量是另一项决定优化算法向最优值收敛的速度的因素。 

如果您想要加快定型速度，可以提高动量。但定型速度加快可能会降低模型的准确率。 

更深入来看，动量是一个范围在0～1之间的变量，是矩阵变化速率的导数的因数。它会影响权重随时间变化的速率。 

### L2正则化常数 

L2是[这一等式](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm)中的λ。

*预定型步骤*

预定型即是在每一层中通过重构来学习特征，一个层被定型后，会将其输出传递给下一个层。

*微调步骤*

最后，[逻辑回归](http://en.wikipedia.org/wiki/Multinomial_logistic_regression)输出层被定型，然后对每一个层进行反向传播。

多层网络包括以下类型：

* [堆叠式降噪自动编码器](./stackeddenoisingautoencoder.html)
* [深度置信网络](./deepbeliefnetwork.html)
