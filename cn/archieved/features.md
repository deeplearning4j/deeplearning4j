---
title: Deeplearning4j的功能
layout: cn-default
---

# 功能

以下是Deeplearning4j功能的不完全列表。我们会不断添加新推出的网络类型和工具。 

### 集成

* Spark
* Hadoop/YARN
* 导入Keras模型

### API

* Scala
* Java 


### 库

* [ND4J：面向JVM的N维数组](http://nd4j.org)
* [libND4J：ND4J的原生CPU/GPU操作](https://github.com/deeplearning4j/libnd4j)
* [DataVec：DL4J的数据准备](https://github.com/deeplearning4j/DataVec)
* [Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)

### 网络类型

* [受限玻尔兹曼机](./restrictedboltzmannmachine.html)
* [卷积网络](./convolutionalnets.html)
* [递归自动编码器](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/models/featuredetectors/autoencoder/recursive/RecursiveAutoEncoderTest.java)
* [循环网络：长短期记忆（LSTM）](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/models/classifiers/lstm/LSTMTest.java)（包括双向LSTM）
* [深度置信网络](./deepbeliefnetwork.html)
* [降噪及堆叠式降噪自动编码器](./denoisingautoencoder.html)
* [深度自动编码器](./deepautoencoder.html)

Deeplearning4j是一个可组合的框架，用户可以将浅层网络自行组合成不同类型的深度网络。比如，2014年晚些时候，谷歌将卷积网络和循环网络结合，组成了能够准确生成图像描述的神经网络。

### 工具

DL4J包含下列内置向量化算法：

* [DataVec：数据向量化的“罗塞塔石碑”](https://github.com/deeplearning4j/DataVec)
* 用于图像的移动窗口
* 用于文本的移动窗口 
* 用于序列分类的Viterbi算法
* [Word2Vec](./word2vec.html)
* [词数统计和TF-IDF的词袋编码](./bagofwords-tf-idf.html)
* [Doc2Vec与段向量](https://deeplearning4j.org/doc2vec)
* 语法成分解析
* [DeepWalk](http://arxiv.org/abs/1403.6652)

DL4J支持下列优化算法：

* 随机梯度下降
* 带线搜索的随机梯度下降
* 共轭梯度线搜索（参见[Hinton 2006](http://www.cs.toronto.edu/~hinton/science.pdf)）
* L-BFGS

上述每种优化算法均可与下列定型功能（DL4J称之为“更新器”）搭配使用：

* SGD（仅与学习速率相关）
* Nesterov动量
* Adagrad
* RMSProp
* Adam
* AdaDelta

### 超参数

* 丢弃法（dropout，随机省略特征检测器来防止过拟合）
* 稀疏性（sparsity，稀疏/罕见输入的强制激活）
* Adagrad（针对具体特征的学习速率优化）
* L1和L2正则化（权重衰减）
* 权重变换（用于深度自动编码器）
* 初始权重生成的概率分布操控
* 梯度标准化与裁剪

### 损失/目标函数

* MSE：均方差：线性回归
* EXPLL：指数型对数似然函数：泊松回归
* XENT：叉熵二元分类
* MCXENT：多类别叉熵
* RMSE_XENT：RMSE叉熵
* SQUARED_LOSS：平方损失
* NEGATIVELOGLIKELIHOOD：负对数似然函数

### 激活函数 

ND4J中定义的激活函数参见[此处](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms)

* ReLU（修正线性激活函数）
* 带泄露的ReLU
* Tanh
* Sigmoid
* Hard Tanh
* Softmax
* 恒等函数
* [ELU](http://arxiv.org/abs/1511.07289)：指数型线性单元
* Softsign
* Softplus
