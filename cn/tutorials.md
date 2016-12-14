---
title: 深度学习教程
layout: cn-default
---

# 深度学习教程

欢迎来到教程主页。以下的教程专门为刚开始接触深度学习和DeepLearning4J的用户介绍相关概念，还会提供图像识别、文本处理、分类的示例。部分示例会在提供代码的同时给出文字介绍，其他一些示例则是代码编写过程的屏幕录像，配以口头说明。祝您学习愉快！

## 基础神经网络

### MNIST基础教程

MNIST在机器学习领域是相当于“Hello World”的入门示例。本教程将向您介绍如何用MNIST数据集来定型一个单层神经网络。

[阅读教程](http://deeplearning4j.org/zh-mnist-for-beginners.html)<br>
[直接看代码](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java)

### MNIST高级教程

如果采用更复杂的Lenet算法来处理MNIST数据集，可以达到99%的准确率。Lenet是一种深度卷积网络。

[直接看代码](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java)

### Word2Vec教程

Word2vec是一种比较流行的自然语言算法，能创建可以输入深度神经网络的神经词向量。 

[阅读教程](./word2vec)<br>
[直接看代码](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec/Word2VecRawTextExample.java)

### 线性分类器教程

本教程将向您介绍如何用多层感知器进行线性分类。 

[直接看代码](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java)

以下是由Tom Hanlon主讲的机器学习视频讲座。在这一系列的基础教程中，Tom会讲解如何构建简单的神经网络。下面这段屏幕录像主要介绍怎样用Deeplearning4j构建一个线性分类器。

<iframe width="560" height="315" src="https://www.youtube.com/embed/8EIBIfVlgmU" frameborder="0" allowfullscreen></iframe>

### 异或门教程

异或门（XOR）是用于实现异或逻辑的数字逻辑门。“异或”指有且仅有一个输入值为真时，门的输出值为真，即1。

[直接看代码](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/xor/XorExample.java)

### DataVec + Spark教程

本教程主要介绍如何用Skymind的DataVec来从文本文件中摄取逗号分隔值，如何在Spark中用DataVec转换进程（DataVec Transform Process）将这些字段转换为数值形式，以及如何保存修改后的数据。将非数值数据转换为数值形式是用神经网络分析数据的关键预备步骤之一。

<iframe width="560" height="315" src="https://www.youtube.com/embed/MLEMw2NxjxE" frameborder="0" allowfullscreen></iframe>

### 图像数据加工管道教程

#### 图像数据摄取与标记
本教程包括一系列视频和代码示例，介绍如何建立一个完整的数据加工管道。 

第一个示例演示了如何使用DataVec的一些工具来读取目录中的图像，再用ParentPathLabelGenerator为图像生成一个标签。数据读取并标记完毕后，将图像数据的像素值范围从0～255缩放至0～1。 

<iframe width="560" height="315" src="https://www.youtube.com/embed/GLC8CIoHDnI" frameborder="0" allowfullscreen></iframe>


#### 添加神经网络

本教程在图像数据摄取与标记教程的基础上介绍如何为DataVec图像数据加工管道添加一个神经网络，用加载的图像将其定型。具体内容包括MultiLayerNetwork、DataSetIterator、网络定型以及在定型过程中监测模型的表现。 

<iframe width="560" height="315" src="https://www.youtube.com/embed/ECA6y6ahH5E" frameborder="0" allowfullscreen></iframe>

#### 保存和加载已定型网络

网络定型完毕后，您可能会需要保存已定型的网络，以便将来构建应用程序时使用。本教程将会演示已定型模型的保存和加载方法。 

<iframe width="560" height="315" src="https://www.youtube.com/embed/zrTSs715Ylo" frameborder="0" allowfullscreen></iframe>

#### 用自选图像测试已定型网络

网络定型及测试完毕后，就可以将它部署到应用程序中了。本教程将演示如何加载一个定型的模型并添加简易的filechooser界面，以便让模型推测用户输入的图像是哪个数字。在以下的视频中，我们会用一张在谷歌上搜索到的数字3的图像来测试一个用MNIST手写数字数据集定型的前馈神经网络。 

<iframe width="560" height="315" src="https://www.youtube.com/embed/DRHIpeJpJDI" frameborder="0" allowfullscreen></iframe>



### LSTM和循环网络基础教程

循环网络是一类人工神经网络，用于识别诸如文本、基因组、手写字迹、语音等序列数据的模式，或用于识别传感器、股票市场、政府机构产生的数值型时间序列数据。

长短期记忆单元机制让神经网络可以根据经验来学习如何对时间序列事件进行分类、处理和预测。

[阅读教程](https://deeplearning4j.org/cn/lstm.html)

### 通过DL4J使用循环网络

进一步深入探讨循环神经网络（RNN），包括在DL4J中配置使用RNN的方法

[阅读教程](https://deeplearning4j.org/cn/usingrnns)


### <a name="beginner">其他Deeplearning4j教程</a>

* [神经网络简介](./neuralnet-overview)
* [受限玻尔兹曼机](./restrictedboltzmannmachine)
* [本征向量、协方差、PCA和熵](./eigenvector)
* [LSTM和循环网络](./lstm)
* [神经网络与回归分析](./linear-regression)
* [卷积网络](./convolutionalnets)
* [神经词向量、Word2vec与Doc2vec](./word2vec)
