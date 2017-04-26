---
title: MNIST数据集与受限玻尔兹曼机
layout: cn-default
---

# MNIST数据集与受限玻尔兹曼机

MNIST是一个大型手写数字数据集，用于训练神经网络及其他算法进行图像识别，其定型集有6万幅图像，测试集有1万幅图像。 

MNIST全称为“美国国家标准与技术研究院（NIST）混合数据集”，以NIST的数据集为基础创建。MNIST将NIST数据集中的数万幅手写数字的二进制图像重新排序，使之能更好地服务于图像识别算法的定型与测试。[Yann LeCun的网站](http://yann.lecun.com/exdb/mnist/)详细说明了MNIST优于NIST的原因。

MNIST数据集中的每幅图像为一个28 x 28像素的单元，每个单元周围有四条线段构成的像素边框。图像像素的重心即图像的中心。由DL4J的受限玻尔兹曼机重构后的数字如下图所示： 

![Alt text](../img/mnist_render.png)

以下是神经网络对随机采集的MNIST图像样本进行聚类的示例。

![Alt text](../img/mnist_large.jpg)

对于图像识别的探索可以从MNIST数据集起步。以下是一种加载数据并开始定型网络的简易方法。 

# 教程

首先需要从数据集中获取一幅图像并将其二进制化，把图像的像素从连续灰度数据转换为零和一。一般的规则是，每个灰度值大于35的像素变为1，其他则变为0。这一过程需要使用的工具是MNIST数据集迭代类。

[MnistDataSetIterator](./doc/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.html)可以实现这一功能。

DataSetIterator的使用方法如下：

         DataSetIterator iter = ....;

         while(iter.hasNext()) {
         	DataSet next = iter.next();
         	//对数据集进行操作
         }

在处理原始图像输入时，需要用DataVec的工具来对图像进行规范化、二进制化或缩放处理。DL4J示例中包括一个MNIST图像数据加工管道示例，说明了如何对图像目录进行规范化、标记和预处理。MNIST数据被广泛使用，因此有一个专用的预建迭代器MnistDataSetIterator来完成这些操作。 
         
         //用60000个样例定型，每批次10个样例
         DataSetIterator mnistData = new MnistDataSetIterator(10,60000);

用户可以自行设定批次大小及使用的样例总数。

接下来需要定型受限玻尔兹曼机，使之学会重构MNIST数据集。这可以通过如下代码片段实现：

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNSmallMnistExample.java?slice=33:69"></script>

下面我们将介绍如何定型深度置信网络，使之能[重构并识别MNIST图像](./deepbeliefnetwork.html)。
