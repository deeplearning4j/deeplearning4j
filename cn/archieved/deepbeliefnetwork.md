---
title: Java环境中的深度置信网络
layout: cn-default
---

#深度置信网络与MNIST数据集教程

深度置信网络（DBN）可以定义为一系列堆叠起来的[受限玻尔兹曼机（RBM，介绍见此处）](./restrictedboltzmannmachine.html)，每个RBM层都与其前后的层进行通信。单个层中的节点之间不会横向通信。 

深度置信网络可以直接用于处理无监督学习中的未标记数据聚类问题，也可以在RBM层的堆叠结构最后加上一个[Softmax](./glossary.html#softmax)层来构成分类器。 

除了第一个和最后一个层，深度置信网络中的每一层都扮演着双重角色：既是前一层节点的隐藏层，也是后一层节点的输入（或“可见”）层。深度置信网络是由多个单层网络组成的。 

深度置信网络常用于图像、视频序列和动作捕捉数据的识别、聚类与生成。连续型深度置信网络是深度置信网络的一种扩展，它接受连续的十进制数据，而非二进制数据。此类网络由[Geoff Hinton和他的学生于2006年](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)提出。

### 用深度置信网络处理MNIST数据集

对于图像识别和DBN的探索可以从MNIST数据集开始。第一步是从数据集中提取一幅图像并将其二进制化，即把图像的像素从连续的灰度数据转换为零和一。每个灰度值大于35的像素变为1，其他则变为0。MNIST数据集迭代器类可以完成这一操作。

### 超参数

所有多层网络通用的参数[见此处](./neuralnet-configuration)。

变量k表示[对比散度算法](./glossary.html#contrastivedivergence)的运行次数。对比散度算法每运行一次，就是一个马尔可夫链的样本。构成一个深度置信网络时，k的值通常为`1`。

### 深度置信网络的初始化

初始化操作由DataSetIterator类的具体形式**MnistDataSetIterator**完成。 

我们通常用DataSetIterator来处理输入数据，完成数据集所特有的二进制化和标准化等操作。处理MNIST数据集时，以下这行代码将指定批次大小和样例数量——用户通过这两项参数来设定定型所使用的样例有多少（样例更多一般会提高模型的准确率，也会增加定型时间）：
         
         //用60000个样例定型，每批次100个样例
         DataSetIterator iter = new MnistDataSetIterator(100,60000);

用`MnistDataSetIterator`遍历输入数据。（完整的示例参见[此处](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DeepAutoEncoderExample.java)，您也可以下载[DL4J的示例库](https://github.com/deeplearning4j/dl4j-examples/)来获取这个例子。）

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DeepAutoEncoderExample.java?slice=42:45"></script>

用`MultiLayerConfiguration`设置由RBM层组成的DBN网络：

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DeepAutoEncoderExample.java?slice=41:60"></script>

调用`fit`方法，定型模型：

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DeepAutoEncoderExample.java?slice=60:70"></script>

注意此处将*eval*类与[混淆矩阵](./glossary.html#confusionmatrix)和F1值结合使用，实现可视化的数据评估。这可以用于监测网络定型表现随时间变化的情况。 

F1值以百分比形式表示，基本上相当于神经网络作出准确预测的概率。要提高神经网络的性能，您可以对网络进行调试，改变隐藏层的数量和规模，调整学习速率、动量、权重分布、正则化方法等其他参数。

接下来我们要介绍如何用[分布式和多线程计算](./iterativereduce)来提高网络的定型速度。如果您想了解另一类深度网络——深度自动编码器，请[点击此处](./deepautoencoder)。 

### <a name="beginner">其他Deeplearning4j教程</a>
* [神经网络简介](./neuralnet-overview)
* [LSTM和循环网络](./cn/lstm)
* [Word2vec](./word2vec)
* [受限玻尔兹曼机](./restrictedboltzmannmachine)
* [本征向量、协方差、PCA和熵](./cn/eigenvector)
* [神经网络与回归分析](./linear-regression)
* [卷积网络](./convolutionalnets)
