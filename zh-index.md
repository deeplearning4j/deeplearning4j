---
title: 
layout: zh-default
---

# Deeplearning4j是什么？

Deeplearning4j（简称DL4J）是为Java和Scala编写的首个商业级开源分布式深度学习库。DL4J与Hadoop和[Spark](../spark.html)集成，为商业环境（而非研究工具目的）所设计。[Skymind](http://skymind.io)是DL4J的商业支持机构。

Deeplearning4j技术先进，以即插即用为目标，通过更多预设的使用，避免太多配置，让非研究人员也能够进行快速的原型制作。DL4J同时可以规模化定制。DL4J遵循Apache 2.0许可协议，一切以其为基础的衍生作品均属于衍生作品的作者。

您可以根据我们[在快速入门页上的说明](../zh-quickstart.html)，在几分钟内运行您首个定型神经网络示例。

### [神经网络使用情景](../use_cases.html)

* [人脸／图像识别](../facial-reconstruction-tutorial.html)
* 语音搜索
* 文本到语音（转录）
* 垃圾邮件筛选（异常情况探测）
* 欺诈探测 
* 推荐系统（客户关系管理、广告技术、避免用户流失）
* [回归分析](../linear-regression.html)

### 为何选择Deeplearning4j？ 

* 功能多样的[N维数组](http://nd4j.org/zh-getstarted)类，为Java和Scala设计
* 与[GPU](http://nd4j.org/gpu_native_backends.html)集合
* 可在[Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn)、[Spark](../gpu_aws.html)上实现扩缩
* [Canova](../canova.html)：机器学习库的通用向量化工具
* [ND4J：线性代数库，较Numpy快一倍](http://nd4j.org/benchmarking)

Deeplearning4j包括了分布式、多线程的深度学习框架，以及普通的单线程深度学习框架。定型过程以集群进行，也就是说，Deeplearning4j可以快速处理大量数据。神经网络可通过[迭代化简]平行定型，与**Java**、**[Scala](http://nd4j.org/scala.html)**和**[Clojure](https://github.com/wildermuthn/d4lj-iris-example-clj/blob/master/src/dl4j_clj_example/core.clj)**均兼容。Deeplearning4j在开放堆栈中作为模块组件的功能，使之成为首个为[微服务架构](http://microservices.io/patterns/microservices.html)打造的深度学习框架。

### DL4J神经网络

* [受限玻尔兹曼机](../zh-restrictedboltzmannmachine.html)
* [卷积网络](../zh-convolutionalnets.html) （图像）
* [递归网络](../usingrnns.html)/[LSTMs](../lstm.html)（时间序列和传感器数据）
* [递归自动编码器](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/feedforward/autoencoder/recursive/RecursiveAutoEncoder.java)
* [深度置信网络](../deepbeliefnetwork.html)
* [深度自动编码器](http://deeplearning4j.org/deepautoencoder.html)（问－答／数据压缩）
* 递归神经传感器网络（场景、分析）
* [堆叠式降噪自动编码器](../stackeddenoisingautoencoder.html)
* 更多用途请参见[《如何选择神经网络》](../neuralnetworktable.html)

深度神经网络能够实现[前所未有的准确度](../accuracy.html)。对神经网络的简介请参见[概览](../neuralnet-overview.html)页。简而言之，Deeplearning4j能够让你从各类浅层网络（其中每一层在英文中被称为`layer`）出发，设计深层神经网络。这一灵活性使用户可以根据所需，在分布式、生产级、能够在分布式CPU或GPU的基础上与Spark和Hadoop协同工作的框架内，整合受限玻尔兹曼机、其他自动编码器、卷积网络或递归网络。

此处为我们已经建立的各个库及其在系统整体中的所处位置：

![Alt text](../img/schematic_overview.png)

在定型深度学习网络的过程中，有许多可供调节的参数。我们已尽可能对这些参数进行解释，从而使Deeplearning4j能够成为Java、[Scala](https://github.com/deeplearning4j/nd4s)和[Clojure](https://github.com/whilo/clj-nd4j)编程人员的DIY工具。

如果您有任何问题，请[在Gitter上加入我们](https://gitter.im/deeplearning4j/deeplearning4j)；如果需要高级支持，则[请与Skymind联系](http://www.skymind.io/contact/)。[ND4J是基于Java的科学运算引擎](http://nd4j.org/)，用来驱动矩阵操作。在大型矩阵上，我们的基准显示ND4J[较Numpy运算速度快大约一倍](http://nd4j.org/benchmarking)。

### Deeplearning4j教程

* [深度神经网络简介](../neuralnet-overview.html)
* [卷积网络教程](../zh-convolutionalnets.html)
* [LSTM和递归网络教程](../lstm.html)
* [通过DL4J使用递归网络](../usingrnns.html)
* [深度置信网络和MNIST](../mnist-tutorial.html)
* [鸢尾花数据组教程](../iris-flower-dataset-tutorial.html)
* [针对LFW人脸图像数据集进行人脸重构](../facial-reconstruction-tutorial.html)
* [通过Canova库自定义数据准备工作](../image-data-pipeline.html)
* [受限玻尔兹曼机](../zh-restrictedboltzmannmachine.html)
* [本征向量、PCA和熵](../eigenvector.html)
* [深度学习词汇表](../glossary.html)

### 用户反馈

      “我感觉自己像是弗兰肯斯坦。像是小说里的弗兰肯斯坦博士……”――史蒂夫・D． 
      
      “我对在生产中使用Deeplearning4j非常感兴趣。这里蕴藏着价值百亿英镑的巨大商机。”－约翰・M．

### 为Deeplearning4j做出贡献

想要为Deeplearning4j作出贡献的开发人员可先阅读[开发人员指南](../devguide.html)。

### 用Deeplearning4j进行研究

* 斯坦福NLP：“[大规模语言分类](http://nlp.stanford.edu/courses/cs224n/2015/reports/24.pdf)”
