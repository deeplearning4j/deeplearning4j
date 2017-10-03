---
title: TensorFlow与Deeplearning4j
layout: cn-default
---

# TensorFlow与Deeplearning4j

Tensorflow与Deeplearning4j是互补的。它们不仅都像飞机上的零食一样是免费的，还能在实践中相互配合。 

## 将TensorFlow迁移至Java

Deeplearning4j有一项模型导入功能。目前该功能主要面向的是用Keras创建的TensorFlow模型，到2017年晚些时候将能直接应用于TensorFlow模型。所以，需要在JVM栈上运行模型的TensorFlow用户就多了一种简单易行的方法。Deeplearning4j还让运行较复杂的推断相关任务变得简单；此类任务包括：将一部词典向量化，将词向量存储到索引中，然后对其运行K最近邻算法。 

### <a name="tensorflow">TensorFlow概述</a>

* 谷歌开发了TensorFlow来取代Theano，这两个学习库其实很相似。有一些Theano的开发者在谷歌继续参与了TensorFlow的开发，其中包括后来加入了OpenAI的Ian Goodfellow。 
* 目前**TensorFlow**还不支持所谓的“内联（inline）”矩阵运算，必须要复制矩阵才能对其进行运算。复制非常大的矩阵会导致成本全面偏高。TF运行所需的时间是最新深度学习工具的四倍。谷歌表示正在解决这一问题。 
* 和大多数深度学习框架一样，TensorFlow是用一个Python API编写的，通过C/C++引擎加速。尽管它为一项Java API提供实验性支持，但目前看来不够稳定，我们认为这并不是适合Java和Scala用户群的解决方案。 
* TensorFlow的运行速度[远不如其他一些学习框架](https://arxiv.org/pdf/1608.07249v7.pdf)，比如CNTK和MxNet。 
* TensorFlow的用途不止于深度学习。TensorFlow其实还有支持强化学习和其他算法的工具。
* 谷歌似乎已承认TensorFlow的目标包括招募人才，让其研究者的代码可以共享，推动软件工程师以标准化方式应用深度学习，同时为谷歌云端服务带来更多业务——TensorFlow正是为该服务而优化的。 
* TensorFlow不提供商业支持，而谷歌也不太可能会从事支持开源企业软件的业务。谷歌的角色是为研究者提供一种新工具。 
* 和Theano一样，TensforFlow会生成计算图（如一系列矩阵运算，例如z = sigmoid(x)，其中x和z均为矩阵），自动求导。自动求导很重要，否则每尝试一种新的神经网络设计就要手动编写新的反向传播算法，没人愿意这样做。在谷歌的生态系统中，这些计算图会被谷歌大脑用于高强度计算，但谷歌还没有开放相关工具的源代码。TensorFlow可以算是谷歌内部深度学习解决方案的一半。 
* 从企业的角度看，许多公司需要思考的问题在于是否要依靠谷歌来提供这些工具。 
* 注意：有部分运算在TensorFlow中的运作方式与在NumPy中不同。 

### 利与弊

* (+) Python + NumPy
* (+) 与Theano类似的计算图抽象化
* (+) 编译时间快于Theano
* (+) 用TensorBoard进行可视化
* (+) 同时支持数据并行和模型并行
* (-) 速度比其他框架慢
* (-) 比Torch笨重许多；更难理解
* (-) 已预训练的模型不多
* (-) 计算图纯粹基于Python，因此速度较慢
* (-) 不提供商业支持
* (-) 加载每个新的训练批次时都要跳至Python
* (-) 不太易于工具化
* (-) 动态类型在大型软件项目中容易出错

## TensorFlow的Java API

TensorFlow网站承认，“TensorFlow API的稳定性保证并不适用于TensorFlow Java API。”

如需了解将Java语言用于深度学习的更多信息，请参见我们的[快速入门指南](https://deeplearning4j.org/cn/quickstart)。请与我们一同[抗击“博格人”](https://vimeo.com/84760450)！;)

Deeplearning4j在[多GPU系统](https://github.com/deeplearning4j/dl4j-benchmark)上的运行速度快于TensorFlow。您可以在[此处](https://deeplearning4j.org/cn/benchmark)了解如何运行优化的DL4J基准测试。

Deeplearning4j可以导入通过Keras 1.0在TensorFlow上训练的神经网络模型，用于运行推断。
