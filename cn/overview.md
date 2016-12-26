# DeepLearning4J是什么？

DeepLearning4J（DL4J）是一套基于Java语言的神经网络工具包，可以构建、定型和部署神经网络。 

### DeepLearning4J的组件
 
DeepLearning4J包括以下各个子项目。 

* **DataVec**进行数据摄取，将数据标准化并转换为特征向量
* **DeepLearning4J**提供配置神经网络、构建计算图的工具
* **Keras Model Import**（Keras模型导入）帮助用户将已定型的Python和Keras模型导入DeepLearning4J和Java环境。 
* **ND4J**让Java能够访问所需的原生库，使用多个CPU或GPU快速处理矩阵数据。 
* **DL4J-Examples**（DL4J示例）包含图像、时间序列及文本数据分类与聚类的工作示例。
* **ScalNet**是受Keras启发而为Deeplearning4j开发的Scala语言包装。它通过Spark在多个GPU上运行。
* **RL4J**用于在JVM上实现深度Q学习、A3C及其他强化学习算法。
* **Arbiter**帮助搜索超参数空间，寻找最理想的神经网络配置。

-------------------------

## DataVec

数据的摄取、清理、联接、缩放、标准化和转换是开展任何类型的数据分析时都必须完成的工作。这类工作可能有些无趣，但却是深度学习的先决条件。DataVec是专为这一流程设计的工具包。数据科学家和开发人员可以用其中的工具将图像、视频、声音、文本和时间序列等原始数据转变为特征向量，输入神经网络。 

### Github代码库

DataVec的Github代码库请见[此处](https://github.com/deeplearning4j/datavec)。代码库的结构如下。

* [datavec-dataframe](https://github.com/deeplearning4j/DataVec/tree/master/datavec-dataframe)：相当于Pandas Dataframe的DataVec内置工具
* [datavec-api](https://github.com/deeplearning4j/DataVec/tree/master/datavec-api)：数据预处理和数据加工管道定义的规则。
* [datavec-data](https://github.com/deeplearning4j/DataVec/tree/master/datavec-data)：让系统知道如何理解声音、视频、图像、文本数据类型
* [datavec-spark](https://github.com/deeplearning4j/DataVec/tree/master/datavec-spark)：在Spark上运行分布式数据加工管道 
* [datavec-local](https://github.com/deeplearning4j/DataVec/tree/master/datavec-local)：在桌面上独立运行DataVec。用于推断。 
* [datavec-camel](https://github.com/deeplearning4j/DataVec/tree/master/datavec-camel)：与外部Camel组件连接。Camel让您能定义路由，同多个数据来源进行集成。datavec-camel可以将DataVec作为目的地，把数据从任何您所指定的Camel来源发送至DataVec。

### DataVec示例

我们在Github上的示例代码库中提供DataVec示例，详见[此处](https://github.com/deeplearning4j/dl4j-examples)。

相关示例的简介汇总请见[此处](examples-tour)。

### JavaDoc

DataVec的JavaDoc请见[此处](../datavecdoc/)。 

### DataVec概述

神经网络专门处理多维数组形式的数值数据。DataVec可以将来自一个CSV文件或一批图像的数据序列化，转换为数值数组。 

### DataVec：常用类

以下是一些重要的DataVec类：

* Input Split

将数据分为测试集和定型集

* **InputSplit.sample**将数据分为测试集和定型集

数据随机化

* **FileSplit.random**用于数据随机化

数据读取和序列化的基类。RecordReader摄取输入数据，返回一个可序列化对象（Writable）的列表。 

* **RecordReader**

具体的RecordReader实现类

* **CSVRecordReader**用于处理CSV数据
* **CSVNLinesSequenceRecordReader**用于处理序列数据
* **ImageRecordReader**用于处理图像
* **JacksonRecordReader**用于处理JSON数据
* **RegexLineRecordReader**用于解析日志文件
* **WavFileRecordReader**用于处理音频文件
* **LibSvmRecordReader**用于支持向量机
* **VideoRecordReader**用于读取视频

数据的重新组织、联接、标准化和转换。 

* **Transform**

具体的Transform实现类

* **CategoricalToIntegerTransform**用于将类别名称转换为整数
* **CategoricalToOneHotTransform**将类别名称转换为one-hot表示
* **ReorderColumnsTransform**对列进行重排
* **RenameColumnsTransform**对列进行重命名
* **StringToTimeTransform**转换时间字符串

输入数据的标签可以基于图像存储的目录位置。 

* **ParentPathLabelGenerator**基于父目录的标签
* **PatternPathLabelGenerator**生成基于文件路径内一个字符串的标签

数据标准化

* **Normalizer**：虽然属于ND4J，但这里也应当提一下

-------------------------

# DeepLearning4J

DeepLearning4J用于设计神经网络

### Github代码库

DeepLearning4J的Github代码库请见[此处](http://github.com/deeplearning4j/deeplearning4j)。代码库的结构如下。

* [deeplearning4j-core](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core)：包括datasetiterators和在桌面上运行DL4J所需的全部代码。 
* [deeplearning4j-cuda](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-cuda)：包括cudnn和所有同cuda相关的代码。
* [deeplearning4j-graph](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-graph)用于deepwalk的图像处理。
* [deeplearning4j-modelimport](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport)：导入来自Keras的神经网络模型，可以借此导入Theano、Tensorflow、Caffe、CNTK等主流学习框架的模型
* [deeplearning4j-nlp-parent](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nlp-parent)：英语、日语、韩语文本分析以及UIMA等工具集的外部分词器和插件。UIMA原本即包括依赖解析、语义角色标记、关系提取和QA系统等功能。我们与UIMA等工具集进行集成，将数据传递给word2vec。
* [nlp](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nlp-parent/deeplearning4j-nlp)：Word2vec、Doc2vec及其他相关工具。
* [deeplearning4j-nn](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nn) ：精简版的神经网络领域专用语言（DSL），依赖项较少。可配置多层网络，用一种构建器模式来设置超参数。
* [deeplearning4j-scaleout](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout)：AWS预配置，用parallelwrapper进行桌面参数平均化（单机96核），不希望运行Spark时可不运行；一个版本采用参数服务器，另一个不用；streaming文件夹是基于Kafka和Spark的流式数据处理；spark文件夹是基于Spark的网络定型和自然语言处理：分布式Word2Vec
* [deeplearning4j-ui-parent](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-ui-parent)：神经网络定型的启发式评估和可视化

### JavaDoc 

DeepLearning4J的JavaDoc可在[此处](http://deeplearning4j.org/doc/)获取。

### DeepLearning4J示例

DeepLearning4J示例的Github代码库请见[此处](https://github.com/deeplearning4j/dl4j-examples)。 

相关示例的简介汇总请见[此处](examples-tour)。

### DeepLearning4J的常用类

* **MultiLayerConfiguration** 配置一个网络
* **MultiLayerConfiguration.Builder**配置网络的构建器接口
* **MultiLayerNetwork**按配置构建网络
* **ComputationGraph**构建计算图式网络
* **ComputationGraphConfiguration**计算图配置
* **ComputationGraphConfiguration.GraphBuilder**计算图配置的构建器接口
* **UiServer**添加一个基于网页的GUI界面，以便查看定型参数进展以及网络配置

## 模型导入

如果您曾经用过Python的深度学习库Keras，希望将已定型的模型或模型配置导入DeepLearning4J，那么请使用“模型导入”功能。 

### Github代码库

模型导入其实是DeepLearning4J的一部分，但很有必要将其单独列为一段。Github文件夹请见[此处](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport)。

### 模型导入示例

我们将在[此处](https://github.com/deeplearning4j/dl4j-examples/)添加示例

### 模型导入的常用类

* 可以用**KerasModel.Import**将已保存的Keras模型导入至DeepLearning4J的MultiLayerNetwork或计算图中

### 视频 

下面的视频介绍了如何将Keras模型导入DL4J：

<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

-----------

## ND4J

ND4J是DeepLearning4J的数值处理库和张量库，在JVM中实现NumPy的功能。  

### Github代码库

ND4J的Github代码库请见[此处](http://github.com/deeplearning4j/nd4j)。ND4J是用于处理n维数组（NDArrays）的DSL。

* [nd4j-parameter-server-parent](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-parameter-server-parent)：用于分布式神经网络定型的强大的参数服务器，基于Aeron。
* [nd4j-backends](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends)：针对不同硬件的后端，为使用多个GPU和CPU运行而优化。

### JavaDoc 

ND4J的JavaDoc可在[此处](http://nd4j.org/doc/)获取。

### ND4J示例

ND4J示例请见[此处](https://github.com/deeplearning4j/dl4j-examples/tree/master/nd4j-examples)。

### ND4J的常用类

您可能不会直接使用下面的这些类，而是在配置神经网络时用到。在后台，您为OptimizationAlgorithm（优化算法）、Updater（更新器）、LossFunction（损失函数）设定的配置都是在ND4J之中完成的。

* **DataSetPreProcessor**：图像或数值数据标准化的工具
* **BaseTransformOp**：各类激活函数，包括tanh、sigmoid、relu、Softmax……
* **GradientUpdater**：随机梯度下降、AdaGrad、Adam、Nesterov……

-------------------------

## ScalNet

ScalNet针对Scala语言开发，功能相当于Keras。它是DeepLearning4J的Scala语言包装，可以在多个GPU上运行Spark。

### Github代码库

* [Github上的ScalNet代码库](https://github.com/deeplearning4j/ScalNet)

## RL4J 

RL4J是在Java中实现深度Q学习、A3C及其他强化学习算法的库和环境，与DL4J和ND4J相集成。 

### Github代码库

* [RL4J](https://github.com/deeplearning4j/rl4j)
* [Gym集成](https://github.com/deeplearning4j/rl4j/tree/master/rl4j-gym)
* [RL4J玩《Doom》](https://github.com/deeplearning4j/rl4j/tree/master/rl4j-doom)

## Arbiter

Arbiter帮助您搜索超参数空间，为神经网络寻找最理想的参数组合及架构。这非常重要，因为寻找恰当的架构和超参数是一个很大的组合问题。来自微软研发部等企业实验室的ImageNet大赛获胜团队正是通过搜索超参数空间才得出了ResNet这样的150层神经网络。

### Github代码库

Github代码库请见[此处](https://github.com/deeplearning4j/Arbiter)。

* [arbiter-core](https://github.com/deeplearning4j/Arbiter/tree/master/arbiter-core)：Aribter-core用网格搜索等算法来搜索超参数空间。它会提供一个GUI界面。
* [arbiter-deeplearning4j](https://github.com/deeplearning4j/Arbiter/tree/master/arbiter-deeplearning4j)：Arbiter可以同DL4J模型互动。在进行模型搜索时，您需要能运行模型。这样可以对模型进行试点，进而找出最佳的模型。
