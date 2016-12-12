---
title: DataVec - 向量化及表达式模板库
layout: cn-default
---

# DataVec：向量化及表达式模板库

DataVec帮助克服机器学习及深度学习实现过程中最重大的障碍之一：将数据转化为神经网络能够识别的格式。神经网络所能识别的是向量。因此，对许多数据科学家而言，在开始用数据定型自己的算法之前，首先必须要解决向量化的问题。如果您的数据以CSV（逗号分隔值）格式储存在平面文件中，必须先转换为数值格式再加以摄取，又或者您的数据是一些有标签的图像的目录结构，那么DataVec这款工具就可以帮助您组织数据，以供在Deeplearning4J中使用。 


在开始使用DataVec之前，请**通读本页内容**，尤其是有关[读取记录](#record)的段落。



## 简介视频

以下视频介绍了图像数据如何转换为向量。 

<iframe width="420" height="315" src="https://www.youtube.com/embed/EHHtyRKQIJ0" frameborder="0" allowfullscreen></iframe>

## 主要特点
- [DataVec](https://github.com/deeplearning4j/DataVec)采用输入/输出格式系统（就像Hadoop MapReduce用InputFormat来确定具体的InputSplit和RecordReader一样，DataVec也会用不同的RecordReader来将数据序列化）
- 支持所有主要的输入数据类型（文本、CSV、音频、图像、视频），每种类型都有相应的输入格式
- 采用输出格式系统来指定一种与实现无关的向量格式（ARFF、SVMLight等）
- 可以为特殊输入格式（如某些罕见的图像格式）进行扩展；也就是说，您可以编写自定义的输入格式，让余下的基本代码来处理转换加工管道
- 让向量化成为“一等公民”
- 内置数据转换及标准化工具
- 请参阅[DataVec的Javadoc](http://deeplearning4j.org/datavecdoc/)

下文中有一个<a href="#tutorial">简短的教程</a>。

## 应用举例

 * 将基于CSV格式的UCI鸢尾花数据集转换为svmLight开放式向量文本格式
 * 将MNIST数据集的原始二进制文件转换为svmLight文本格式。
 * 将原始文本转换为Metronome向量格式
 * 用TF-IDF方法将原始文本转换为文本向量格式{svmLight, metronome, arff}
 * 将原始文本转换为word2vec文本向量格式{svmLight, metronome, arff}

## 支持的向量化引擎

 * 用脚本语言将各种CSV转换为向量
 * MNIST转换为向量
 * 文本转换为向量
    * TF-IDF
    * 词袋
    * Word2vec

## CSV转换引擎

CSVRecordReader足以处理格式规范的数值数据；但如果数据包含表示布尔值（真/假）的字符串或标签的字符串等非数值字段，那就需要进行架构（Schema）转换。DataVec使用Apache [Spark](http://spark.apache.org/)来进行转换运算。*即使不了解Spark的内部细节，也能成功使用DataVec进行转换

## 架构转换视频

[点击此处](http://www.youtube.com/watch?v=L5DtC8_4F-c)观看一个简单的DataVec转换示例教程及相关代码

## Java代码示例

我们的[示例](https://deeplearning4j.org/quickstart#examples)中包括一组DataVec示例。   

<!-- Note to Tom, write DataVec setup content

## <a name="tutorial">设置DataVec</a>

在Maven中央仓库中搜索[DataVec](https://search.maven.org/#search%7Cga%7C1%7CDataVec)，得到可以使用的JAR文件列表。

将依赖项信息添加到pom.xml当中。

-->


## <a name="record">读取记录，对数据进行迭代</a>

以下代码可将示例中的原始图像转换为DL4J和ND4J可以识别的格式：

``` java
// 将RecordReader实例化。指定图像的高和宽。
RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);

// 指向数据路径。 
recordReader.initialize(new FileSplit(new File(labeledPath)));
```

RecordReader是DataVec中的一个类，可以帮助将字节式输入转换为记录式的数据，亦即一组数值固定并以独特索引ID加以标识的元素。向量化是将数据转换为记录的过程。记录本身是一个向量，每个元素是一项特征。

[ImageRecordReader](https://github.com/deeplearning4j/Canova/blob/f03f32dd42f14af762bf443a04c4cfdcc172ac83/canova-nd4j/canova-nd4j-image/src/main/java/org/canova/image/recordreader/ImageRecordReader.java)是RecordReader的子类，用于自动载入28 x 28像素的图像。所以LFW数据集的图像会被缩放为28像素 x 28像素的大小。你可以改变输入ImageRecordReader的参数，将尺寸改为自定义图像的大小，但务必要调整超参数`nIn`，使之等于图像高与宽的乘积。 

上一例中还包括一些其他的参数：`true`指示加载器为记录追加一个标签，`labels`是一组用于验证神经网络模型结果的监督值（目标值）。以下是DataVec中所有预设的RecordReader扩展（显示方式是在IntelliJ中右击`RecordReader`，点击下拉菜单中的`Go To`，再选择`Implementations`）：

![Alt text](./img/recordreader_extensions.png)

DataSetIterator是用于遍历列表元素的一个Deeplearning4J类。迭代器按顺序访问数据列表中的每个项目，同时通过指向当前的元素来记录进度，在遍历过程中每前进一步就自动指向下一个元素。

``` java
// 从DataVec到DL4J
DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784, labels.size());
```

DataSetIterator对输入数据集进行迭代，每次迭代均抓取一个或多个新样例，将其载入神经网络可以识别的DataSet对象。上述代码还指示[RecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/3e5c6a942864ced574c7715ae548d5e3cb22982c/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/canova/RecordReaderDataSetIterator.java)将图像转换为一条元素的直线（向量），而非一个28 x 28的网格（矩阵）；此外还设定了标签数的限制。

`RecordReaderDataSetIterator`的参数可以设置任意特定的recordReader（针对图像或声音等数据类型）和批次大小。进行有监督学习时，还可指定标签索引和可以为输入添加的标签数量（LFW数据集的标签数为5,749）。 

如需要详细了解将数据从DataVec移动至Deeplearning4j的其他步骤，可以参阅[定制图像数据加工管道的指南](./cn/zh-simple-image-load-transform)。

## 执行方式

可以作为本地串行进程和无需代码变化的MapReduce（MR引擎已在规划中）向外扩展进程来运行。

## 支持的向量格式
* svmLight
* libsvm
* Metronome
* ARFF

## 内置通用功能
* 能够用核函数哈希和TF-IDF等方法将常规文本转换为向量
