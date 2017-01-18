---
title: 用定制的数据加工管道将图像载入深度神经网络
layout: default
---

# 针对图像等数据的定制数据加工管道

Deeplearning4j示例所使用的基准数据集不会对数据加工管道造成任何障碍，因为我们已通过抽象化将这些障碍去除。但在实际工作中，用户接触的是未经处理的杂乱数据，需要先预处理、向量化，再用于定型神经网络，进行聚类或分类。 

*DataVec*是我们的机器学习向量化库，可以用于定制数据准备方法，便于神经网络学习。([DataVec Javadoc参见此处](http://deeplearning4j.org/datavecdoc/)。)

本页的教程将介绍如何加载图像数据集并对其进行转换操作。为求简明易懂，本教程仅使用*牛津花卉数据集*中的三个类别，各有十幅图像。下列代码片段仅供参考，请勿直接复制粘贴使用。 
[请使用此处完整示例中的代码](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/ImagePipelineExample.java)

## 为图像建立正确的目录结构
简而言之，数据集中的图像必须按类别/标签存放在不同的目录下，而标签/类别目录则位于父目录下。

* 下载数据集。 

* 建立父目录。

* 在父目录中创建子目录，按相应的标签/类别名称命名。

* 将所有属于某一类别/标签的图像移动到相应的目录下。

通常需要采用的目录结构如下图所示。

>                                   parentDir
>                                 /   / | \  \
>                                /   /  |  \  \
>                               /   /   |   \  \
>                              /   /    |    \  \
>                             /   /     |     \  \
>                            /   /      |      \  \
>                      label_0 label_1....label_n-1 label_n


在本页的示例中，parentDir（父目录）对应`$PWD/src/main/resources/DataExamples/ImagePipeline/`，而子目录labelA、labelB、labelC（标签A、B、C）下各有十幅图像。 

## 加载图像前的详细设置
* 指定包含已标记图像的各个目录所在的父目录路径。
 
~~~java
File parentDir = new File(System.getProperty("user.dir"), "src/main/resources/DataExamples/ImagePipeline/");
~~~

* 指定将数据集分为测试集和定型集时允许的扩展名和需要使用的随机数生成器。 

~~~java
FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
~~~

* 设置一个标签生成器，这样就无需手动指定标签。生成器会将子目录名称用作标签/类别的名称。

~~~java
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
~~~

* 指定路径筛选器，以便精确控制每种类别所要加载的最小/最大样例数。以下是最基本的代码。有关细节请参阅javadoc。

~~~java
BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
~~~

* 指定测试集与定型集的比例，此处为80%和20%

~~~java
InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
InputSplit trainData = filesInDirSplit[0];
InputSplit testData = filesInDirSplit[1];
~~~

## 图像数据加工管道的详细设置

* 为图像记录加载器指定高度和宽度，调整数据集中所有图像的尺寸。 

~~~java
ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
~~~
请注意：*数据集中的图像不必都是相同的尺寸*。DataVec可以调整图像尺寸。本示例中的图像均为不同尺寸，它们都会被调整为指定的高度和宽度

* 指定转换操作

神经网络的优势在于无需人工进行特征工程。但是，通过转换图像来人为地扩大数据集规模可能会带来益处，例如这一kaggle竞赛的优胜作品<http://benanne.github.io/2014/04/05/galaxy-zoo.html>。或者，你可能需要裁剪图像，只留下相关的部分。例如检测出图像中的人脸，然后将图像裁剪至人脸的大小。DataVec拥有OpenCV所有内置的强大功能。以下是一个将图像翻转后加以显示的基本功能示例：

~~~java
ImageTransform transform = new MultiImageTransform(randNumGen,new FlipImageTransform(), new ShowImageTransform("After transform"));
~~~

可以用以下方法实现链式转换操作，自行编写自动运行任何功能的类。

~~~java
ImageTransform transform = new MultiImageTransform(randNumGen, new CropImageTransform(10), new FlipImageTransform(),new ScaleImageTransform(10), new WarpImageTransform(10));
~~~

* 用定型数据和转换操作链对记录加载器进行初始化

~~~java
recordReader.initialize(trainData,transform);
~~~

## 递交匹配
DL4J的神经网络也需要用一个数据集或数据集迭代器来进行匹配。数据集和迭代器是我们的学习框架的基本概念。迭代器的具体使用方法请参见其他示例。以下是用图像记录加载器构建数据集迭代器的方法。

~~~java
DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
~~~

DataSetIterator通过recordReader对输入数据集进行迭代，每次迭代均抓取一个或多个新样例，将其载入神经网络可以识别的DataSet对象。

## 缩放DataSet
由DataIterator传入的DataSet将包含一个或多个像素值数组。例如，假设我们指定RecordReader的高度为10，宽度为10， 通道为1，即灰阶图像

~~~java
        ImageRecordReader(height,width,channels)
~~~

那么返回的DataSet将是一个10 x 10的矩阵，其元素为0到255之间的数值。0代表黑色像素，255代表白色像素。100则代表灰色。如果图像是彩色的，就会有三个通道。 

将图像像素值的范围从0～255缩放到0～1可能会更有帮助。 

这可以通过以下的代码来实现。 

~~~java
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
~~~
