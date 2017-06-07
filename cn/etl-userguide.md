---
title: DeepLearning4J－ETL用户指南
layout: cn-default
---


# DL4J：ETL用户指南

神经网络需要处理的数据有许多种不同的来源和类型，例如日志文件、文本文档、表格数据、图像、视频等。神经网络数据加工的目标是将各类数据转换为一系列存放于多维数组（Multi-Dimensional Array）中的值。 

数据可能还需要进行各种预处理，包括转换、缩放、标准化、变换、合并、划分为定型与测试数据集、随机排序等。本页主要介绍目前可用的数据加工工具及其使用方法。 

* 记录读取器 
* 标准化器
* 转换

## 现有ETL路径的流程图

![Alt text](../img/ETL.svg)

## 记录读取器

<!-- put border on the table -->
<style>
table
{border:1px solid black;
}
td
{border:1px solid black;
}
th
{border:1px solid black;
}

</style>

记录读取器是Skymind团队开发的ETL流程管理库DataVec中的一种类，名称为`RecordReader`。

### 可用的记录读取器

<!-- table generated with http://www.tablesgenerator.com/markdown_tables from CSV export of google sheets -->

| 名称                           | 概述                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 用途                                                      |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| BaseImageRecordReader          | 图像记录读取器的基础类                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 图像数据                                                |
| CodecRecordReader              | 编解码器记录读取器，用于解析：H.264（AVC）主流画质解码器、MP3编/解码器、Apple ProRes解码器和编码器、AAC编码器、H.264基本画质编码器、Matroska（MKV）解复用器和复用器、MP4（ISO BMF、QuickTime）解复用器/复用器及工具、MPEG 1/2解码器（支持隔行扫描）、MPEG PS/TS解复用器、Java player applet、VP8解码器、MXF解复用器；感谢jcodec提供基础解析器                                                                                                                                                                                                                                                                                                                                                                                                                     | 视频                                                     |
| CollectionRecordReader         | 集合记录读取器。主要用于测试。| 测试                                                   |
| CollectionSequenceRecordReader | 用于序列的集合读取器。主要用于测试。| 序列数据                                             |
| ComposableRecordReader         | 用于各个数据加工管道的RecordReader。单项记录是两个集合的串联。创建一个RecordReader来对RecordReader进行迭代并将其串联起来。hasNext为所有后续的RecordReader串联之和，对集合使用addAll返回一项记录                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 合并后的数据                                               |
| CSVNLinesSequenceRecordReader  | CSV序列记录读取器，使用条件：（a）所有时间序列位于单个文件中；（b）每个时间序列长度相等（在构造器中指定）；（c）时间序列之间未使用分隔符。例如，若nLinesPerSequence=10，0～9行为第一个时间序列，10～19行为第二个时间序列，以此类推。| 表格序列数据                                     |
| CSVRecordReader                | 简单的CSV记录读取器                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 表格数据                                              |
| CSVSequenceRecordReader        | CSV序列记录读取器，用于读取CSV格式的序列数据，其中每个序列各有一个文件（同时存在多个文件），文件中的每一行表示一个时间步                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 表格序列数据                                     |
| FileRecordReader               | 文件读写器                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 文件                                                     |
| ImageRecordReader              | 图像记录读取器。读取本地文件系统，解析给定高度和宽度的图像。所有图像的高度、宽度及通道数量都会被缩放、变换至给定的值。也可以添加指定的标签（基于目录结构的one-of-k编码，根目录下的每个子目录都是一个索引标签）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 图像数据                                                |
| JacksonRecordReader            | 支持JSON、XML和YAML：每个文件仅限一项记录，通过Jackson ObjectMapper实现| JSON、XML、YAML                                            |
| LibSvmRecordReader             | 用于SVM（支持向量机）内容的记录读取器。| LibSVM内容                                            |
| LineRecordReader               | 逐行读取文件                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 文本                                                      |
| ListStringRecordReader         | 对一系列字符串进行迭代，返回一项记录。只接受@link {ListStringInputSplit}作为输入。| 文本                                                      |
| MatlabRecordReader             | Matlab记录读取器                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Matlab                                                    |
| RegexLineRecordReader          | RegexLineRecordReader：逐行读取一个文件，用一个正则表达式将其切分为字段。具体而言，我们采用的是Pattern和Matcher类。可加载整个文件。示例：格式如“2016-01-01 23:59:59.001 1 DEBUG First entry message!”的数据可用正则表达式字符串"(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)"切分为4个可写文本对象：["2016-01-01 23:59:59.001", "1", "DEBUG", "First entry message!"]| 带有正则表达式的文本                                           |
| RegexSequenceRecordReader      | RegexSequenceRecordReader：逐行读取一个文件（作为序列），用一个正则表达式将其切分为字段。具体而言，我们采用Pattern和Matcher类来切分文件。示例：格式如“2016-01-01 23:59:59.001 1 DEBUG First entry message!”的数据可用正则表达式字符串"(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)"切分为4个可写文本对象：["2016-01-01 23:59:59.001", "1", "DEBUG", "First entry message!"]注：RegexSequenceRecordReader支持多种错误处理模式，通过RegexSequenceRecordReader.LineErrorHandling实现。与指定的正则表达式不相匹配的无效文本可以导致异常（FailOnInvalid），可以无提示跳过（SkipInvalid），或者在跳过无效文本的同时记录警告（SkipInvalidWithWarning）| 文本序列数据正则表达式                                  |
| SequenceRecordReader           | 用于一个记录序列。sequenceRecord()通常在本地使用。sequenceRecord(URI uri, DataInputStream dataInputStream)用于Spark等                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 序列数据                                             |
| SVMLightRecordReader           | 改编自weka svmlight读取器2015年6月版－改编后能够识别HDFS式的数据块切分                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | SVMLight                                                  |
| TfidfRecordReader              | TF-IDF记录读取器（包装一个TF-IDF向量化器，用于传递标签并确保与记录读取器接口相符）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 自然语言处理、词频－逆文档频率 |
| VideoRecordReader              | 视频就是一系列变化的图片，处理时应当考虑到这一点。该方法对一个根目录进行迭代，返回                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 视频                                                     |
| WavFileRecordReader            | Wav文件加载器                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 音频                                                     |



--------------------

<!-- End Table -->

## 图像基础

为神经网络读取图像时，所有的图像数据在某一时刻必须都缩放至相同的尺寸。图像的初始缩放由`ImageRecordReader`完成。 

在定型、测试或推断之前加载一系列数据记录的方法如以下代码所示。读取灰阶图像时请将`channels`设定为1。 

```
ImageRecordReader recordReader = new ImageRecordReader(height,width,channels);
```

加载单幅图像用于推断： 

```
NativeImageLoader loader = new NativeImageLoader(height, width, channels); \\ 加载和缩放
INDArray image = loader.asMatrix(file); \\ 创建INDarray
INDArray output = model.output(image);   \\ 获得模型对图像的预测
```

## 图像数据增强

如果您的图像数据不足以完成神经网络的定型，您可以将现有的图像转换、采样或裁剪，生成额外的有效输入，增加可用的定型数据。 

## 添加标签

在构建分类器时，标签是您想要预测的输出值，而与这些标签相关联的数据是输入。就CSV文件而言，标签可能就是数据记录本身的一部分，与相关输入紧邻着存放在同一行。`CSVRecordReader`让您可以把特定的字段指定为标签。 

DataVec转换流程可将文本标签变换为数值。标签可能需要依据文件路径生成，比如图像分别存放在一系列目录中，目录名称代表了图像的标签。标签也有可能由文件名本身表示，此时就需要从文件所在的目录内部收集数据。 

您可以用[ParentPathLabelGenerator](http://github.com/deeplearning4j/DataVec/blob/master/datavec-api/src/main/java/org/datavec/api/io/labels/ParentPathLabelGenerator.java)和[PathLabelGenerator](https://github.com/deeplearning4j/DataVec/blob/master/datavec-api/src/main/java/org/datavec/api/io/labels/PathLabelGenerator.java)这两个DataVec中的类来添加标签。 

依据父目录名称对图像进行标记的示例如下。 

```
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
```


## 图像转换

图像会被系统作为像素值的数组读取。像素值通常是8比特，因此一幅包括一黑一白2个像素的图像的数组是`[0,255]`。虽然神经网络可以直接用原始数据来定型，但最好还是先将数据标准化。“零均值单位方差”即让所有的值减去实际平均值，然后缩放至-1与1之间，以0为平均值。 

图像定型数据可以通过旋转样例或倾斜图像来增强。 

### 可用的图像转换方法

| 转换方法           | 具体说明                                                                 |
|--------------------------|-----------------------------------------------------------------------------------|
| BaseImageTransform       | 基础类                                                                        |
| ColorConversionTransform | 采用CVT（cvtcolor）的色彩变换                                   |
| CropImageTransform       | 以明确或随机方式裁剪图像                                        |
| EqualizeHistTransform    | 用于改善图像的对比度                                          |
| FilterImageTransform     | 用FFmpeg（libavfilter）过滤图像                                         |
| FlipImageTransform       | 以明确或随机方式翻转图像                                        |
| ImageTransform           | 基础类                                                                        |
| MultiImageTransform      | 应用多种转换方法                                                         |
| ResizeImageTransform     | 调整图像尺寸，该转换方法适用于将整个加工管道中的图像强制调整为同样的尺寸。|
| RotateImageTransform     | 以明确或随机方式旋转图像                                       |
| ScaleImageTransform      | 以明确或随机方式缩放图像                                       |
| ShowImageTransform       | 在屏幕上显示图像，仅用于可视化，不具备转换功能            |
| WarpImageTransform| 以明确或随机方式对图像进行透视变换                     |



## 数据转换

通过DataVec摄取数据时，可以用包含多个步骤的转换流程来转换数据。 

DataVec现有的数据转换功能如下：

| 转换方法                        | 具体说明                                                                   |
|---------------------------------------|-------------------------------------------------------------------------------------|
| BaseColumnsMathOpTransform            | 多列数学运算的基础类。|
| BaseColumnTransform                   | “将单个列中的值映射至新的值。例如：对单个列进行字符串 -> 字符串，或空白 -> x类型的转换” |
| BaseDoubleTransform                   | 基础类                                                                          |
| BaseIntegerTransform                  | “抽象整数转换（单列）”                                    |
|                                      |                                                                                     |
| BaseStringTransform                   | 抽象字符串列转换                                                    |
| BaseTransform                         | “基础转换：一种抽象转换列”                                          |
| CategoricalToIntegerTransform         | 将分类数据转换为整数                                                   |
| CategoricalToOneHotTransform          | 将分类数据转换为one-hot形式                                                   |
| ConditionalCopyValueTransform         | “如果一项条件得到满足/为真，将一个特定列中的值替换为来自另一个列的新值。”|
| ConditionalReplaceValueTransform      | 基于条件的替换                                              |
| DeriveColumnsFromTimeTransform        | 转换流程                                                                   |
| DoubleColumnsMathOpTransform          | “新增一个double类型的列，由一个或多个其他列的值计算得出。”|
| DoubleMathOpTransform                 | double类型数据的数学运算                                                       |
| DuplicateColumnsTransform             | 复制一个或多个列。|
| IntegerColumnsMathOpTransform          | “新增一个整数类型的列，由一个或多个其他列的值计算得出。”
| IntegerMathOpTransform                | 整数数学运算                                                      |
| IntegerToCategoricalTransform         | 将整数类型的列转换为分类数据列                                   |
| Log2Normalizer                        | 标准化：scale * log2((in-columnMin)/(mean-columnMin) + 1)               |
| LongColumnsMathOpTransform            | “新增一个长整数类型的列，由一个或多个其他列的值计算得出。”|
| LongMathOpTransform                   | 让长整型列的值与一个长整型标量进行就地运算。|
| MapAllStringsExceptListTransform      | 从映射表中剔除的列表                                                            |
| MinMaxNormalizer                      | 标准化方法之一，用于将（最小值到最大值）线性映射至（新最小值到新最大值）。|
| ReduceSequenceByWindowTransform       | “对序列数据应用窗口函数，对数据窗口应用reduce函数”                        |
| RemoveAllColumnsExceptForTransform    | 保留指定的列                                                            |
| RemoveColumnsTransform                | 删除指定的列                                                            |
| RemoveWhiteSpaceTransform             | 删除空格                                                                   |
| RenameColumnsTransform                | 重命名列                                                                      |
| ReorderColumnsTransform               | 将列重新排序                                                                    |
| ReplaceEmptyIntegerWithValueTransform | 将空缺的整数替换为指定的值                                          |
| ReplaceEmptyStringTransform           | 将空缺的字符串替换为指定的值                                           |
| ReplaceInvalidWithIntegerTransform    | 将无效值替换为整数                                                  |
| StandardizeNormalizer                 | “采用(x-mean)/stdev公式进行标准化。也称为标准分数、规范化等。” |
| StringListToCategoricalSetTransform   | 字符串转换为分类数据列表                                                             |
| StringMapTransform                    | 字符串转换为映射表                                                                       |
| StringToCategoricalTransform          | 字符串转换为分类数据                                                               |
| StringToTimeTransform                 | 由字符串生成数值时间                                                   |
| SubtractMeanNormalizer                | 减去平均值                                                                       |
| TimeMathOpTransform                   | 时间变换                                                                    |


## 缩放和标准化 

通过`RecordReader`获取的数据通常会传递给一个数据集迭代器，迭代器会遍历数据并对其进行预加工，以便输入神经网络。可以摄取的数据已经变为一个多维数组（INDarray），不再是用迭代器读取一个数据记录序列的形式。该阶段也有一系列转换和缩放的工具。由于数据已经是INDarray，此处介绍的工具都属于Skymind的科学计算库ND4J的一部分。文档参见[此处](http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/DataNormalization.html)

### 代码示例

```
	DataNormalization scaler = new ImagePreProcessingScaler(0,1);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);
```

### 可用的ND4J预处理器


| ND4J数据集预处理器 | 用途                                                                                                                                           |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| ImagePreProcessingScaler   | 指定最大值、最小值的缩放，可设定区间。像素值可以从0->255缩放至minRange->maxRange的区间内，默认minRange = 0，maxRange = 1 |
| NormalizerMinMaxScaler     | 指定最大值、最小值的缩放，可设定区间：X -> (X - min/(max-min)) * (given_max - given_min) + given_mi                                        |
| NormalizerStandardize      | 标准缩放器，计算列的移动方差与均值                                                                             |


	
## 使用JavaCV、OpenCV和ffmpeg滤镜的图像转换

ffmpeg和OpenCV是用于过滤、转换图像及视频的开源库。在7.2及以上版本中获取ffmpeg过滤器的方法是向`pom.xml`文件中添加下列代码，将依赖项替换为当前版本。 

```
<dependency> <groupId>org.bytedeco</groupId> <artifactId>javacv-platform</artifactId> <version>1.3</version> </dependency>
```

文档
* [JavaCV](https://github.com/bytedeco/javacv)
* [OpenCV](http://opencv.org/)
* [ffmpeg](http://ffmpeg.org/)


## 自然语言处理

DeepLearning4J提供自然语言处理（NLP）的工具包。详情参见[此页](https://deeplearning4j.org/cn/bagofwords-tf-idf)。 

## 时间序列或序列数据

循环神经网络可用于分析序列和时间序列数据。DataVec提供的`CSVSequenceReader`类可以从文件中读取序列数据。`UCISequenceClassificationExample`就是一个很好的例子。 

数据被分为测试和定型两个数据集，因此代码分别为每个数据集创建了一个迭代器。 

这一数据集中共有六种标签。特征目录中每个包含数据的文件都有一个对应的标签文件，位于标签目录下。标签文件中仅有单个项，而特征文件则包含了相应设备的活动序列记录。 

```
private static File baseDir = new File("src/main/resources/uci/");
private static File baseTrainDir = new File(baseDir, "train");
private static File featuresDirTrain = new File(baseTrainDir, "features");
private static File labelsDirTrain = new File(baseTrainDir, "labels");
private static File baseTestDir = new File(baseDir, "test");
private static File featuresDirTest = new File(baseTestDir, "features");
private static File labelsDirTest = new File(baseTestDir, "labels");

```

`NumberedFileInputFormat`用`String.Format`从文件名中提取索引。数据目录包含文件0.csv->449.csv

以下是读取特征及标签的代码。 

```
SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
```

## 摄取图像数据并输入预定型的模型

`NativeImageLoader`可以读取一幅图像并将其转变为一个INDArray。请注意，您需要用网络定型时调整尺寸、缩放、标准化的方式来缩放和调整导入的图像。 

### 单一图像路径的流程图

![ETL Single Image](../img/ETL_single_image.svg)
