---
title: 用定制的数据加工管道将图像载入深度神经网络
layout: cn-default
redirect_from: /zh-image-data-pipeline
---

# 针对图像等数据的定制数据加工管道

Deeplearning4j示例所使用的基准数据集不会对数据加工管道造成任何障碍，因为我们已通过抽象化将这些障碍去除。但在实际工作中，用户接触的是未经处理的杂乱数据，需要先预处理、向量化，再用于定型神经网络，进行聚类或分类。 

*DataVec*是我们的机器学习向量化库，可以用于定制数据准备方法，便于神经网络学习。([DataVec Javadoc参见此处](http://deeplearning4j.org/datavecdoc/)。)

本教程将介绍一些在处理图像时可能会遇到的关键问题，包括标签生成、向量化以及如何配置神经网络 


## 简介视频

<iframe width="420" height="315" src="https://www.youtube.com/embed/EHHtyRKQIJ0" frameborder="0" allowfullscreen></iframe>

## 数据加工管道教学视频

我们还提供一系列配合屏幕录像进行讲解的教学视频，内容包括如何编写代码来处理数据目录中的图像、根据路径生成标签、搭建神经网络并用图像数据定型网络。同一系列中的其他视频还介绍了如何保存已定型网络、加载已定型网络、用搜集自互联网的未见图像进行测试。 

以下是该系列的第一个视频

<iframe width="420" height="315" src="https://www.youtube.com/embed/GLC8CIoHDnI" frameborder="0" allowfullscreen></iframe>

## 加载标签

我们的示例代码库中有一个使用ParentPathLabelGenerator的例子。类的名称是ImagePipelineExample.java 

        File parentDir = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/DataExamples/ImagePipeline/");
        //将父目录下各子目录中包含“允许的扩展名”的文件分为定型集和测试集，这一步需要用一个随机数生成器来确保可复现性
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //您无需手动指定标签。这个类（实例化如下）会
        //解析父目录并将子目录名称用作标签/类别的名称
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

## <a name="record">读取记录，对数据进行迭代</a>

以下代码可将原始图像转换为DL4J和ND4J相兼容的格式：

        // 将RecordReader实例化。指定图像的高和宽。
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        
        // 通道指图像的色彩深度，1为灰阶，3为RGB

        // 指向数据路径。 
        recordReader.initialize(new FileSplit(new File(parentDir)));

RecordReader是DataVec中的一个类，可以帮助将字节式输入转换为记录式的数据，亦即一组数值固定并以独特索引ID加以标识的元素。向量化是将数据转换为记录的过程。记录本身是一个向量，每个元素是一项特征。

更多详情参见DataVec的[JavaDoc](http://deeplearning4j.org/datavecdoc/org/datavec/image/recordreader/ImageRecordReader.html)。 

[ImageRecordReader](https://github.com/deeplearning4j/DataVec/blob/master/datavec-data/datavec-data-image/src/main/java/org/datavec/image/recordreader/ImageRecordReader.java)是RecordReader的子类，用于自动载入28 x 28像素的图像。你可以改变输入ImageRecordReader的参数，将尺寸改为自定义图像的大小，但务必要调整超参数`nIn`，使之等于图像高与宽的乘积。如需加载大小为28 x 28的图像，MultiLayerNetwork的配置中应当包含`.nIn(28 * 28)`

如果使用LabelGenerator，则调用ImageRecordReader时，其参数应包括labelGenerator。
`ImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator)`

<!-- ![Alt text](../img/recordreader_extensions.png) 
Rebuild this image from a screenshot of dl4j 
-->

DataSetIterator是用于遍历列表元素的一个Deeplearning4J类。迭代器按顺序访问数据列表中的每个项目，同时通过指向当前的元素来记录进度，在遍历过程中每前进一步就自动指向下一个元素。

        // 从DataVec到DL4J
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        // 参数包括：DataVec recordReader的类型、批次大小、标签的索引值、标签类别
        // 的数量

DataSetIterator对输入数据集进行迭代，每次迭代均抓取一个或多个（batchSize）新样例，将其载入神经网络可以识别的DataSet（INDArray）对象。上述代码还指示[RecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.java)将图像转换为一条元素的直线（向量），而非一个28 x 28的网格（矩阵）；此外还指定了标签的配置。

`RecordReaderDataSetIterator`的参数可以设置任意特定的recordReader（针对图像或声音等数据类型）和批次大小。进行有监督学习时，还可指定标签的索引值，设置输入样例的标签可能有多少种不同的类别（LFW数据集共有5749种标签）。 

## 配置模型

以下是一个神经网络的配置示例。[NeuralNetConfiguration类的术语表](./neuralnet-configuration.html)对许多超参数都已作了说明，所以此处仅对一些比较特殊的参数设置进行简要概述。

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DeepAutoEncoderExample.java?slice=29:71"></script>

* *optimizationAlgo*依赖LINE_GRADIENT_DESCENT，而不是LBFGS。 
* *nIn*设定为784，让每个图像像素成为一个输入节点。如果你的图像尺寸改变（即总像素数发生变化），则nIn也应当改变。
* *list*操作符设为4，表明有三个受限玻尔兹曼机（RBM）隐藏层和一个输出层。一个以上的RBM组成一个深度置信网络（DBN）。
* *lossFunction*设为RMSE，即均方根误差。这种损失函数用于定型网络，使之能正确地重构输入。 

## 模型的建立和定型

配置结束时，调用build并将网络的配置传递给一个MultiLayerNetwork对象。

                }).build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

可以用下列代码示例之一来设定在神经网络定型时显示性能并帮助进行调试的监听器：

        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(10), new GradientPlotterIterationListener(10)));

        或

        network.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

## 模型定型

数据加载后，模型框架构建完成，用数据对模型进行定型。对数据迭代器调用next，让迭代器读取下一批数据，每次将按批次大小返回一定量的数据。以下的代码显示了如何用数据集迭代器来进行循环，随后对模型运行fit，用数据定型模型。

        // 定型
        while(iter.hasNext()){
            DataSet next = iter.next();
            network.fit(next);
        }

## 评估模型

模型定型完毕后，再将数据输入其中进行测试，评估模型的性能。通常较好的做法是采用交叉验证，事先将数据集分为两部分，用模型未曾见过的数据进行测试。以下的例子展示了如何重置当前的迭代器，初始化evaluation对象，再将数据输入其中以获得性能信息。

        // 用同样的定型数据作为测试数据。 
        
        iter.reset();
        Evaluation eval = new Evaluation();
        while(iter.hasNext()){
            DataSet next = iter.next();
            INDArray predict2 = network.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict2);
        }
        
        System.out.println(eval.stats());

在这一过程中使用交叉验证的另一方法是加载全部数据，将其分为一个定型集和一个测试集。鸢尾花数据集足够小，可以载入全部的数据，再完成划分。但许多用于生产型神经网络的数据集并非如此。在本例中可通过以下代码使用替代方法：

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

如要将较大的数据集分为测试集和定型集，则必须对测试和定型两个数据集都进行迭代。这一操作就暂且交由读者自己思考。 
