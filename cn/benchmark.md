---
title: Deeplearning4j基准测试
layout: cn-default
---

# 如何运行Deeplearning4j基准测试

总训练时间始终等于ETL时间加计算时间。也就是说，神经网络用一个数据集进行训练所需的时间同时取决于数据加工管道和矩阵操作。 

熟悉Python的程序员在运行把Deeplearning4j与主流Python学习框架进行对标的基准测试时，往往只考虑了Python框架用于计算的时间，将其同DL4J用于ETL + 计算的时间相比。这并不是“苹果对苹果”的同类比较。下文将介绍如何优化几项相关参数。 

JVM系统有多个调试开关，只要懂得调试方法，就能大幅提高深度学习在JVM环境下的运行速度。对于JVM而言，您需要记住这几件事：

* 增加[堆空间](http://javarevisited.blogspot.com/2011/05/java-heap-space-memory-size-jvm.html)
* 调整垃圾回收器
* 将ETL设为异步运行
* 预存数据集（即“腌制”，pickling，又译序列化）

## 设置堆空间

用户必须自行重新配置JVM，包括设置堆空间。我们无法给您提供预配置的系统，但可以为您介绍操作方法。以下是堆空间最重要的两个调节开关。

* Xms设置堆空间下限
* Xmx设置堆空间上限

您可以在IntelliJ和Eclipse等IDE中设置，或者用以下方法通过命令行界面设置：

		java -Xms256m -Xmx1024m 此处为类的名称

在[IntelliJ中，这是一项虚拟机参数](https://www.jetbrains.com/help/idea/2016.3/setting-configuration-options.html)，不是程序属性。在IntelliJ中点击运行（绿色按钮）后，参数将作为运行时配置设定。IntelliJ会启动一个采用您所指定的配置的Java虚拟机。 

`Xmx`设置为怎样的大小最合适？这取决于计算机的RAM容量。一般来说，您认为JVM完成工作需要多少堆空间，就分配多少。假设您用的是一台有16G RAM的笔记本计算机，那么可以分给JVM 8G的RAM。对于RAM容量较小的计算机，比较稳妥的下限是3G。 

		java -Xmx3g

虽然听起来可能有些奇怪，但下限和上限应当一样，`Xms`应当等于`Xmx`。如果两者不相等，JVM会按需求逐步分配更多内存，直至到达上限，而这一逐步分配的过程会减慢运行速度。应该一开始就预先分配内存。所以请用： 

		java -Xms3g -Xmx3g 此处为类的名称

IntelliJ会自动指定相关的[Java主类](https://docs.oracle.com/javase/tutorial/getStarted/application/)。

另外一种方法是设置环境变量。您需要找到并更改隐藏的`.bash_profile`文件。这一文件在bash中添加环境变量。要了解具体有哪些变量，请在命令行中输入`env`。要添加更多堆空间，请在控制台输入下列命令：

		echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile

需要增加堆空间是因为Deeplearning4j在后台加载数据，这意味着我们会占用更多的RAM内存。为JVM分配更多堆空间后，我们可以将更多数据缓存到内存中。 

## 垃圾回收

垃圾回收器是在JVM上运行的一个程序，用于去除Java应用程序不再使用的对象，实现自动化内存管理的功能。在Java中创建新对象会消耗堆内内存。一个新的Java对象默认会占用8比特的内存。因此每个新创建的`DatasetIterator`会多占用8比特。 

您可能需要修改Java使用的垃圾回收算法。可以在命令行中输入：

		java -XX:+UseG1GC

改善垃圾回收算法可以提高吞吐量。关于这一问题的详细介绍可参阅InfoQ上的[这篇文章](https://www.infoq.com/articles/Make-G1-Default-Garbage-Collector-in-Java-9)。

DL4J与垃圾回收器紧密相联。JVM与C++之间的桥梁[JavaCPP](https://github.com/bytedeco/javacpp)会严格遵守您用`Xmx`设置的堆空间限制，大量利用堆外空间进行工作。堆外空间的使用量不会超过您所指定的堆空间容量。 

JavaCPP是由Skymind的一位工程师编写的，它依靠垃圾回收器来了解哪些对象已不再使用。我们依靠Java GC来确定回收什么；Java GC指出目标，我们知道如何用JavaCPP来对其解除分配。使用GPU时的情况也与此相同。 

您设定的批次越大，占用的RAM内存就越多。 

## ETL和异步ETL

在`dl4j-examples`示例库中，我们并未将ETL设为异步运行，因为示例必须保持简单。但对于现实应用中的问题，您需要采用异步的ETL，我们会用示例来介绍具体方法。 

数据存储于硬盘上，而硬盘的速度比较慢。这是默认的情况。所以，将数据加载至硬盘时会产生瓶颈。吞吐量优化过程中，瓶颈始终是最慢的部分。比方说，一项分布式Spark任务使用三个GPU工作器、一个CPU工作器，其瓶颈必定是那个CPU。GPU必须等待CPU完成工作。 

Deeplearning4j的`DatasetIterator`类掩盖了在硬盘上加载数据的复杂性。使用Datasetiterator的代码都是相同的，调用过程看起来也一样，但实际工作原理却有所不同。 

* 一种从硬盘加载 
* 一种进行异步加载
* 一种加载RAM中预存的数据

以下是处理MNIST数据集时调用DatasetIterator的统一方法：

            while(mnistTest.hasNext()){
	                DataSet ds = mnistTest.next();
	                INDArray output = model.output(ds.getFeatureMatrix(), false);
	                eval.eval(ds.getLabels(), output);
            }

您可以利用在后台异步运行的加载器来进行优化。Java可以实现真正意义上的多线程。它可以在后台加载数据，同时让其他线程负责计算。所以您可以在运行计算指令的同时向GPU中加载数据。从内存中抓取新数据时，神经网络仍在继续训练。

相关代码参见[此处](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/deeplearning4j-scaleout-parallelwrapper/src/main/java/org/deeplearning4j/parallelism/ParallelWrapper.java#L136)，尤其注意第三行：

        MultiDataSetIterator iterator;
        if (prefetchSize > 0 && source.asyncSupported()) {
            iterator = new AsyncMultiDataSetIterator(source, prefetchSize);
        } else iterator = source;

异步数据集迭代器其实分为两种。大多数情况下用到的是`AsyncDataSetIterator`。其介绍参见[此处的Javadoc](https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/AsyncDataSetIterator.html)。

在某些特殊情况下，比如应用于时间序列的循环网络，或者计算图网络，您需要使用`AsyncMultiDataSetIterator`，其介绍参见[此处的Javadoc](https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/AsyncMultiDataSetIterator.html)。

请注意，在上文的代码中，`prefetchSize`也是一个需要设置的参数。常规的批次大小可能是1000个样例，但如果将`prefetchSize`设为3，就会预提取3000项实例。

## ETL：将Python学习框架与Deeplearning4j进行对比

在Python中，程序员会把数据转换为“[泡菜（pickles）](https://docs.python.org/2/library/pickle.html)”，亦即二进制数据对象。如果处理的是一个比较小的玩具数据集，他们会把所有的泡菜都加载到RAM当中。这实际上等于是绕过了在处理大规模数据集时的一项主要任务。而另一方面，在DL4J的对比测试中，他们又不把所有数据都加载到RAM。所以他们等于是在把DL4J训练计算 + ETL的时间和Python框架的训练计算时间作比较。 

就搬运大规模数据而言，Java其实有很完善的工具可用，如果进行正确的比较，其速度远远快于Python。据Deeplearning4j用户社区报告，在ETL和计算都得到充分优化的前提下，DL4J的速度最快可比Python框架提高多达3700%。

Deeplearning4j的ETL和向量化库是DataVec。DataVec对数据集采用哪种格式没有硬性要求，这与其他深度学习工具不同。（比如，Caffe就会强制要求您使用[hdf5](https://support.hdfgroup.org/HDF5/)格式。）

我们力求提高灵活度。这也就是说，您可以把原始照片文件塞给DL4J，它照样能加载图像，转换数据，再将数据放入一个多维数组，同时生成数据集。 

但是，如果您的训练数据加工管道每次都要进行这一系列操作，Deeplearning4j看起来就会比其他框架慢十倍——因为创建数据集的时间也包括在内了。每次调用`fit`命令时，您都会重新创建一个数据集，如此反复。为了使用方便，我们允许进行这种操作，但提高速度就要用别的方法。我们有办法让它变得和其他框架一样快。 

办法之一是用和Python框架类似的方式预存数据集。（泡菜就是预先格式化的数据。）预存数据集时需要建立一个独立的类。

预存数据集的方法见[此处](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/presave/PreSave.java)。

`Recordreaderdatasetiterator`类会与DataVec互动，为DL4J输出数据集。 

加载预存数据集的方法见[此处](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/presave/LoadPreSavedLenetMnistExample.java)。

在第90行可以看到异步ETL。这个例子对预存的迭代器进行了包装，同时利用了上文的两种方法，在训练网络的同时在后台异步加载预存数据。 

## MKL和基于CPU的推断测试

如果您在CPU上运行推断性质的基准测试，请确保您已将Deeplearning4j与英特尔的MKL库配合使用，后者可以通过一项点击许可获取；Deeplearning4j没有与MKL进行捆绑，这与PyTorch等学习库使用的Anaconda有所不同。 
