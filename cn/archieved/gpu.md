---
title: 用GPU运行Deeplearning4j
layout: cn-default
---

# 用GPU运行Deeplearning4j

Deeplearning4j同时支持分布式和本机GPU。用户可以用NVIDIA Tesla、Titan、GeForce GTX等单个本地GPU运行DL4J，也可以使用云端的NVIDIA GRID GPU来运行。 

若要在GPU上定型神经网络，您需要对根目录下的POM.xml文件做一项更改。[快速入门指南](./quickstart)中提到了一个默认状态下将项目配置成在CPU上运行的POM文件。具体配置内容如下：

            <name>DeepLearning4j Examples Parent</name>
            <description>Examples of training different data sets</description>
            <properties>
                <nd4j.backend>nd4j-native-platform</nd4j.backend>

将最后一行改为如下形式即可：

        <nd4j.backend>nd4j-cuda-8.0-platform</<nd4j.backend>

ND4J是驱动Deeplearning4j的数值运算引擎。它依靠各种“后端”在不同类型的硬件上运行。[Deeplearning4j线上交流群](https://gitter.im/deeplearning4j/deeplearning4j)的用户常会提到后端，他们说的就是指向某种芯片的软件包。我们在后端上开展硬件优化工作。

## 疑难解答

如果您有多个GPU，但系统却只允许使用一个，解决方法如下：将`CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);`添加至`main()`方法的第一行即可。

<p align="center">
<a href="./quickstart" class="btn btn-custom" onClick="ga('send', 'event', ‘quickstart', 'click');">Deeplearning4j的快速入门指南</a>
</p>


## 多GPU数据并行模式

如果您的系统安装有多个GPU，就可以用数据并行模式来定型模型。我们有一个实现数据并行的简单的包装类。您可以考虑以下的用法：

        ParallelWrapper wrapper = new ParallelWrapper.Builder(YourExistingModel)
            .prefetchBuffer(24)
            .workers(4)
            .averagingFrequency(1)
            .reportScoreAfterAveraging(true)
            .useLegacyAveraging(false)
            .build();

ParallelWrapper（并行包装类）将您的现有模型作为主要参数，以并行模式进行定型。若使用GPU，我们建议确保工作节点数量等于或高于GPU的数量。具体数值需要进行调试，因为它们取决于具体的任务以及可用的硬件。

`ParallelWrapper`会复制您的初始模型，每个工作节点分别定型自己的模型。每进行*X*次迭代后（该迭代次数由`averagingFrequency(X)`设置），所有模型将被平均化，然后继续定型。 

需要提醒的是，我们建议数据并行模式的定型采用更高的学习速率。初始速率应当可以提高20%左右。

## 采用并行包装类的早停法

专用早停类`EarlyStoppingParallelTrainer`可以实现与单GPU设备上的早停法相似的功能。更多有关早停法的内容参见[此处](./earlystopping)。

## HALF数据类型

如果您的应用程序有条件使用半精度浮点数（神经网络一般都能支持），启用这一数据类型可以带来以下好处：

* GPU RAM使用量减少一半
* 内存占用量大的运算的性能最高可以提升200%，不过实际的性能提升幅度取决于具体的任务和所用硬件。

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

将这一条调用命令置于应用程序代码的首行，让所有后续的分配/计算以HALF数据类型进行。

但是应当注意：HALF数据类型的精确率远小于FLOAT和DOUBLE类型，因此神经网络调试的难度可能也会大幅上升。

此外，目前我们暂时不能为HALF数据类型提供完全的LAPACK支持。

## 扩大网格

默认设定值适用于大多数GPU，但如果您使用的是高端硬件，且数据量足够庞大，那么或许可以尝试提高网格/块的上限。比如可以采用如下方法：

    CudaEnvironment.getInstance().getConfiguration()
          .setMaximumGridSize(512)
          .setMaximumBlockSize(512);

这不会迫使所有的运算（甚至是次要的运算）都使用特定的网格尺寸，但会为其设定理论限制。 

## 扩大缓存

ND4J基于Java，因此缓存大小对于CUDA后端非常重要，有可能大幅提升或降低性能表现。如果您的RAM容量足够大，直接扩大缓存容量即可。

比如可以采用如下方法：

        CudaEnvironment.getInstance().getConfiguration()
        .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
        .setMaximumDeviceCache(6L * 1024 * 1024 * 1024L)
        .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
        .setMaximumHostCache(6L * 1024 * 1024 * 1024L);

上述代码将允许最多把6GB的GPU RAM用作缓存（实际并不一定会分配这么多），而主机和GPU内存的每个已缓存内存块最大可达1GB。 

由于ND4J的缓存采用一项“再利用”范式，这些较高的设置值不会造成任何负面影响。只有分配给您的应用程序的内存块才能缓存以供再利用。

## 设置环境变量BACKEND_PRIORITY_CPU和BACKEND_PRIORITY_GPU

环境变量BACKEND_PRIORITY_CPU和BACKEND_PRIORITY_GPU的设置可以决定采用的是GPU还是CPU后端。具体用法是将BACKEND_PRIORITY_CPU和BACKEND_PRIORITY_GPU设置为整数。最高的值对应的后端将被采用。 


## 具体运作方式

由于GPU和x86之间的区别，CUDA后端与本机后端相比存在一些设计上的差异。 

相关要点如下：

- CUDA后端高度依赖各类内存缓存。
    * 每个内存块被分配一次，从JVM上下文释放后，我们将其缓存以供之后再次利用。
    * ShapeInfo和TAD缓存用GPU设备的常量内存提高从内核（kernel）上下文访问的速度。
- 内核是“原子性”的（atomic，即不可分割）：一项运算 = 一个预编译的内核（多数情况下均是如此）
- CUDA后端会在实际内核启动之前处理并行配置
- 有些情况下，我们可以在一次运算调用中进行2项运算。这种执行模式称为mGRID，有利于PairwiseTransform运算及之后的其他运算。
- 与nd4j-native后端相似，CUDA后端支持两种并行模式：
    * 元素层级的并行：网格中的所有线程均使用同一个线性缓冲区。
    * TAD层级的并行：网格被划分为多个块，每个线程块处理一个TAD。
- 设备内存释放进程由WeakReferences处理（随后是上文提到的缓存机制）
- 多GPU环境实行线程 <-> 设备关联管理。一个Java线程在任何时候都与一个GPU相关联。


### 扩展阅读

* [基于Spark的Deeplearning4j（采用GPU）](./spark)
* [用OpenMP和SIMD指令对Deeplearning4j进行本机优化](./native)
