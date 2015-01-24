---
title: 
layout: zh-default
---

# ND4J

DeepLearning4j的矩阵运算是使用ND4J为它的科学计算内核。如果您要使用Deeplearning4j,您必须选择一个[ND4J](http://nd4j.org/downloads.html)后端,这将基于您的绘图处理器(GPU)或原计算性能。

许多深度学习研究人员已经将Cuda的GPU的并行或多重处理和矩阵计算标准化。不幸的是,行业的从业者必须设法解决被传统硬件限制的问题。即使面对这样的限制,使用CPU仍然可以提高深度学习的计算速度。

我们创建ND4J是因为没有一个以BLAS为基础的JVM允许用户有一个简单统一接口来运行矩阵运算。简单日志门面([SLF4J](http://slf4j.org/))是唯一的答案。

此外,我们认为一个用于创建机器学习算法的通用API是我们值得追求的目标。我们相信没有人愿意重写这些文库,如果他们发现自己的矩阵能更快速的运行。

## 下载

在这里,您会发现GPU、原生与其它组件的deeplearning4j下载包。

就像[ND4J](http://nd4j.org/downloads.html)后端下载, Deeplearning4j需要ND4J来运行。下面有一些二进制包让您可以在不同的后端捆绑起来使用。

## Native

### Jblas

#### 最新
* [tar存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.zip)

#### 0.0.3.2.5
* [tar存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.zip)

### Netlib Blas

#### 最新
* [tar存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.tar.gz)
* [bz2存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.tar.bz2)
* [zip存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.zip)

#### 0.0.3.2.5
* [tar存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.zip)

## 绘图处理器 (GPUs)

### Jcublas

#### 最新
* [tar存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.tar.gz)
* [bz2存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.tar.bz2)
* [zip存档](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.zip)

#### 0.0.3.2.5
* [tar存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.tar.gz)
* [bz2存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.tar.bz2)
* [zip存档](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.zip)
