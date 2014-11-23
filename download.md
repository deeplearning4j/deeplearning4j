---
title: 
layout: default
---

# ND4J

DeepLearning4j uses [ND4J](http://nd4j.org/) as a computational kernel for matrix operations. To get started with Deeplearning4j, you need to pick an [ND4J backend](http://nd4j.org/downloads.html), which will rely on GPUs or native computations. 

##Why Swappable Backends?

Many deep learning researchers have standardized on Cuda GPUs for parallel processing and matrix computations. Unfortunately, industry practicioners have to grapple with more limited options and legacy hardware. If you have a lot of legacy hardware, however, throwing CPUs at a deep-learning problem can work. 

We created ND4J because no JVM Blas-based libraries allowed users to have a swappable interface for different fast-matrix operations. Swappable backends (a la [SLF4J](http://slf4j.org/)) was the only answer. 

In addition, we felt a common API for creating machine-learning algorithms was a worthy goal. No one wants to rewrite their libraries if they find that their matrix run-time is faster.

##Downloads

Below, you will find bundled downloads of deeplearning4j for GPUs and native, among other components.

Much like [ND4J backend downloads](http://nd4j.org/downloads.html), Deeplearning4j needs an implementation of ND4J to use. Below are several binary bundles you can use bundled with different backends.

#Native

## Jblas

### Latest
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.zip)

### 0.0.3.2.5
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.zip)

## Netlib Blas

### Latest
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.zip)

### 0.0.3.2.5
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.zip)

# GPUs

## Jcublas

###Latest
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.zip)

###0.0.3.2.5
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.zip)
