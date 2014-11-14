---
title: 
layout: default
---

# Datasets & Downloads

We'll talk about three types of downloads here.  

### datasets

The first are preserialized datasets that can be downloaded directly for use with DL4J neural nets. Preserialized means they're in the correct format for ingestion. You can load them to RAM without using a dataset iterator — you don’t have to create them.  Here's how they can be loaded:

             DataSet d = new DataSet();
             BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
             d.load(bis);
             bis.close();

### distribution

Much like [nd4j backend downloads](http://nd4j.org/downloads.html) deeplearning4j needs an implementation of nd4j to use. Below are several binary bundles you can use bundled with different backends.

#Native

## Jblas

### Latest
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.bz2)
* [zip arcive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.zip)

### 0.0.3.2.5
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.bz2)
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

### models

Preserialized models, which we will upload shortly, are light-weight, pre-trained neural nets that you can simply call load on. 
