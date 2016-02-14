---
layout: default
title: Dataset methods
---

# Dataset methods

The class *[DataSet](https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/DataSet.java)* is made to hold a pair of input feature/label matrices. Labels are binarized: all labels considered true are 1s; the rest are 0s.

A new DataSet can be created like this:

 <script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/fetchers/BaseDataFetcher.java?slice=57:70"></script>

The first parameter should specify the feature matrix, the second should point to the labels.

Once you have a DataSet, you'll want to know what to do with it. For neural nets to work well, DataSets require preprocessing, which standardizes data. Standardization means you'll be comparing apples to apples, and not some other strange fruit.

**MatrixUtil.toOutcomeVector** creates a binary matrix. If you have five labels, and you're dealing with label one, then the value of column zero is 1, and so forth. For example, the following matrix represents label three:

		[0,0,1,0,0]

That's for supervised learning (a.k.a. classification or labeling). 

[Deepautoencoders](../deepautoencoder.html) present another labeling problem. With that type of net, the input and the output are the same, because you're teaching them to reconstruct. 

Two main ways to standardize data are through mean removal and variance scaling. Mean removal subtracts the mean of the data from every point in the set, effectively centering the set on zero. Variance scaling will "normalize" your data so that its maximum is 1 and its minimum is -1. This is achieved by dividing all data points by the standard deviation. Both mean removal and variance scaling can be accomplished with one method: **normalizeZeroMeanZeroUnitVariance**.

Normalizing data is another way of saying you're putting it in the shape of a Bell curve. Sometimes you will want to inflate or deflate your DataSet to give it other shapes. Two methods that will do that are **multiplyBy** and **divideBy**. (These methods and others exist simply to make manipulating the internal feature matrix easier.)

You can also use **reshape** a DataSet by defining the number of rows and columns you want it to include.

The **shuffle** method is fairly self-explanatory. It rearranges the examples within a dataset around. That can be a useful step before you split the data into batches, or divide your test and training sets (see below).

The method **binarize** transforms continuous data into Boolean values by setting thresholds that will classify any data point below them as 0, and above them as 1.

Finally, you'll want to know how to separate your test set from your training set. This is done with the method **splitTestAndTrain**.

