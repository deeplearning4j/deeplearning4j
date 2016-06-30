---
title: Introduction to the Core DL4J Concepts
layout: default
---

# WORK IN PROGRESS

# Introduction to the Core DL4J Concepts

> Before diving deeper into DL4J please make sure you have finished the 
> [Quickstart Guide](http://deeplearning4j.org/quickstart). This will ensure 
> that you have everything set up correctly and DL4J working smoothly.

> This guide assumes that you are using the newest release of DL4J. If you are
> not sure which version is the newest, clone the examples as shown in the 
> quickstart guide and take a look at the `pom.xml` file.


## Overview

Every machine learning application consists of two parts. The first 
part is loading your data and preparing it to be used for learning. We
refer to this part as the ETL (extract, transform, load) process. 
[Canova](http://deeplearning4j.org/simple-image-load-transform) is the library 
we built to make this process easier. The second part is the actual learning system itself
- this is the core of DL4J.

All machine learning is based on vector maths, and DL4J requires
a library called [ND4J](http://nd4j.org/). It provides us with the ability to
work with arbitraty n-dimensional arrays (also called tensors), and thanks to its
different backends, it even allows use of both CPU and GPU resources.

When using DL4J you will often need all of these parts to get your work done
quickly and reliably. 


## Preparing your data for learning and prediction

Unlike other machine learning or deep learning frameworks, DL4J keeps loading data and training as separate processes. You don't simply point the model at data somewhere on the disk - instead you load 
it using Canova. This offers a lot more flexiblity, and retains the convenience of simple data loading.

Before you can start learning, you have to prepare your data, even if you already have a trained model. Preparing data means loading it and bring it into the right shape and value
range. Implementing this on your own is very error prone, so use Canova whereever possible.

Deep learning works with a lot of different data types, such as images, csv, arff, 
plain text and due to the upcoming [Apache Camel](https://camel.apache.org/) 
integration, pretty much any other data type you can think of.

In order to use Canova you will need one of the implementations of the
[RecordReader](http://deeplearning4j.org/canovadoc/org/canova/api/records/reader/RecordReader.html)
interface along with the [RecordReaderDataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/canova/RecordReaderDataSetIterator.html)
(see [Simple Image Load Transform](http://deeplearning4j.org/simple-image-load-transform) 
for a detailed explanation).

Once you have a [DataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html)
you can use it to retrieve your data in a format that is more suited for
training your model.


## Normalizing your Data

Neural networks work best when the data they are using is constrained to a
range between -1 and 1. The reason for this is that they are trained using
[gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and their 
activation functions usually having an active range somewhere between -1 and 1.
But even when using an activation function that doesn't saturate quickly, it is 
still good practice to constrain your values to this range; even then it typically improves performance.

TBC: Using Canova and Preprocessors for Normalization.


## DataSet, INDArray and Mini-Batches

As the name suggests, a DataSetIterator returns [DataSet](http://nd4j.org/doc/org/nd4j/linalg/dataset/DataSet.html)
objects. DataSet objects are just containers for the features and labels of your
data. But it isn't constrained to holding just a single example at once. Instead
a DataSet can contain as many examples as needed.

It does that by keeping the values in several instances of [INDArray](http://nd4j.org/doc/org/nd4j/linalg/api/ndarray/INDArray.html):
one for the features of your examples, one for the labels and two 
addional ones for masking if you are using timeseries data (see 
[Using RNNs / Masking](http://deeplearning4j.org/usingrnns#masking) for more 
information). 

An INDArray is one of the n-dimensional arrays, or tensors, provided by ND4J. In the case of the features, it is a matrix of the size 
`Number of Examples x Number of Features`. Even with only a single 
example, it will have this shape.

Why doesn't it contain all of the examples at once? This is another important concept for deep learning: mini-batching. In order to produce 
highly accurate results, a lot of real world training data is often needed. 
Often that is more data than can fit in available memory, so storing it in a
single DataSet sometimes isn't possible. But even if there is enough data storage, there is another important reason not to use all of your data
at once. With mini-batches you can get more updates to your model in a
single epoch.

So why bother having more than one example in a DataSet? Since the model
is trained using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), 
it requires a good gradient to follow. Using only one example at
a time will create a gradient that only takes errors in the current
example into consideration. This will make the learning behavior erratic
and slow down the learning considerably, and may even result in not
converging on a usable result.

A mini-batch should be large enough to provide a representative sample of the
real world (or at least your data). That means that it should always contain all
of the classes that you want to predict and that the count of those classes
should be distributed in approximately the same way as they are in your overall data.


## Building a Model

TBD: Builder Pattern for Declarative Model Building. Updater/Optimization duality. Layer types. CNN Setup Helper. Iterations vs Epochs.


## Training a Model

TBD: Listeners. Early Stopping.

## Troubleshooting your Model

TBD: Visualization UI

