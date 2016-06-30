# WORK IN PROGRESS

# Introduction to the Core DL4J Concepts

> Before diving deeper into DL4J please make sure you have finished the 
> [Quickstart Guide](http://deeplearning4j.org/quickstart). This will ensure 
> that you have everything setup correctly and DL4J works for you.

> This guide assumes that you are using the newest release of DL4J. If you are
> not sure which version is the newest, clone the examples as shown in the 
> quickstart guide and take a look at the `pom.xml` file.


## Overview

Every machine learning application usually consists of two parts. The first 
part is loading your data and getting it ready to be used for learning. We
refer to this part as the ETL (extract, transform, load) process and 
[Canova](http://deeplearning4j.org/simple-image-load-transform) is our library 
to make this process easier. The other part is the actual learning system itself
and this at the core of deeplearning4j.

As all of machine learning is based on vector maths, deeplearning4j requires
a library called [ND4J](http://nd4j.org/). It provides us with the ability to
work with arbitraty n-dimensional arrays (also called tensors), and due to its
different backends even allows us to use both CPU or GPU ressources.

When using DL4J you will often encouter all of this parts to get your work done
quickly and reliably. 


## Preparing your data for learning and prediction

In contrast to other machine learning / deep learning frameworks, we have 
decided that loading data and training should be as seperate as possible, so you
don't just point the model at some data somewhere on the disk, instead you load 
it using Canova. This way you can be a lot more flexible when you need to, and
still have the convenience of simple data loading.

Before you can start learning, and even when you already have a trained model,
you have to prepare your data. Preparing data means that it has so be loaded
from somewhere and that it has to be brought into the right shape and value
range. Implementing this on your own is very error prone and takes a lot of 
effort to do right, for this reason you should use Canova where ever possible.

Deep learning works with a lot of different data types, like images, csv, arff, 
plain text and due to the upcoming [Apache Camel](https://camel.apache.org/) 
integration, pretty much everything that you can think of.

In order to use Canova you will have to use one of the implementations of the
[RecordReader](http://deeplearning4j.org/canovadoc/org/canova/api/records/reader/RecordReader.html)
interface along with the [RecordReaderDataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/canova/RecordReaderDataSetIterator.html)
(see [Simple Image Load Transform](http://deeplearning4j.org/simple-image-load-transform) 
for a detailed explanation).

Once you have a [DataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html)
you can use it to retrieve your data in a format that is more suited for
training your model.


## Normalizing your Data

Neural networks work best, when the data they are using is constrained to a
range between -1 and 1. The reason for this is that they are trained using
[gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and their 
activation functions usually having an active range somewhere between -1 and 1.
But even when using an activation function that doesn't saturate quickly, it is 
still good practice to constrain your values to this range, as even there it
usually improves performance.

TBC: Using Canova and Preprocessors for Normalization.


## DataSet, INDArray and Mini-Batches

As the name allows to guess, a DataSetIterator returns [DataSet](http://nd4j.org/doc/org/nd4j/linalg/dataset/DataSet.html)
objects. DataSet objects are just containers for the features and labels of your
data. But it isn't constrained to holding just a single example at once. Instead
a DataSet can contain as many examples as needed.

It does that by keeping the values in several instances of [INDArray](http://nd4j.org/doc/org/nd4j/linalg/api/ndarray/INDArray.html).
One for the features of your examples, one for the labels of them and two 
addional ones for masking if you are going to be using timeseries data (see 
[Using RNNs / Masking](http://deeplearning4j.org/usingrnns#masking) for more 
information on that). 

An INDArray is exactly one of those n-dimensional arrays, or tensors, that ND4J
provides for us. In the case of the features, it is a matrix of the size 
`Number of Examples x Number of Features`. Even if you only have a single 
example in there, it will be having this shape.

Why doesn't it contain all of the examples at once? The reason for it is another
very important concept in deep learning: Mini-Batching. In order to produce 
great results, often a lot of real world data is needed to train a model. 
Usually that is more data than can fit in your memory, so having all of it in a
single DataSet often isn't even possible. But even if all of your data does fit
into memory, there is another very important reason not to use all of your data
at once. By using mini-batches you can get more updates to your model in a
single epoch.

But why then even bother having more than one example in a DataSet? As a model
is trained using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), 
it is necessary to get a good gradient to follow. Using only a single example at
a time, will create a gradient that only takes the errors for the current
example into consideration. This will make the learning behavior very erratic,
and it will slow down the learning considerably, and may even result in not
converging on a usable result.

A mini-batch should be large enough to provide a representative sample of the
real world (or at least your data). That means that it should always contain all
of the classes that you are going to predict and that the count of those classes
should be approximately distributed the same as they are in your overall data.


## Building a Model

TBD: Builder Pattern for Declarative Model Building. Updater/Optimization duality. Layer types. CNN Setup Helper. Iterations vs Epochs.


## Training a Model

TBD: Listeners. Early Stopping.

## Troubleshooting your Model

TBD: Visualization UI

