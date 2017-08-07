---
title: Introduction to the Core DL4J Concepts
layout: default
---

# Introduction to the Core DL4J Concepts

> Before diving deeper into DL4J please make sure you have finished the 
> [Quickstart Guide](http://deeplearning4j.org/quickstart). This will ensure 
> that you have everything set up correctly and DL4J working smoothly.

> This guide assumes that you are using the newest release of DL4J. If you are
> not sure which version is the newest, clone the [examples](https://github.com/deeplearning4j/dl4j-examples)
> as shown in the quickstart guide and take a look at the `pom.xml` file.


## Overview

Every machine learning application consists of two parts. The first 
part is loading your data and preparing it to be used for learning. We
refer to this part as the ETL (extract, transform, load) process. 
[DataVec](http://deeplearning4j.org/simple-image-load-transform) is the library 
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
it using DataVec. This offers a lot more flexiblity, and retains the convenience of simple data loading.

Before you can start learning, you have to prepare your data, even if you already have a trained model. Preparing data means loading it and bringing it into the right shape and value
range. Implementing this on your own is very error prone, so use DataVec whereever possible.

Deep learning works with a lot of different data types, such as images, csv, arff, 
plain text and due to the upcoming [Apache Camel](https://camel.apache.org/) 
integration, pretty much any other data type you can think of.

In order to use DataVec you will need one of the implementations of the
[RecordReader](http://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/RecordReader.html)
interface along with the [RecordReaderDataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.html)
(see [Simple Image Load Transform](http://deeplearning4j.org/simple-image-load-transform) 
for a detailed explanation).

Once you have a [DataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html)
you can use it to retrieve your data in a format that is more suited for
training your model.


### Normalizing your Data

Neural networks work best when the data they are using is constrained to a
range between -1 and 1. The reason for this is that they are trained using
[gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and their 
activation functions usually having an active range somewhere between -1 and 1.
But even when using an activation function that doesn't saturate quickly, it is 
still good practice to constrain your values to this range; even then it typically improves performance.

Normalizing your data is pretty straight forward in DL4J. All you have to do is
to decide how you want to normalize your data, and set the coresponding 
[DataNormalization](http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/DataNormalization.html) up as a preprocessor for your DataSetIterator. Currently you
can choose from [ImagePreProcessingScaler](http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/ImagePreProcessingScaler.html), [NormalizerMinMaxScaler](http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/NormalizerMinMaxScaler.html) and [NormalizerStandardize](http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/NormalizerStandardize.html). 
The ImagePreProcessingScaler is obviously a good choice for image data, for
other data you the NormalizerMinMaxScaler is a good choice if you have a uniform
range along all dimensions of your input data and NormalizerStandardize is 
what you would usually use in other cases.

If you have other normalization needs, you are also free to implement the
DataNormalization interface.

If you end up using NormalizerStandardize, you should also notice that this is a
normalizer that depends on statistics that it extracts from the data. So will
have to save those statistics along with the model in order to restore them when
you restore your model.


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

DL4J allows you to build a deep learning model on a very high level. It uses a
builder pattern in order to declaratively build up the model, as you can see in
this (simplified) example:

~~~ java
MultiLayerConfiguration conf = 
	new NeuralNetConfiguration.Builder()
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		.updater(Updater.NESTEROVS).momentum(0.9)
		.learningRate(learningRate)
		.list(
			new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation("relu").build(),
			new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(numHiddenNodes).nOut(numOutputs).build()
		).backprop(true).build();
~~~

If you are familiar with other deep learning frameworks, you will notice that
this looks a bit like the python based Keras.

Unlike many other frameworks, DL4J has decided to split the optimization
algorithm from the updater algorithm. This allows you to be flexible while 
trying to find a combination that works best for your data and problem.

Besides the [DenseLayer](http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/DenseLayer.html)
and [OutputLayer](http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/OutputLayer.html)
that you have seen in the example above, there are several [other layer types](http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/package-summary.html),
like GravesLSTM, ConvolutionLayer, RBM, EmbeddingLayer, etc. . Using those 
layers you can define not only simple neural networks, but also [recurrent](http://deeplearning4j.org/usingrnns) 
and [convolutional](http://deeplearning4j.org/convolutionalnets) networks. 


## Training a Model

After defining your model, you will have to train it. The simplest case is to
simply call the `.fit()` method on the model configuration with your
DataSetIterator as an argument. This will train the model on all of your data
once. Such a single pass over the data is called an Epoch. DL4J has several
different methods of how you can pass over your data more than just once.

The simplest way, is to reset your DataSetIterator and loop over the fit call
as many times as you want. This way you can train your model for as many epochs
as you think is a good fit.

Another one of them is the `.iterations(N)` configuration parameter. It decides
how ofter the network should iterate (i.e. train) over a a single mini-batch in
a row. So, if you had 3 mini-batches A, B and C, setting `.iterations(3)` would
result in your network learning with the data as `AAABBBCCC`, in contrast using
3 epochs with `.iterations(1)` would feed the data to the network as `ABCABCABC`.

Yet another way would be to use an [EarlyStoppingTrainer](http://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/EarlyStoppingTrainer.html). 
You can configure this trainer to run for as many epochs as you like and
additionally for as long as you like. It will evaluate the performance of your
network after each epoch (or what ever you have configured) and save the best
performing version for later use. 

Also note that DL4J does not only support training just MultiLayerNetworks, but
it also supports a more flexible [ComputationGraph](http://deeplearning4j.org/compgraph).

### Evaluating model performance

As you train your model, you will want how well it currently performs. For this
you should have set aside dedicated data set that will not be used for training
but instead will only be used for evaluating your model. This data should have
the same distribution as the real life data will have when you wan to actually
use your model. The reason why you can't simply use your training data for
evaluation is because machine learning methods are prone to overfitting if they
are large enough.

Evaluating your model is done using the [Evaluation](http://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html)
class. Depending on if you want to evaluate a normal feed forward network or
a recurrent network you will have to use slightly different methods. For more
details on using it, take a look at the corresponding [examples](https://github.com/deeplearning4j/dl4j-examples).


## Troubleshooting your Model

Using neural networks to solve problems is a very empirical process. So you will
have to try different settings and architectures in order to find something that
performs best for you.

DL4J assists you in this endevor by providing a listener facility on your
network. You can set up listeners for your model, that will be called after each
mini-batch. The two most often used listeners that DL4J ships out of the box,
are [ScoreIterationListener](http://deeplearning4j.org/doc/org/deeplearning4j/optimize/listeners/ScoreIterationListener.html)
and [HistogramIterationListener](http://deeplearning4j.org/doc/org/deeplearning4j/ui/weights/HistogramIterationListener.html). While ScoreIterationListener will simply print
the current error score for your network, HistogramIterationListener will start
up a web ui that will provide you with a host of different information that you
can use to fine tune your network configuration. See [Visualize, Monitor and Debug Network Learning](http://deeplearning4j.org/visualization) 
on how to interpret that data.

See also [Troubleshooting neural nets](http://deeplearning4j.org/troubleshootingneuralnets) 
for more information on how to improve your results.
