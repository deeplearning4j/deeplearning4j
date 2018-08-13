---
title: Core Concepts in Deeplearning4j
short_title: Core Concepts
description: Introduction to core Deeplearning4j concepts.
category: Get Started
weight: 1
---

## Overview

Every machine-learning workflow consists of at least two parts. The first is loading your data and preparing it to be used for learning. We refer to this part as the ETL (extract, transform, load) process. [DataVec](./datavec-overview) is the library we built to make building data pipelines easier. The second part is the actual learning system itself. That is the algorithmic core of DL4J. 

All deep learning is based on vectors and tensors, and DL4J relies on a tensor library called [ND4J](./nd4j-overview). It provides us with the ability to work with *n-dimensional arrays* (also called tensors). Thanks to its different backends, it even enables us to use both CPUs and GPUs.  

## Preparing Data for Learning and Prediction

Unlike other machine learning or deep learning frameworks, DL4J treats the tasks of loading data and training algorithms as separate processes. You don't just point the model at data saved somewhere on disk, you load the data using DataVec. This gives you a lot more flexibility, and retains the convenience of simple data loading.

Before the algorithm can start learning, you have to prepare the data, even if you already have a trained model. Preparing data means loading it and putting it in the right shape and value range (e.g. normalization, zero-mean and unit variance). Building these processes from scratch is error prone, so use DataVec wherever possible.

Deeplearning4j works with a lot of different data types, such as images, CSV, ARFF, plain text and, with [Apache Camel](https://camel.apache.org/) [integration](https://github.com/deeplearning4j/DataVec/tree/master/datavec-camel), pretty much any other data type you can think of.

To use DataVec, you will need one of the implementations of the [RecordReader](http://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/RecordReader.html) interface along with the [RecordReaderDataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.html).

Once you have a [DataSetIterator](http://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html), which is just a pattern that describes sequential access to data, you can use it to retrieve the data in a format suited for training a neural net model.

### Normalizing Data

Neural networks work best when the data they're fed is normalized, constrained to a range between -1 and 1. There are several reasons for that. One is that nets are trained using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and their activation functions usually having an active range somewhere between -1 and 1. Even when using an activation function that doesn't saturate quickly, it is still good practice to constrain your values to this range to improve performance.

Normalizing data is pretty easy in DL4J. Decide how you want to normalize your data, and set the corresponding [DataNormalization](./datavec-normalization) up as a preprocessor for your DataSetIterator.

The `ImagePreProcessingScaler` is obviously a good choice for image data. The `NormalizerMinMaxScaler` is a good choice if you have a uniform range along all dimensions of your input data, and `NormalizerStandardize` is what you would usually use in other cases.

If you need other types of normalization, you are also free to implement the `DataNormalization` interface.

If you use `NormalizerStandardize`, note that this is a normalizer that depends on statistics that it extracts from the data. So you will have to save those statistics along with the model to restore them when you restore your model.

## DataSets, INDArrays and Mini-Batches

As the name suggests, a DataSetIterator returns [DataSet](http://deeplearning4j.org/api/{{page.version}}/org/nd4j/linalg/dataset/DataSet.html) objects. DataSet objects are containers for the features and labels of your data. But they aren't constrained to holding just a single example at once. A DataSet can contain as many examples as needed.

It does that by keeping the values in several instances of [INDArray](http://nd4j.org/api/{{page.version}}/org/nd4j/linalg/api/ndarray/INDArray.html): one for the features of your examples, one for the labels and two additional ones for masking, if you are using timeseries data (see [Using RNNs / Masking](./deeplearning4j-nn-recurrent) for more information).

An INDArray is one of the n-dimensional arrays, or tensors, used in ND4J. In the case of the features, it is a matrix of the size `Number of Examples x Number of Features`. Even with only a single example, it will have this shape.

Why doesn't it contain all of the data examples at once? 

This is another important concept for deep learning: mini-batching. In order to produce accurate results, a lot of real-world training data is often needed. Often that is more data than can fit in available memory, so storing it in a single `DataSet` sometimes isn't possible. But even if there is enough data storage, there is another important reason not to use all of your data at once. With mini-batches you can get more updates to your model in a single epoch.

So why bother having more than one example in a DataSet? Since the model is trained using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), it requires a good gradient to learn how to minimize error. Using only one example at a time will create a gradient that only takes errors produced with the current example into consideration. This would make the learning behavior erratic, slow down the learning, and may not even lead to a usable result.

A mini-batch should be large enough to provide a representative sample of the real world (or at least your data). That means that it should always contain all of the classes that you want to predict and that the count of those classes should be distributed in approximately the same way as they are in your overall data.

## Building a Neural Net Model

DL4J gives data scientists and developers tools to build a deep neural networks on a high level using concepts like `layer`. It employs a builder pattern in order to build the neural net declaratively, as you can see in this (simplified) example:

```java
MultiLayerConfiguration conf = 
	new NeuralNetConfiguration.Builder()
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		.updater(new Nesterovs(learningRate, 0.9))
		.list(
			new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation("relu").build(),
			new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(numHiddenNodes).nOut(numOutputs).build()
		).backprop(true).build();
```

If you are familiar with other deep learning frameworks, you will notice that this looks a bit like Keras.

Unlike other frameworks, DL4J splits the optimization algorithm from the updater algorithm. This allows for flexibility as you seek a combination of optimizer and updater that works best for your data and problem.

Besides the [DenseLayer](http://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/nn/conf/layers/DenseLayer.html)
and [OutputLayer](http://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/nn/conf/layers/OutputLayer.html)
that you have seen in the example above, there are several [other layer types](http://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/nn/conf/layers/package-summary.html), like `GravesLSTM`, `ConvolutionLayer`, `RBM`, `EmbeddingLayer`, etc. Using those layers you can define not only simple neural networks, but also [recurrent](http://deeplearning4j.org/usingrnns) and [convolutional](./deeplearning4j-nn-convolutional) networks. 

## Training a Model

After configuring your neural, you will have to train the model. The simplest case is to simply call the `.fit()` method on the model configuration with your `DataSetIterator` as an argument. This will train the model on all of your data
once. A single pass over the entire dataset is called an *epoch*. DL4J has several different methods for passing through the data more than just once.

The simplest way, is to reset your `DataSetIterator` and loop over the fit call as many times as you want. This way you can train your model for as many epochs as you think is a good fit.

Yet another way would be to use an [EarlyStoppingTrainer](http://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/earlystopping/trainer/EarlyStoppingTrainer.html). 
You can configure this trainer to run for as many epochs as you like and
additionally for as long as you like. It will evaluate the performance of your
network after each epoch (or what ever you have configured) and save the best
performing version for later use. 

Also note that DL4J does not only support training just `MultiLayerNetworks`, but it also supports a more flexible [ComputationGraph](./deeplearning4j-nn-computationgraph).

### Evaluating Model Performance

As you train your model, you will want to test how well it performs. For that test, you will need a dedicated data set that will not be used for training but instead will only be used for evaluating your model. This data should have the same distribution as the real-world data you want to make predictions about with your model. The reason you can't simply use your training data for evaluation is because machine learning methods are prone to overfitting (getting good at making predictions about the training set, but not performing well on larger datasets).

The [Evaluation](http://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/eval/Evaluation.html)
class is used for evaluation. Slightly different methods apply to evaluating a normal feed forward networks or recurrent networks. For more details on using it, take a look at the corresponding [examples](https://github.com/deeplearning4j/dl4j-examples).

## Troubleshooting a Neural Net Model

Building neural networks to solve problems is an empirical process. That is, it requires trial and error. So you will have to try different settings and architectures in order to find a neural net configuration that performs well.

DL4J provides a listener facility help you monitor your network's performance visually. You can set up listeners for your model that will be called after each mini-batch is processed. The two most often used listeners that DL4J ships out of the box are [ScoreIterationListener](http://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/optimize/listeners/ScoreIterationListener.html)
and [HistogramIterationListener](http://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/ui/weights/HistogramIterationListener.html). 

While `ScoreIterationListener` will simply print the current error score for your network, `HistogramIterationListener` will start up a web UI that to provide you with a host of different information that you can use to fine tune your network configuration. See [Visualize, Monitor and Debug Network Learning](./deeplearning4j-nn-visualization) on how to interpret that data.

See [Troubleshooting neural nets](./deeplearning4j-troubleshooting-training) for more information on how to improve results.