# What Is DeepLearning4j?

DeepLearning4J is a java based toolkit for building, training and deploying Neural Networks. 

### DeepLearning4j Components
 
DeepLearning4J has the following sub-projects. 

* **DataVec** performs data ingestion, normalization and transformation into feature vectors
* **DeepLearning4j** provides tools to configure neural networks and build computation graphs
* **Keras Model Import** helps import trained models from Python and Keras to DeepLearning4J and Java. 
* **ND4J** allows Java to access Native Libraries to quickly process Matrix Data on CPUs or GPUs. 
* **DL4J-Examples** contains working examples for classification and clustering of images, time series and text.
* **ScalNet** is a Scala wrapper for Deeplearning4j inspired by Keras. Runs on multi-GPUs with Spark.
* **RL4J** implements Deep Q Learning, A3C and other reinforcement learning algorithms for the JVM.
* **Arbiter** helps search the hyperparameter space to find the best neural net configuration.

---------------------------

## DataVec

Ingesting, cleaning, joining, scaling, normalizing and transforming data are jobs that must be done in any sort of analysis. This work may not be exciting, but it's a precondition of deep learning. DataVec is our toolkit for that process. We give data scientists and developers tools to turn raw data such as images, video, audio, text and time series into feature vectors for neural nets. 

### Github Repo

The DataVec Github repo is [here](https://github.com/deeplearning4j/datavec). Here is how the repo breaks down.

* [datavec-dataframe](https://github.com/deeplearning4j/DataVec/tree/master/datavec-dataframe) : DataVec's built-in equivalent of Pandas Dataframe
* [datavec-api](https://github.com/deeplearning4j/DataVec/tree/master/datavec-api) : rules for preprocessing data and defining data pipelines.
* [datavec-data](https://github.com/deeplearning4j/DataVec/tree/master/datavec-data) : knows how to understand audio, video, images, text data types
* [datavec-spark](https://github.com/deeplearning4j/DataVec/tree/master/datavec-spark) : runs distributed data pipelines on Spark 
* [datavec-local](https://github.com/deeplearning4j/DataVec/tree/master/datavec-local) : runs Datavec standalone on desktop. For inference. 
* [datavec-camel](https://github.com/deeplearning4j/DataVec/tree/master/datavec-camel) : connects to external Camel components. Camel allows you to define routes and integrates with many data sources. DataVec-camel sends data to datavec as a destination from whichever Camel source you specify.

### DataVec Examples

There are DataVec examples in our examples repo on Github [here](https://github.com/deeplearning4j/dl4j-examples).

A descriptive summary of many of the examples is [here](examples-tour).

### JavaDoc

DataVec JavaDoc is [here](./datavecdoc/). 

### DataVec overview

Neural Networks process multi-dimensional arrays of numerical data. Getting your data from a CSV file, or a directory of images serialized into numeric arrays is the job of DataVec. 

### DataVec: Commonly used classes

Here's a list of some important DataVec classes:

* Input Split

Splitting data into Test and Train

* **InputSplit.sample** to split data into Test and Train

Randomize Data

* **FileSplit.random** to randomize data

Base class for reading and serializing data. RecordReaders ingest your data input and return a List of Serializable objects (Writables). 

* **RecordReader**

Implementations of particular RecordReaders

* **CSVRecordReader** for CSV data
* **CSVNLinesSequenceRecordReader** for Sequence Data
* **ImageRecordReader** for images
* **JacksonRecordReader** for JSON data
* **RegexLineRecordReader** for parsing log files
* **WavFileRecordReader** for audio files
* **LibSvmRecordReader** for Support Vector Machine
* **VideoRecordReader** for reading Video

For re-organizing, joining, Normalizing and transforming data. 

* **Transform**

Specific Transform implementations

* **CategoricalToIntegerTransform** to convert category names to integers
* **CategoricalToOneHotTransform** convert catagory name to onehot representation
* **ReorderColumnsTransform** rearrange columns
* **RenameColumnsTransform** rename columns
* **StringToTimeTransform** convert timestring

The labels for data input may be based on the directory where the image is stored. 

* **ParentPathLabelGenerator** Label based on parent directory
* **PatternPathLabelGenerator** Derives label based on a string within the file path

DataNormalization

* **Normalizer**  Although part of  ND4J seems like it should be mentioned here

-------------------------

# DeepLearning4J

DeepLearning4J is where you design your Neural Networks

### Github Repo

The DeepLearning4J Github repo is [here](http://github.com/deeplearning4j/deeplearning4j). Here's how the repo breaks down.

* [deeplearning4j-core](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core) : datasetiterators and everything you need to run dl4j on the desktop. 
* [deeplearning4j-cuda](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-cuda) : cudnn and anything cuda specific.
* [deeplearning4j-graph](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-graph) : graph processing for deepwalk.
* [deeplearning4j-modelimport](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport): this imports neural net models from Keras, which in turn can import models from major frameworks like Theano, Tensorflow, Caffe and CNTK
* [deeplearning4j-nlp-parent](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nlp-parent): text analytics for English, Japanese and Korean as well as external tokenizers and plugins to toolsets like like UIMA, which itself performs dependency parsing, semantic role labeling, relation extraction and QA systems. We integrate with toolsets like UIKMA to pass stuff to word2vec.
* [nlp](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nlp-parent/deeplearning4j-nlp): Word2vec, doc2vec and other tools.
* [deeplearning4j-nn](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nn) : a pared-down neural net DSL with fewer dependencies. Configures multilayer nets with a builder pattern for setting hyperparameters.
* [deeplearning4j-scaleout](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout) : AWS provisioning, parallelwrapper desktop parameter averaging (single box 96 cores) so you don't have to run spark if you don't want to; one for parameter server and the other not; streaming is kafka and spark streaming; spark is spark training and nlp on spark: dist. word2vec
* [deeplearning4j-ui-parent](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-ui-parent) : neural net training heuristics and visualization

### JavaDoc 

The DeepLearning4J JavaDoc is available [here](http://deeplearning4j.org/doc/).

### DeepLearning4J Examples

There are DeepLearning4J examples in the Github repository [here](https://github.com/deeplearning4j/dl4j-examples). 

A descriptive summary of many of the examples is [here](examples-tour).

### DeepLearning4J frequently used classes

* **MultiLayerConfiguration** Configure a network
* **MultiLayerConfiguration.Builder** Builder interface to configure a network
* **MultiLayerNetwork** Builds a Network from the configuration
* **ComputationGraph** Builds a Computation Graph style Network
* **ComputationGraphConfiguration** Configuration for Computation Graph
* **ComputationGraphConfiguration.GraphBuilder** Builder interface for Computation Graph configuration
* **UiServer** Adds a web based Gui to view training parameter progress and configuration of Network

## Model Import

If you have worked with Keras the Python Library for Deeplearning and would like to import a trained model, or a model configuration into DeepLearning4J see our Model Import feature. 

### Github Repo

The Model Import is actually part of DeepLearning4J, but it is worth it's own section. Github folder is [here](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport).

### Model Import Examples

We will add examples [here](https://github.com/deeplearning4j/dl4j-examples/)

### Model Import Frequently Used Classes

* **KerasModel.Import** saved Keras Model to DeepLearning4J MultiLayerNetwork or Computation Graph

### Video 

Here's a video showing how to import of a Keras model to DL4J:

<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

-----------

## ND4J

ND4J is the numerical processing library for DeepLearning4J, a tensor library, the JVM's answer to Numpy.  

### Github Repo

The ND4J Github repo is [here](http://github.com/deeplearning4j/nd4j). ND4J is a DSL for handling n-dimensional arrays (NDArrays).

* [nd4j-parameter-server-parent](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-parameter-server-parent) : a robust parameter server for distributed neural net training using Aeron.
* [nd4j-backends](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends) : hardware specific backends the optimize on GPUs and CPUs.

### JavaDoc 

The ND4J JavaDoc is available [here](http://nd4j.org/doc/).

### ND4J Examples

There are ND4J examples [here](https://github.com/deeplearning4j/dl4j-examples/tree/master/nd4j-examples).

### ND4J Frequently Used Classes

You may not use some of these classes directly, but when you configure a Neural Network. Behind the scenes the configurations you set for OptimizationAlgorithm, Updater, and LossFunction are all done in ND4J.

* **DataSetPreProcessor** tools for Normalizing an image or numerical data
* **BaseTransformOp** Activation functions, tanh, sigmoid, relu, Softmax ...
* **GradientUpdater** Stochastic Gradient Descent, AdaGrad, Adam, Nesterovs ..

-------------------------

## ScalNet

ScalNet is Keras for Scala. It's a Scala wrapper for Deeplearning4j that can run Spark on multi-GPUs.

### Github Repo

* [ScalNet on Github](https://github.com/deeplearning4j/ScalNet)

## RL4J 

RL4J is a library and environment for for Deep Q learning, A3C and other algorithms implemented in Java and integrated with DL4J and ND4J. 

### Github Repo

* [RL4J](https://github.com/deeplearning4j/rl4j)
* [Gym integration](https://github.com/deeplearning4j/rl4j/tree/master/rl4j-gym)
* [Doom with RL4J](https://github.com/deeplearning4j/rl4j/tree/master/rl4j-doom)

## Arbiter

Arbiter helps you search the hyperparameter space to find the best tuning and architecture for a neural net. This is important because finding the right architecture and hyperparamters is a very large combinatorial problem. The winning ImageNet teams at corporate labs like Microsoft are searching through hyperparameters to surface 150-layer networks like ResNet.

### Github Repo

The Github repository is [here](https://github.com/deeplearning4j/Arbiter).

* [arbiter-core](https://github.com/deeplearning4j/Arbiter/tree/master/arbiter-core) : Aribter-core searches the hyperparameter space with algorithms like grid search. Provides a GUI.
* [arbiter-deeplearning4j](https://github.com/deeplearning4j/Arbiter/tree/master/arbiter-deeplearning4j) : Arbiter can talk to DL4J models. When you do model search, you need to be able to run the model. This pilots the model and finds the best model.
