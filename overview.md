# What is DeepLearning4J

DeepLearning4J is a java based toolkit for building, training and deploying Neural Networks. 

# DeepLearning4J Components
 
DeepLearning4J has the following sub-projects. 

* **DataVec** for Data Ingestion Normalization and Transformation
* **DeepLearning4J** provides tools to build MultiLayerNetworks and Computation graphs
* **ND4J** allows Java to access Native Libraries to quickly process Matrix Data on CPU or GPU. 
* **Examples,** DeepLearning4J has a Github Repo of working examples.
* **Keras Model Import** to support the migration from Python and Keras to DeepLearning4J and Java. 

---------------------------

# DataVec

Ingesting cleaning, joining, scaling, normalizing and transforming data are jobs that must be done in any sort of Data analysis. This work is not exciting but it is necessary.

DataVec is our toolkit for that process.

## DataVec Examples

There are DataVec examples in our examples repo on github. 
[here](https://github.com/deeplearning4j/dl4j-examples) .

A descriptive summary of many of the examples is 
[here](https://github.com/deeplearning4j/examples-tour.md)

## Github Repo

The DataVec Github repo is [here](https://github.com/deeplearning4j/datavec).

## JavaDoc

DataVec JavaDoc is [here](./datavecdoc/). 


### DataVec overview

Neural Networks process multi Dimensional arrays of numeric data. Getting your data from a CSV file, or a directory of images serialized into Numeric Arrays is the job of DataVec. 


### DataVec Commonly used classes

Here is a list of some important DataVec classes


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

## Github Repo

The DeepLearning4J Github repo is [here](http://github.com/deeplearning4j/deeplearning4j).

## JavaDoc 

The DeepLearning4J JavaDoc is available [here](http://deeplearning4j.org/doc/)


## DeepLearning4J Examples


There are DeepLearning4J examples in our examples repo on github. 
[here](https://github.com/deeplearning4j/dl4j-examples) 

A descriptive summary of many of the examples is 
[here](https://github.com/deeplearning4j/examples-tour.md)

## DeepLearning4J frequently used classes

* **MultiLayerConfiguration** Configure a network
* **MultiLayerConfiguration.Builder** Builder interface to configure a network
* **MultiLayerNetwork** Builds a Network from the configuration
* **ComputationGraph** Builds a Computation Graph style Network
* **ComputationGraphConfiguration** Configuration for Computation Graph
* **ComputationGraphConfiguration.GraphBuilder** Builder interface for Computation Graph configuration
* **UiServer** Adds a web based Gui to view training parameter progress and configuration of Network

-----------

# ND4J

ND4J is the numerical processing library for DeepLearning4J

## Github Repo

The ND4J Github repo is [here](http://github.com/deeplearning4j/nd4j).

## JavaDoc 

The ND4J JavaDoc is available [here](http://nd4j.org/doc/)


## ND4J Examples

There are ND4J examples [here](https://github.com/deeplearning4j/dl4j-examples/tree/master/nd4j-examples)

## ND4J Frequently Used Classes

You may not use some of these classes directly, but when you configure a Neural Network. Behind the scenes the configurations you set for OptimizationAlgorithm, Updater, and LossFunction are all done in ND4J

* **DataSetPreProcessor** tools for Normalizing an image or numerical data
* **BaseTransformOp** Activation functions, tanh, sigmoid, relu, Softmax ...
* **GradientUpdater** Stochastic Gradient Descent, AdaGrad, Adam, Nesterovs ..

-------------------------

# Model Import

If you have worked with Keras the Python Library for Deeplearning and would like to import a trained model, or a model configuration into DeepLearning4J see our Model Import feature. 

## Github Repo

The Model Import is actually part of DeepLearning4J, but it is worth it's own section. Github folder is [here](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport).


## Model Import Examples

We will add examples [here](https://github.com/deeplearning4j/dl4j-examples/)

## Model Import Frequently Used Classes


* **KerasModel.Import** saved Keras Model to DeepLearning4J MultiLayerNetwork or Computation Graph





## Video 

A video demonstrating the import of a Keras model is available

<iframe width="560" height="315" src="http://www.youtube.com/watch?v=bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>
