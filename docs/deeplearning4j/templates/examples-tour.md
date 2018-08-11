---
title: Tour of Eclipse Deeplearning4j Examples
short_title: Examples Tour
description: Brief tour of available examples in DL4J.
category: Get Started
weight: 10
---

## Survey of DeepLearning4j Examples

Deeplearning4j's Github repository has many examples to cover its functionality. The [Quick Start Guide](https://deeplearning4j.org/quickstart) shows you how to set up Intellij and clone the repository. This page provides an overview of some of those examples.

## DataVec examples

Most of the examples make use of DataVec, a toolkit for preprocessing and clearning data through normalization, standardization, search and replace, column shuffles and vectorization. Reading raw data and transforming it into a DataSet object for your Neural Network is often the first step toward training that network. If you're unfamiliar with DataVec, here is a description and some links to useful examples. 

### IrisAnalysis.java

This example takes the canonical Iris dataset of the flower species of the same name, whose relevant measurements are sepal length, sepal width, petal length and petal width. It builds a Spark RDD from the relatively small dataset and runs an analysis against it. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/analysis/IrisAnalysis.java)

### BasicDataVecExample.java

This example loads data into a Spark RDD. All DataVec transform operations use Spark RDDs. Here, we use DataVec to filter data, apply time transformations and remove columns.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/basic/BasicDataVecExample.java)

### PrintSchemasAtEachStep.java

This example shows the print Schema tools that are useful to visualize and to ensure that the code for the transform is behaving as expected. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/debugging/PrintSchemasAtEachStep.java)

### JoinExample.java

You may need to join datasets before passing to a neural network. You can do that in DataVec, and this example shows you how. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/join/JoinExample.java)

### LogDataExample.java

This is an example of parsing log data using DataVec. The obvious use cases are cybersecurity and customer relationship management. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/logdata/LogDataExample.java)

### MnistImagePipelineExample.java

This example is from the video below, which demonstrates the ParentPathLabelGenerator and ImagePreProcessing scaler. 

<iframe width="560" height="315" src="http://www.youtube.com/embed/GLC8CIoHDnI" frameborder="0" allowfullscreen></iframe>

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/MnistImagePipelineExample.java)

### PreprocessNormalizerExample.java

This example demonstrates preprocessing features available in DataVec.

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/PreprocessNormalizerExample.java)

### CSVExampleEvaluationMetaData.java

DataMeta data tracking - i.e. seeing where data for each example comes from - is useful when tracking down malformed data that causes errors and other issues. This example demostrates the functionality in the RecordMetaData class. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExampleEvaluationMetaData.java)

---

## DeepLearning4J Examples

To build a neural net, you will use either `MultiLayerNetwork` or `ComputationGraph`. Both options work using a Builder interface. A few highlights from the examples are described below. 

### MNIST dataset of handwritten digits

MNIST is the "Hello World" of deep learning. Simple, straightforward, and focussed on image recognition, a task that Neural Networks do well. 

### MLPMnistSingleLayerExample.java

This is a Single Layer Perceptron for recognizing digits. Note that this pulls the images from a binary package containing the dataset, a rather special case for data ingestion.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java)

### MLPMnistTwoLayerExample.java

A two-layer perceptron for MNIST, showing there is more than one useful network for a given dataset. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistTwoLayerExample.java)

### Feedforward Examples

Data flows through feed-forward neural networks in a single pass from input via hidden layers to output.

These networks can be used for a wide range of tasks depending on they are configured. Along with image classification over MNIST data, this directory has examples demonstrating regression, classification, and anomoly detection.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward)

### Convolutional Neural Networks

Convolutional Neural Networks are mainly used for image recognition, although they apply to sound and text as well. 

### AnimalsClassification.java

This example can be run using either LeNet or AlexNet. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/AnimalsClassification.java)

---

## Saving and Loading Models

Training a network over a large volume of training data takes time. Fortunately, you can save a trained model and
load the model for later training or inference.

### SaveLoadComputationGraph.java

This demonstrates saving and loading a network build using the class ComputationGraph.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/modelsaving/SaveLoadComputationGraph.java)

### SaveLoadMultiLayerNetwork.java

Demonstrates saving and loading a Neural Network built with the class MultiLayerNetwork.

### Saving/loading a trained model and passing it new input 

Our video series shows code that includes saving and loading models, as well as inference. 

[Our YouTube channel](https://www.youtube.com/channel/UCa-HKBJwkfzs4AgZtdUuBXQ)

---

## Custom Loss Functions and Layers

Do you need to add a Loss Function that is not available or prebuilt yet? Check out these examples.

### CustomLossExample.java

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/lossfunctions/CustomLossExample.java)

### CustomLossL1L2.java

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/lossfunctions/CustomLossL1L2.java)

### Custom Layer

Do you need to add a layer with features that aren't available in DeepLearning4J core? This example show where to begin. 

### CustomLayerExample.java

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/customlayers/CustomLayerExample.java)

---

## Natural Language Processing

Neural Networks for NLP? We have those, too.

### GloVe 

Global Vectors for Word Representation are useful for detecting relationships between words. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/glove/GloVeExample.java)

### Paragraph Vectors

A vectorized representation of words. Described [here](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/paragraphvectors/ParagraphVectorsClassifierExample.java)

### Sequence Vectors

One way to represent sentences is as a sequence of words. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/sequencevectors/SequenceVectorsTextExample.java)

### Word2Vec

Described [here](https://deeplearning4j.org/word2vec.html)

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec/Word2VecRawTextExample.java)

---

## Data Visualization

t-Distributed Stochastic Neighbor Embedding (t-SNE) is useful for data visualization. We include an example in the NLP section since word similarity visualization is a common use. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/tsne/TSNEStandardExample.java)

---

## Recurrent Neural Networks

Recurrent Neural Networks are useful for processing time series data or other sequentially fed data like video. 

The examples folder for Recurrent Neural Networks has the following:

### BasicRNNExample.java

An RNN learns a string of characters.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/basic/BasicRNNExample.java)

### GravesLSTMCharModellingExample.java

Takes the complete works of Shakespeare as a sequence of characters and Trains a Neural Net to generate "Shakespeare" one character at a time.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/GravesLSTMCharModellingExample.java)

### SingleTimestepRegressionExample.java

Regression with an LSTM (Long Short Term Memory) Recurrent Neural Network. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/regression/SingleTimestepRegressionExample.java)

### AdditionRNN.java

This example trains a neural network to do addition. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seq2seq/AdditionRNN.java)

### RegressionMathFunctions.java

This example trains a neural network to perform various math operations. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/regression/RegressionMathFunctions.java)

### UCISequenceClassificationExample.java

A publicly available dataset of time series data of six classes, cyclic, up-trending, etc. Example of an RNN learning to classify the time series. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seqclassification/UCISequenceClassificationExample.java)

### VideoClassificationExample.java

How do autonomous vehicles distinguish between a pedestrian, a stop sign and a green light? A complex neural net using Convolutional and Recurrent layers is trained on a set of training videos. The trained network is passed live onboard video and decisions based on object detection from the Neural Net determine the vehicles actions. 

This example is similar, but simplified. It combines convolutional, max pooling, dense (feed forward) and recurrent (LSTM) layers to classify frames in a video. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/video/VideoClassificationExample.java)

### SentimentExampleIterator.java

This sentiment analysis example classifies sentiment as positive or negative using word vectors and a Recurrent Neural Network. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/Word2VecSentimentRNN.java)

---

## Distributed Training on Spark

DeepLearning4j supports using a Spark Cluster for network training. Here are the examples. 

### MnistMLPExample.java

This is an example of a Multi-Layer Perceptron training on the Mnist data set of handwritten digits. 
[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/mlp/MnistMLPExample.java)

### SparkLSTMCharacterExample.java

An LSTM recurrent Network in Spark. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/rnn/SparkLSTMCharacterExample.java)

---

## ND4J Examples

ND4J is a tensor processing library. It can be thought of as Numpy for the JVM. Neural Networks work by processing and updating MultiDimensional arrays of numeric values. In a typical Neural Net application you use DataVec to ingest and convert the data to numeric. Classes used would be RecordReader. Once you need to pass data into a Neural Network, you typically use RecordReaderDataSetIterator. RecordReaderDataSetIterator returns a DataSet object. DataSet consists of an NDArray of the input features and an NDArray of the labels. 

The learning algorithms and loss functions are executed as ND4J operations. 

### Basic ND4J examples

This is a directory with examples for creating and manipulating NDArrays.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/tree/master/nd4j-examples/src/main/java/org/nd4j/examples)

---

## Reinforcement Learning Examples

Deep learning algorithms have learned to play Space Invaders and Doom using reinforcement learning. DeepLearning4J/RL4J examples of Reinforcement Learning are available here: 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/tree/master/rl4j-examples)