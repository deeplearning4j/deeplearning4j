# Survey of DeepLearning4J Examples

We have a Github repository with many examples of DeepLearning4j functionality. Our [Quick Start Guide](https://deeplearning4j.org/quickstart) shows you how to set up Intellij and clone the repository. This page provides an overview of some of those examples.

## DataVec examples

Most of the examples make use of DataVec, since reading raw data and transforming it into a DataSet object for the Neural Network is often the first step. If you are unfamiliar with DataVec here is a description and links to some examples that will be useful. 

### IrisAnalysis.java

An example that takes the canonical Iris Dataset of Iris species measurements of sepal length, sepal width, 
petal length and petal width. This example builds a Spark RDD from the relatively small dataset and runs some analysis against it. 

* [Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/analysis/IrisAnalysis.java)

### BasicDataVecExample.java

This example loads data into a Spark RDD. All DataVec transform operations use Spark RDDs. In this example, we use DataVec to filter data, apply time transformations and remove columns.

* [Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/basic/BasicDataVecExample.java)

### PrintSchemasAtEachStep.java

This example shows the print Schema tools that are useful to visualize and ensure that the code to transform is behaving as expected. 

* [Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/debugging/PrintSchemasAtEachStep.java)

### JoinExample.java

You may need to join datasets before passing to A Neural Network. You can do that in DataVec, this example shows you how. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/join/JoinExample.java)

### LogDataExample.java

An example of parsing Log Data using DataVec

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/datavec-examples/src/main/java/org/datavec/transform/logdata/LogDataExample.java)

### MnistImagePipelineExample.java

This example is from the video here
<iframe width="560" height="315" src="http://www.youtube.com/embed/GLC8CIoHDnI" frameborder="0" allowfullscreen></iframe>


The Video demonstrates ParentPathLabelGenerator, and ImagePreProcessing scaler. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/MnistImagePipelineExample.java)

### PreprocessNormalizerExample.java

A nice example demonstrating pre-processing features available in DataVec.

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/PreprocessNormalizerExample.java)

### CSVExampleEvaluationMetaData.java

DataMeta data tracking - i.e., where data for each example comes from is useful when tracking down malformed data that causes errors or other issues. 

This example demostrates the functionality in the RecordMetaData class. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/CSVExampleEvaluationMetaData.java)

-------------------

# DeepLearning4J Examples


To build a Neural Net you will use either MultiLayerNetwork, or ComputationGraph. 

Both options include a  convenient Builder interface. 

A few highlights from the examples are described below. 

### MNIST dataset of handwritten digits

This has become the "hello world" of deep learning. Simple, straightforward, and focussed on a task that Neural Networks perform well on. 

### MLPMnistSingleLayerExample.java

Single Layer Perceptron for recognizing digits. Note that this pulls the images from a binary package containing the dataset, a rather special case for data ingestion.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java)

### MLPMnistTwoLayerExample.java

Two layer perceptron for MNist, showing there is more than one way to build a useful Network. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistTwoLayerExample.java)

### Feedforward Examples

FeedForward Neural Networks have a single pass from input to hidden layers to output.

They can be used for a wide range of tasks depending on they are configured. Along with the image classification over Mnist data, this directory has examples demonstrating regression, classification, and anomoly detection.


[Show me the code](http://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward)

### Convolutional Neural Networks

Convolutional Neural Networks are useful for image recognition. 

### AnimalsClassification.java

This example can be run using either Lenet, or AlexNet. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/AnimalsClassification.java)

---------------

# Saving and Loading Models

Training a Network over a large volume of training data takes time. Fortunately you can save a trained model and
load the model for later training or for inference

### SaveLoadComputationGraph.java

Demonstrates saving and loading a network build using ComputationGraph, 
[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/modelsaving/SaveLoadComputationGraph.java)

### SaveLoadMultiLayerNetwork.java

Demostrates saving and loading a Neural Network built with MultiLayer Network


### Saving/loading a trained model and passing it new input. 

Our video series demonstrates code that includes saving loading and inference. 

[Our youtube channel](https://www.youtube.com/channel/UCa-HKBJwkfzs4AgZtdUuBXQ)

----------

# Custom Loss Functions and Layers

Need to add a Loss Functions that is not yet available prebuilt? 

Check out these examples

### CustomLossExample.java

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/lossfunctions/CustomLossExample.java)

### CustomLossL1L2.java

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/lossfunctions/CustomLossL1L2.java)

### Custom Layer

Need to add a Layer with features that are not available in DeepLearning4J core? This example show where to begin. 

### CustomLayerExample.java

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/customlayers/CustomLayerExample.java)

-----------

# Natural Language Processing

Neural Networks for NLP? Sure enough here are the examples. 

### GloVe 

Global Vectors for Word Representation are useful for detecting the relationships between words. 

Our example is here.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/glove/GloVeExample.java)

### Paragraph Vectors

A vectorized representation of words. Describe (here.)[https://cs.stanford.edu/~quocle/paragraph_vector.pdf]


[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/paragraphvectors/ParagraphVectorsClassifierExample.java)

### Sequence Vectors

One way to represent sentences is a sequence of words. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/sequencevectors/SequenceVectorsTextExample.java)

### Word2Vec

Described (here)[https://deeplearning4j.org/word2vec.html]


[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec/Word2VecRawTextExample.java)

-------------------------

# Data Visualization

t-Distributed Stochastic Neighbor Embedding (t-SNE) is useful for data visualization. We include an example in the NLP section since word similarity visualization is a common use. 


[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/tsne/TSNEStandardExample.java)

------------------------

# Recurrent Neural Networks

Recurrent Neural Networks are useful for processing Sequential or time series data. 

The examples folder for Recurrent Neural Networks has the following

### BasicRNNExample.java

An RNN learns a string of characters.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/basic/BasicRNNExample.java)

 
### GravesLSTMCharModellingExample.java

Takes the complete works of Shakespeare as a sequence of characters and Trains a Neural Net to generate "Shakespeare" one character at a time.

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/GravesLSTMCharModellingExample.java)

### SingleTimestepRegressionExample.java

Regression with a LSTM (Long Short Term Memory) Recurrent Neural Network. 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/regression/SingleTimestepRegressionExample.java)

### AdditionRNN.java

Trains a Neural Net to do addition. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seq2seq/AdditionRNN.java)

### RegressionMathFunctions.java

Trains a Neural Net to do various math operations. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/regression/RegressionMathFunctions.java)

### UCISequenceClassificationExample.java

A publicly available dataset of timeseries data of 6 classes, cyclic, up-trending, etc.

Example of an RNN learning to classify the time series. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seqClassification/UCISequenceClassificationExample.java)

### VideoClassificationExample.java

How do autonomous vehicles determine the difference between a pedestrian, and a stop sign? A complex Neural net using Convolutional and Recurrent layers is trained on a set of  training videos. The trained network is passed live onboard video and decisions based on object detection from the Neural Net determine the vehicles actions. 

This is an example is similar but a simplified. The example combines convolutional, max pooling, dense (feed forward) and recurrent (LSTM) layers to classify frames in a video. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/video/VideoClassificationExample.java)

### SentimentExampleIterator.java

Sentiment analysis example that classifies sentiment as positive or negative using word vectors and a Recurrent Neural Network. 

[Show me the code](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/Word2VecSentimentRNN.java)

---------------

# Distributed Training on Spark

DeepLearning4j supports using a Spark Cluster for network training. 

Here are the examples. 

### MnistMLPExample.java

An example of a Multi Layer Perceptron training on the Mnist data set of handwritten digits. 
[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/mlp/MnistMLPExample.java)




### SparkLSTMCharacterExample.java

An LSTM recurrent Network in Spark. 
[Show me the code](http://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/rnn/SparkLSTMCharacterExample.java)

---------------------

# ND4J Examples

ND4J is the Matrix processing library we have built. It can be thought of as NumPY for java. Neural Networks work by processing and updating MultiDimensional arrays of numeric values. In a typical Neural Net application you use DataVec to ingest and convert the data to numeric. Classes used would be recordreader. Once you need to pass data into a Neural Network you typically use RecordReaderDataSetIterator. REcordReaderDataSetIterator returns a DataSet object. DataSet consists of an NDArray of the input features and an NDArray of the Labels. 

The learning algorithms, Loss Functions, are executed as ND4J operations. 

### Basic ND4J examples

A directory with examples for creating andmanipulating NDArrays

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/tree/master/nd4j-examples/src/main/java/org/nd4j/examples)

-------------


# Reinforcement Learning Examples

Neural Networks have learned to play Space Invaders, or Doom using Reinforcement learning.  DeepLearning4J examples of Reinforcement Learning are available 

[Show me the code](http://github.com/deeplearning4j/dl4j-examples/tree/master/rl4j-examples)


