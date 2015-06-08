---
title: 
layout: default
---

# IRIS Classifed With a Deep-Belief Net

Deep-belief networks are multi-class classifiers. Given many inputs belonging to various classes, a DBN can first learn from a small training set, and then classify unlabeled data according to those various classes. It can take in one input and decide which label should be applied to its data record. 

Given an input record, the DBN will choose one label from a set. This goes beyond a Boolean ‘yes’ or ‘no’ to handle a broader, multinomial taxonomy of inputs, where the label chosen is represented as a 1, and all other possible labels are 0s. 

That is, the network outputs a vector containing one number per output node. The number of output nodes represented in the vector equals the number of labels to choose from. Each of those outputs are going to be a 0 or 1, and taken together, those 0s and 1s form the vector. 

*(To run the Iris example, [use this file](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/iris/IrisExample.java) and explore others from our [Quick Start page](../quickstart.html).)*

### The IRIS Dataset

The [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/Iris) is widely used in machine learning to test classification techniques. We will use it to verify the effectiveness of a deep-belief net.

The dataset consists of four measurements taken from 50 examples of each of three species of Iris, so 150 flowers and 600 data points in all. The various iris species have petals and sepals (the green, leaflike sheaths at the base of petals) of different lengths, measured in centimeters. The length and width of both sepals and petals were taken for the species *Iris setosa, Iris virginica* and *Iris versicolor*. Each species name serves as a label. 

The continuous nature of those measurements make the Iris dataset a perfect test for continuous deep-belief networks. Those four features alone can be sufficient to classify the three species accurately. That is, success here consists of teaching a neural net to classify by species the data records of individual flowers while knowing only their dimensions, and failure to do the same is a very strong signal that your neural net needs fixing. 

The dataset is small, which can present its own problems, and the species I. virginica and I. versicolor are so similar that they partially overlap -- a useful and interesting twist... 

Here is a single record:

![data record table](../img/data_record.png)

While the table above is human readable, Deeplearning4j’s algorithms need it to be something more like

     5.1,3.5,1.4,0.2,i.setosa

In fact, let’s take it one step further and get rid of the words, arranging numerical data in two objects:

Data:  5.1,3.5,1.4,0.2    
Label: 0,1,0

Given three output nodes making binary decisions, we can label the three iris species as 1,0,0, or 0,1,0, or 0,0,1. 

### Loading the data

DL4J uses DataSetIterator and DataSet objects to load and store data for neural networks. A DataSet holds data and its associated labels, which we want to make predictions about, while the DataSetIterator feeds the data in piece by piece. 

The columns First and Second, below, are both NDArrays. One of the NDArrays will hold the data’s attributes; the other holds the labels. 

![input output table](../img/ribbit_table.png)

(Contained within a DataSet object are two NDArrays, the fundamental data structures that DL4J uses for numeric computation. N-dimensional arrays are scalable, multi-dimensional arrays suitable for sophisticated mathematical operations and frequently used in [scientific computing](http://nd4j.org).) 

The IRIS dataset, like many others, comes as a CSV (comma-separated value) file. We use a [general machine-learning vectorization lib called Canova](http://deeplearning4j.org/canova.html) to parse it. 

### Creating a Neural Network (NN)

Now we’re ready to create a deep-belief network, or DBN, to classify our inputs.

With DL4J, creating a neural network of any kind involves several steps. 

First, we need to create a configuration object:

 <script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNIrisExample.java?slice=53:81"></script>

This has everything that our DBN classifier will need. As you can see, there are a lot of parameters, or ‘knobs’, that you will learn to adjust over time to improve your nets’ performance. These are the pedals, clutch and steering wheel attached to DL4J's deep-learning engine. 

These include but are not limited to: the momentum, regularizations (yes or no) and its coefficient, the number of iterations (or passes as the algorithm learns), the velocity of the learning rate, the number of output nodes, and the transforms attached to each node layer (such as Gaussian or Rectified). 

 <script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNIrisExample.java?slice=80:82"></script>

By training a model on a dataset, your algorithm learns to extract those specific features of the data that are useful signals for classifying the target input, the features that distinguish one species from another.

Training is the repeated attempt to classify inputs based on various machine-extracted features. It consists of two main steps: the comparison of those guesses with the real answers in a test set; and the reward or punishment of the net as it moves closer to or farther from the correct answers. With sufficient data and training, this net could be unleashed to classify unsupervised iris data with a fairly precise expectation of its future accuracy. 

You should see some output from running that last line, if debugs are turned on. 

### **Evaluating our results**

Consider the code snippet below, which would come after our *fit()* call.

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNIrisExample.java?slice=83:96"></script>

DL4J uses an **Evaluation** object that collects statistics about the model’s performance. The INDArray output is created by a chained call of *DataSet.getFeatureMatrix()* and **output**. The getFeatureMatrix call returns an NDArray of all data inputs, which is fed into **output()**. This method will label the probabilities of an input, in this case our feature matrix. *eval* itself just collects misses and hits of predicted and real outcomes of the model. 

The **Evaluation** object contains many useful calls, such as *f1()*. This method esimates the accuracy of a model in the form of a probability (The f1 score below means the model considers its ability to classify about 77 percent accurate). Other methods include *precision()*, which tells us how well a model can reliably predict the same results given the same input, and *recall()* which tells us how many of the correct results we retrieved.

In this example, we have the following

     Actual Class 0 was predicted with Predicted 0 with count 50 times

     Actual Class 1 was predicted with Predicted 1 with count 1 times

     Actual Class 1 was predicted with Predicted 2 with count 49 times

     Actual Class 2 was predicted with Predicted 2 with count 50 times

    ====================F1Scores========================
                     0.767064393939394
    ====================================================

After your net has trained, you'll see an F1 score like this. In machine learning, an F1 score is one metric used to determine how well a classifier performs. It's a number between zero and one that explains how well the network performed during training. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy. It's basically the probability that your net's guesses are correct. 

In the end, a working network's visual representation of the Iris dataset will look something like this

![Alt text](../img/iris_dataset.png)
