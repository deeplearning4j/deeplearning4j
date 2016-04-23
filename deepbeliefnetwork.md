---
title: Deep-Belief Networks in Java
layout: default
---

# Tutorial: Deep-Belief Networks & MNIST

A deep-belief network can be defined as a stack of [restricted Boltzmann machines, explained here](../restrictedboltzmannmachine.html), in which each RBM layer communicates with both the previous and subsequent layers. The nodes of any single layer don't communicate with each other laterally. 

This stack of RBMs might end with a a [Softmax](../glossary.html#softmax) layer to create a classifier, or it may simply help cluster unlabeled data in an unsupervised learning scenario. 

With the exception of the first and final layers, each layer in a deep-belief network has a double role: it serves as the hidden layer to the nodes that come before it, and as the input (or "visible") layer to the nodes that come after. It is a network built of single-layer networks. 

Deep-belief networks are used to recognize, cluster and generate images, video sequences and motion-capture data. A continuous deep-belief network is simply an extension of a deep-belief network that accepts a continuum of decimals, rather than binary data. They were introduced by [Geoff Hinton and his students in 2006](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).

## MNIST for Deep-Belief Networks

MNIST is a good place to begin exploring image recognition and DBNs. The first step is to take an image from the dataset and binarize it; i.e. convert its pixels from continuous gray scale to ones and zeros. Typically, every gray-scale pixel with a value higher than 35 becomes a 1, while the rest are set to 0. The MNIST dataset iterator class does that.

### Hyperparameters

See the [parameters common to all multilayer networks in our Iris DBN tutorial](../iris-flower-dataset-tutorial).

The variable k represents the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain. In composing a deep-belief network, a typical value is `1`.

### Initiating a Deep-Belief Network

An **MnistDataSetIterator**, just a form of a more general DataSetIterator, will do that. 

Typically, a DataSetIterator handles inputs and dataset-specific concerns like binarization or normalization. For MNIST, the following line specifies the batch size and number of examples, two parameters which allow the user to specify the sample size they want to train on (more examples tend to increase both model accuracy and training time):
         
         //Train on batches of 100 out of 60000 examples
         DataSetIterator iter = new MnistDataSetIterator(100,60000);

Traverse the input data with the `MnistDataSetIterator`. (You can see the entire example [here](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DBNMnistFullExample.java), and download it with [DL4J's examples repo](https://github.com/deeplearning4j/dl4j-0.4-examples/).)

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DBNMnistFullExample.java?slice=42:45"></script>

Set up the DBN with a `MultiLayerConfiguration` whose layers are RBMs:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DBNMnistFullExample.java?slice=45:75"></script>

Train the model by calling `fit`:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DBNMnistFullExample.java?slice=75:82"></script>

Then evaluate the performance of the net:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DBNMnistFullExample.java?slice=82:91"></script>

Note that the *eval* class combines [confusion matrices](../glossary.html#confusionmatrix) and f1 scores to allow for easy display and evaluation of data. This is useful for tracking how well your network trains over time. 

F1 scores are expressed as percentages. They are basically the probability that your net's guesses are correct. To improve a neural net's performance, you can tune it by modifying the number and size of the hidden layers, and tweaking other parameters such as learning rate, momentum, weight distribution and various types of regularization.

Next, we'll show you how to use [distributed and multithreaded computing](../iterativereduce) to train your networks more quickly. To read about another type of deep network, the deep autoencoder, [click here](../deepautoencoder). 

### <a name="beginner">Other Deeplearning4j Tutorials</a>
* [Introduction to Neural Networks](../neuralnet-overview)
* [LSTMs and Recurrent Networks](../lstm)
* [Word2vec](../word2vec)
* [Restricted Boltzmann Machines](../restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](../eigenvector)
* [Neural Networks and Regression](../linear-regression)
* [Convolutional Networks](../convolutionalnets)
