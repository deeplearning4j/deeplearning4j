---
title: Deeplearning4j's Features
layout: default
---

# Features

Here's a non-exhaustive list of Deeplearning4j's features. We'll be updating it as new nets and tools are added. 

### Integrations

* Spark
* Hadoop/YARN
* Akka + AWS

### APIs

* Scala
* Java 
* (A Python SDK is forthcoming)

### Libraries

* [ND4J: N-dimensional arrays for the JVM](http://nd4j.org)

### Nets

* [Restricted Boltzmann machines](../restrictedboltzmannmachine.html)
* [Convolutional nets](../convolutionalnets.html)
* [Recursive neural tensor networks](http://nlp.stanford.edu/sentiment/)
* [Recursive autoencoders](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/models/featuredetectors/autoencoder/recursive/RecursiveAutoEncoderTest.java)
* [Recurrent nets: Long Short-Term Memory (LSTM)](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/models/classifiers/lstm/LSTMTest.java)
* [Deep-belief networks](../deepbeliefnetwork.html)
* [Denoising and Stacked Denoising autoencoders](../denoisingautoencoder.html)
* [Deep autoencoders](../deepautoencoder.html)

Since Deeplearning4j is a composable framework, users can arrange shallow nets to create various types of deeper nets. Combining convolutional nets with recurrent nets, for example, is how Google accurately generated captions from images in late 2014.

### Tools

DL4J contains the following built-in vectorization algorithms:

* [Canova: The Rosetta Stone of Vectorization](https://github.com/deeplearning4j/Canova)
* Moving-window for images
* Moving-window for text 
* Viterbi for sequential classification
* [Word2Vec](../word2vec.html)
* [Bag-of-Words encoding for word count and TF-IDF](../bagofwords-tf-idf.html)
* Constituency parsing

DL4J supports two kinds of back propagation (optimization algorithms):

* Normal stochastic gradient descent
* Conjugate gradient line search (c.f. [Hinton 2006](http://www.cs.toronto.edu/~hinton/science.pdf))

### Hyperparameters

* Dropout (random ommission of feature detectors to prevent overfitting)
* Sparsity (force activations of sparse/rare inputs)
* Adagrad (feature-specific learning-rate optimization)
* L2 regularization (weight decay)
* Weight transforms (useful for deep autoencoders)
* Probability distribution manipulation for initial weight generation

### Loss/objective functions

* Reconstruction entropy
* Squared loss
* MC class cross entropy for classification
* Negative log likelihood
* Momentum

### Activation functions 

* Tanh
* Sigmoid
* HardTanh
* Softmax
* Linear
