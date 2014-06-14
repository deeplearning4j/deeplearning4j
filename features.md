---
title: 
layout: default
---

# features

Here's a non-exhaustive list of Deeplearning4j's features. We'll be updating it as new nets and tools are added. 

### nets

* [Restricted Boltzmann machines](../restrictedboltzmannmachine.html)
* [Deep-belief networks](../deepbeliefnetwork.html)
* [Denoising and Stacked Denoising autoencoders](../denoisingautoencoder.html)
* [Deep autoencoders](../deepautoencoder.html)

*Coming soon*

* [Convolutional deep-belief networks](../convolutionalnets.html)
* [Recursive neural tensor networks](http://nlp.stanford.edu/sentiment/)

### tools

DL4J contains the following built-in vectorization algorithms:

* Moving-window for images
* Moving-window for text 
* Viterbi for sequential classification
* [Word2Vec](../word2vec.html)
* [Bag-of-Words encoding for word count and TF-IDF](../bagofwords-tf-idf.html)

*Coming soon*

* Dependency parsing

DL4J supports two kinds of back propagation (optimization algorithms):

* Normal stochastic gradient descent
* Conjugate gradient line search (c.f. [Hinton 2006](http://www.cs.toronto.edu/~hinton/science.pdf))

### knobs

* Dropout (random ommission of feature detectors to prevent overfitting)
* Sparsity (force activations of sparse/rare inputs)
* Adagrad (feature-specific learning-rate optimization)
* L2 regularization (weight decay)
* Weight transforms (useful for deep autoencoders)
* Probability distribution manipulation for initial weight generation

### loss/objective functions

* Reconstruction entropy
* Squared loss
* MC class cross entropy for classification
* Negative log likelihood
* Momentum

### activation functions 

* Tanh
* Sigmoid
* HardTanh
* Softmax
* Linear
