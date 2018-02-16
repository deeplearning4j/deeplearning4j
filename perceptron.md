---
title: A Beginner's Guide to Perceptrons
layout: default
---

# A Beginner's Guide to Perceptrons

The perceptron, that neural network whose name recalls how the 1950s imagined the future, is a simple algorithm intended to perform binary classification; i.e. the input belongs to a category of interest or it does not: `fraud` or `not_fraud`, `cat` or `not_cat`. 

The perceptron holds a special place in the history of neural networks and artificial intelligence, because the initial hype about its performance led to a [rebuttal by Minsky and Papert](https://drive.google.com/file/d/1UsoYSWypNjRth-Xs81FsoyqWDSdnhjIB/view?usp=sharing), and wider spread backlash that cast a pall on neural network research for decades, a neural net winter that only broke with Geoff Hinton's research in the 2000s. 

Frank Rosenblatt, the godfather of the perceptron, popularized it as a device rather than an algorithm. The perceptron is an algorithm that first entered the world as hardware. Rosenblatt, a psychologist who studied and later lectured at Cornell University, received funding from the U.S. Office of Naval Research to build a machine that could learn. 

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH PERCEPTRONS</a>
</p>

A perceptron is a linear classifier; that is, it is an algorithm that classifies input by separating two categories with a straight line. Input is typically a feature vector `x` multiplied by weights `w` and added to a bias `b`: `y = w * x + b`. 

Rosenblatt built a single-layer perceptron. That is, his hardware-algorithm did not include multiple layers, which allow neural networks to model a feature hierarchy. It was, therefore, a shallow neural network, which prevented his perceptron from performing non-linear classification, such as the XOR function (an XOR operator trigger when input exhibits either one trait or another, but not both; it stands for "exclusive OR"), as Minsky and Papert showed in their book. 



### Further Reading

* [Perceptron (Wikipedia)](https://en.wikipedia.org/wiki/Perceptron)
* [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain, Cornell Aeronautical Laboratory, Psychological Review, by Frank Rosenblatt (PDF)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf)
* [Perceptrons: An Introduction to Computational Geometry, by Marvin Minsky & Seymour Papert](https://drive.google.com/file/d/1UsoYSWypNjRth-Xs81FsoyqWDSdnhjIB/view?usp=sharing)

### <a name="beginner">Other Machine Learning Tutorials</a>
* [Introduction to Neural Networks](./neuralnet-overview)
* [Deep Reinforcement Learning](./deepreinforcementlearning)
* [Symbolic AI and Deep Learning](./symbolicreasoning)
* [Using Graph Data with Deep Learning](./graphdata)
* [Recurrent Networks and LSTMs](./lstm)
* [Word2Vec: Neural Embeddings for NLP](./word2vec)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](./eigenvector)
* [Neural Networks & Regression](./logistic-regression)
* [Convolutional Networks (CNNs)](./convolutionalnets)
* [Open Datasets for Deep Learning](./opendata)
* [Inference: Machine Learning Model Server](./modelserver)
