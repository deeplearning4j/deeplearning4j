---
title: 
layout: default
---

# Single-layer neural network

A single-layer neural network in deep learning is a net composed of an input layer, which is  visible layer, and a hidden output layer. 

The single-layer network's goal, or [objective function](../glossary.html#objectivefunction), is to learn features by minimizing [reconstruction entropy](../glossary.html#reconstructionentropy).

This allows it to autolearn features of the input, which leads to finding good correlations and higher accuracy in identifying discriminatory features. From there, a multilayer network leverages this to accurately classify the data. This is the pretraining step.

Each single-layer network has the following attributes:

* Hidden bias: The bias for the output
* Visible Bias: The bias for the input
* Weight Matrix: The weights for the machine 

### Training a single-layer network

Train a network by joining the input vector to the input layer. Distort the input with some Gaussian noise. This noise function will vary depending on the network. Then minimize reconstruction entropy through pretraining until the network learns the best features for reconstructing the input data.

### Learning rate

A typical learning-rate value is between 0.001 and 0.1. The learning rate, or step rate, is the rate at which a function steps within a search space. Smaller learning rates mean higher training times, but may lead to more precise results.

### Momentum

Momentum is an extra factor in determining how fast an optimization algorithm converges.

### L2 regularization constant

L2 is the lambda discussed in the equation [here](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm).

Here are the different kinds of single-layer networks:

* [Restricted Boltzmann Machines](../restrictedboltzmannmachine.html)
* [Continuous Restricted Boltzmann Machine](../continuousrestrictedboltzmannmachine.html)
* [Denoising AutoEncoder](../denoisingautoencoder.html)
