---
title: 
layout: default
---

single-layer neural network
======================================

A single-layer neural network in deep learning is a neural network composed of an input, the visible layer, and a hidden output layer.

The single layer network's goal, or [objective function](../glossary.html#objectivefunction), is to learn features by minimizing [reconstruction entropy](../glossary.html#reconstructionentropy)

This allows it to auto learn features of the input which leads to finding good correlations and higher accuracy on finding discriminatory features. From there, a multilayer network leverages this to do accurate classification of data. This is called the pretraining step in the literature.

Each single-layer network has the following attributes:

* Hidden bias: The bias for the output
* Visible Bias: The bias for the input
* Weight Matrix: The weights for the machine 

### training a single-layer network

Train a network by joining the input vector to the input. Distort the input with some Gaussian noise. Depending on the network, this noise function will vary. Then minimize reconstruction entropy to carry out pretraining until the network learns the best features for reconstructing the input data.

## parameters

### learning Rate

Typical value is between 0.001 and 0.1. The learning rate, or step rate, is the rate at which a function steps within a search space. Smaller learning rates mean higher training times, but may lead to more precise results.

### momentum

Momentum is an extra factor in determining how fast an optimization algorithm converges.

### L2 regularization constant

This is the lambda discussed in the equation [here](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm).

If you're curious about the internals of the dj4j implementation, please see [BaseNeuralNetwork](../doc/com/ccc/deeplearning/nn/BaseNeuralNetwork.html)

Below are the different kinds of single-layer networks:

*[Restricted Boltzmann Machines](../restrictedboltzmannmachine.html)
*[Continuous Restricted Boltzmann Machine](../continuousrestrictedboltzmannmachine.html)
*[Denoising AutoEncoder](../denoisingautoencoder.html)