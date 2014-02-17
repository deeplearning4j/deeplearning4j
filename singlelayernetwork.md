---
title: 
layout: default
---

single-layer neural network
======================================

A single layer neural network with respect to deep learning is a neural network composed of an input (visible) layer and a hidden (output layer).

The single layer network's goal, or [objective function](../glossary.html#objectivefunction), is to learn features by minimizing [reconstruction entropy](../glossary.html#reconstructionentropy)

This allows it to auto learn features of the input which leads to finding good correlations and higher accuracy on finding discriminatory features. 

From there, a multi layer network leverages this to do accurate classification of data. This is called the pretraining step in the literature.


Each Single Layer Network has the following attributes:

Hidden bias: The bias for the output

Visible Bias: The bias for the input.

Weight Matrix: The weights for the machine.


###Training a single layer network:

Train a network by clamping the input vector on to the input. Distort the input with some gaussian noise.

Depending on the network, this noise function will be different. Then minimize reconstruction entropy

to do pretraining until the network learns good features for reconstructing the input data.


##PARAMETERS

###Learning Rate: Typical value is between 0.001 and 0.1. The learning rate, or step rate, is the rate

at which a function steps within a search space. Smaller learning rates mean higher training times,

but may lead to more precise results.


###Momentum: An extra factor in determining how fast an optimization algorithm converges.


###L2: L2 Regularization constant: This is the lambda discussed in the equation [here](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm)


If you are curious about the internals of the dj4j implementation, please see [BaseNeuralNetwork](../doc/com/ccc/deeplearning/nn/BaseNeuralNetwork.html)



Below are the different kinds of Single Layer Networks:



[Restricted Boltzmann Machines](../restrictedboltzmannmachine.html)
[Continuous Restricted Boltzmann Machine](../continuousrestrictedboltzmannmachine.html)
[Denoising AutoEncoder](../denoisingautoencoder.html)