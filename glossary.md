---
title: 
layout: default
---



glossary
=============================

### <a name="nonlineartransformfunction">Nonlinear Transform Function</a>  
A function that maps input on a nonlinear scale such as [sigmoid](http://en.wikipedia.org/wiki/Sigmoid_function) or [tanh](http://en.wikipedia.org/wiki/Hyperbolic_function).

### <a name="activation">Activation</a> 
An activation for a neural network is defined as the mapping of the input to the output via a [non-linear transform function](#nonlineartransformfunction).

### <a name="feedforwardneuralnetwork">Feed-Forward Network</a>
A neural network that takes the initial input and triggers the [activation](#activation) of each layer of the network.

###<a name="objectivefunction">Objective Function</a> 
Also called a Loss Function. An objective function is a heuristic function for reducing prediction error in a machine-learning algorithm.

###<a name="reconstructionentropy">Reconstruction Entropy</a> 
After applying Gaussian noise, a kind of statistical white noise, to the data, this objective function punishes the network for any result that is not closer to the original input. That signal prompts the network to learn different features in an attempt to reconstruct the input better and minimize error. 