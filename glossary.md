---
title: 
layout: default
---

# glossary

### <a name="activation">Activation</a> 
An activation for a neural network is defined as the mapping of the input to the output via a [non-linear transform function](#nonlineartransformfunction).

###<a name="binarization">Binarization</a> 
The process of transforming data in to a set of zeros and ones. An example would be gray-scaling an image by transforming a picture from the 0-255 spectrum to a 0-1 spectrum. 

### <a name="confusionmatrix">Confusion Matrix</a>
Also known as an error matrix or contingency table. Confusions matrices allow you to see if your algorithm is systematically confusing two labels, by contrasting your net's predictions against a benchmark.

###<a name="etl">ETL</a>  
Extract, transform, load: Data is loaded from disk into memory with the proper transforms such as [binarization](#binarization) and [normalization](#normalization).

### <a name="feedforwardneuralnetwork">Feed-Forward Network</a>
A neural network that takes the initial input and triggers the [activation](#activation) of each layer of the network.

### <a name="gaussian">Gaussian Distribution</a>
A Gaussian, or [normal](https://en.wikipedia.org/wiki/Normal_distribution), distribution, is a continuous probability distribution that represents the probability that any given observation will occur on different points of a range. Visually, it resembles what's usually called a Bell curve. 

### <a name="nonlineartransformfunction">Nonlinear Transform Function</a>  
A function that maps input on a nonlinear scale such as [sigmoid](http://en.wikipedia.org/wiki/Sigmoid_function) or [tanh](http://en.wikipedia.org/wiki/Hyperbolic_function).

###<a name="normalization">Normalization</a> 
The process of transforming the data to span a range from 0 to 1. 

###<a name="objectivefunction">Objective Function</a> 
Also called a Loss Function. An objective function is a heuristic function for reducing prediction error in a machine-learning algorithm.

###<a name="reconstructionentropy">Reconstruction Entropy</a> 
After applying Gaussian noise, a kind of statistical white noise, to the data, this objective function punishes the network for any result that is not closer to the original input. That signal prompts the network to learn different features in an attempt to reconstruct the input better and minimize error. 
