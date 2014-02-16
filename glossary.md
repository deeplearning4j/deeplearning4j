---
title: deeplearning4j
layout: default
---



Glossary
=============================


### <a name="nonlineartransformfunction"></a> Non-Linear Transform Function: A function that maps input on a non linear scale such as [sigmoid](http://en.wikipedia.org/wiki/Sigmoid_function) or [tanh](http://en.wikipedia.org/wiki/Hyperbolic_function)

### <a name="activations"></a>Activations: An activation for a neural network is defined as the mapping of the input to the output via a [non-linear transform function](#nonlineartransformfunction).

### <a name="feedforwardneuralnetwork"></a>Feed Forward Network: A neural network that takes the initial input and triggers the [activations](#activations) of each layer of the network.

###<a name="objectivefunction"></a>Objective Function: Also called a Loss Function. An objective function is a heuristic function for reducing prediction error in a machine learning algorithm.

###<a name="reconstructionentropy"></a> Reconstruction Entropy: After applying gaussian noise (think white noise) to the data. This objective function punishes anything not closer to the original input. This allows a network to learn how to reconstruct the input and thus learning different features. 

###<a name="etl"></a> ETL: Extract, transform, load: Data is loaded from disk in to memory with the proper transforms such as [binarization](#binarization) and [normalization](#normalization)

###<a name="binarization"></a> Binarization: The process of transforming data in to a set of zeros and 1s. An example would be grayscaling an image by tranforming a picture from the 0-255 spectrum to a zero or 1 spectrum. 

###<a name="normalization"></a> Normalization: The process of transforming the data in to a range of 0 to 1. 