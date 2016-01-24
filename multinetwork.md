---
title: 
layout: default
---

# Creating deep-learning networks

A multilayer network is a stacked representation of a [single-layer neural network](../singlelayernetwork.html). The input layer is tacked onto the first-layer neural network and a [feed-forward network](../glossary.html#feedforward). Each subsequent layer after the input layer uses the output of the previous layer as its input.

A multilayer network will accept the same kinds of inputs as a single-layer network. The multilayer network parameters are also typically the same as their single-layer network counterparts.

The output layer for a multilayer network is typically a [logistic regression classifier](http://en.wikipedia.org/wiki/Multinomial_logistic_regression), which sorts results into zeros and ones. This is a discriminatory layer used for classification of input features based on the final hidden layer of the deep network. 

A multilayer network is composed of the following kinds of layers:

* *K* single layer networks 

* A softmax regression output layer.

### Parameters

Below are the parameters what you need to think about when training a network.

### Learning rate 

The learning rate, or step rate, is the rate at which a function steps through the search space. The typical value of the learning rate is between 0.001 and 0.1. Smaller steps mean longer training times, but can lead to more precise results. 

### Momentum 

Momentum is an additional factor in determining how fast an optimization algorithm converges on the optimum point. 

If you want to speed up the training, increase the momentum. But you should know that higher speeds can lower a model's accuracy. 

To dig deeper, momentum is a variable between zero and one that is applied as a factor to the derivative of the rate of change of the matrix. It affects the change rate of the weights over time. 

### L2 regularization constant 

L2 is the lambda discussed in [this equation](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm).

*Pretraining step*

For pretraining -- i.e. learning the features via reconstruction at each layer -- a layer is trained and then the output is piped to the next layer.

*Finetuning step*

Finally, the [logistic regression](http://en.wikipedia.org/wiki/Multinomial_logistic_regression) output layer is trained, and then back propagation happens for each layer.

Below are the different kinds of multilayer networks:

* [Stacked Denoising AutoEncoders](../stackeddenoisingautoencoder.html)
* [Deep Belief Networks](../deepbeliefnetwork.html)
* [Continuous Deep Belief Networks](../continuousdeepbeliefnetwork.html)
