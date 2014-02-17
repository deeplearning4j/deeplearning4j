---
title: 
layout: default
---

# creating deep-learning networks

By now, you should have read about Single-Layer Neural Networks (if you haven't read about them, you can [here](({{ site.baseurl }}/singlelayernetwork.html)).

A multilayer network is a stacked representation of a [single-layer neural network]({{ site.baseurl }}/singlelayernetwork.html).

The input layer is tacked onto the first-layer neural network and a [feed-forward network]({{ site.baseurl }}/glossary.html#feedforward). Each subsequent layer from the input layer on uses the output of the previous layer as its input.

A multilayer network will accept the same kinds of inputs as a single-layer network. The multilayer network parameters are also typically the same as their single-layer network counterparts.

The output layer for a multilayer network is typically a [logistic regression classifier](http://en.wikipedia.org/wiki/Multinomial_logistic_regression), which sorts results in zeros and ones.

This is a discriminatory layer used for classification of input features based on the final hidden layer of the deep network.

A multi layer network is composed of the following kinds of layers:

* *k* single layer networks 

* a softmax regression output layer.

Training a multi layer network:

## parameters

### learning rate 

The typical value is between 0.001 and 0.1. The learning rate, or step rate, is the rate at which a function steps within a search space. Smaller learning rates mean higher training times, but may lead to more precise results.

### momentum 

Momentum is an additional factor in determining how fast an optimization algorithm converges.

### L2 Regularization constant 

This is the lambda discussed in the equation [here](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm).

Pretraining Step

For pretraining -- i.e. learning the features via reconstruction at each layer -- each network is trained and then the output is piped into the next layer.

Finetuning step

Finally, the [logistic regression](http://en.wikipedia.org/wiki/Multinomial_logistic_regression) output layer is trained, and then back propagation happens for each layer.

If you're curious about the internals of the dl4j implementation, please see [BaseMultiLayerNetwork]({{ site.baseurl }}/doc/com/ccc/deeplearning/nn/BaseMultiLayerNetwork)

Below are the different kinds of Multi Layer Networks:

* [Stacked Denoising AutoEncoders]({{ site.baseurl }}/stackeddenoisingautoencoder.html)
* [Deep Belief Networks]({{ site.baseurl }}/deepbeliefnetwork.html)
* [Continuous Deep Belief Networks]({{ site.baseurl }}/continuousdeepbeliefnetwork.html)