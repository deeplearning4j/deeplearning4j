---
title: 
layout: default
---


Up to this point we have assumed you have read about Single Layer Neural Networks, if not please start at: [link]

# Creating Deep-Learning Networks

A multi layer network is a stacked representation of a [single layer neural network]({{ site.baseurl }}/singlelayernetwork.html).

The input layer is tacked on to the first layer neural network and a [feed forward network]({{ site.baseurl }}/glossary.html#feedforward).

Each subsequent layer from the input then uses the output of the previous layer as its input.


A multi layer network will accept all of the same kinds of inputs as a single layer network.

The multi layer network parameters are also typically the same as their single layer network counterparts.


The output layer for a multi layer network is typically a [logistic regression classifier](http://en.wikipedia.org/wiki/Multinomial_logistic_regression).


This is a discriminatory layer used for classification of input features based on the final hidden layer of the deep network.



A multi layer network is composed of the following kinds of layers

k single layer networks 

a softmax regression output layer.


Training a multi layer network:



##PARAMETERS

###Learning Rate: Typical value is between 0.001 and 0.1. The learning rate, or step rate, is the rate

at which a function steps within a search space. Smaller learning rates mean higher training times,

but may lead to more precise results.


###Momentum: An extra factor in determining how fast an optimization algorithm converges.

###L2: L2 Regularization constant: This is the lambda discussed in the equation [here](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm)


Pretrain Step:

For pre training (learning the features via reconstruction at each layer) each network is trained and then the input of that is piped in to the next layer.


Finetune step:

Finally, the [logistic regression](http://en.wikipedia.org/wiki/Multinomial_logistic_regression) output layer is trained, and then back prop happens for each layer.




If you are curious about the internals of the dj4j implementation, please see [BaseMultiLayerNetwork]({{ site.baseurl }}/doc/com/ccc/deeplearning/nn/BaseMultiLayerNetwork)


Below are the different kinds of Multi Layer Networks:

[Stacked Denoising AutoEncoders]({{ site.baseurl }}/stackeddenoisingautoencoder.html)
[Deep Belief Networks]({{ site.baseurl }}/deepbeliefnetwork.html)
[Continuous Deep Belief Networks]({{ site.baseurl }}/continuousdeepbeliefnetwork.html)

