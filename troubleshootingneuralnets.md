---
title: Troubleshooting Neural Net Training
layout: default
---

# Troubleshooting Neural Net Training

Neural networks can be difficult to tune. If the network hyperparameters are poorly chosen, the network may learn slowly, or perhaps not at all. This page aims to provide some baseline steps you should take when tuning your network.

Many of these tips have already been discussed in the academic literature. Our purpose is to consolidate them in one site and express them as clearly as possible.

## Data Normalization

What's distribution of your data? Are you scaling it properly? As a general rule:

- For continuous values: you want these to be in the range of -1 to 1, 0 to 1 or ditributed normally with mean 0 and standard deviation 1. This does not have to be exact, but ensuring your inputs are approximately in this range can help during training. Scale down large inputs, and scale up small inputs.
- For discrete classes (and, for classification problems for the output), generally use a one-hot representation. That is, if you have 3 classes, then your data will be represeted as [1,0,0], [0,1,0] or [0,0,1] for each of the 3 classes respectively.

Note that it's very important to use the exact same normalization method for both the training data and testing data.

## Weight Initialization

Deeplearning4j supports several different kinds of weight initializations with the weightInit parameter. These are set using the .weightInit(WeightInit) method in your configuration.

You need to make sure your weights are neither too big nor too small. Xavier weight initialization is usually a good choice for this. For networks with rectified linear (relu) or leaky relu activations, RELU weight initialization is a sensible choice.

## Number of Epochs and Number of Iterations

An epoch is defined as a full pass of the data set. An iteration in DL4J is defined as the number of parameter updates in a row, for each minibatch.

Generally, you want to use multiple epochs and one iteration (.iterations(1) option) when training; multiple iterations are generally only used when doing full-batch training on very small data sets.

Too few epochs don't give your network enough time to learn good parameters; too many and you might overfit the training data. One way to choose the number of epochs is to use early stopping. [Early stopping](http://deeplearning4j.org/earlystopping) can also help to prevent the neural network from overfitting (i.e., can help the net generalize better to unseen data).

## Learning Rate

The learning rate is one of, if not the most important hyperparameter. If this is too large or too small, your network may learn very poorly, very slowly, or not at all. Typical values for the learning rate are in the range of 0.1 to 1e-6, though the optimal learning rate is usually data (and network architecture) specific. Some simple advice is to start by trying three different learning rates – 1e-1, 1e-3, and 1e-6 – to get a rough idea of what it should be, before further tuning this. Ideally, they run models with different learning rates simultaneously to save time.

The usual approach to selecting an appropriate learning rate is to use [DL4J's visualization interface](http://deeplearning4j.org/visualization) to visualize the progress of training. You want to pay attention to both the loss over time, and the ratio of update magnitudes to parameter magnitudes (a ratio of approximately 1:1000 is a good place to start). For more information on tuning the learning rate, see [this link](http://cs231n.github.io/neural-networks-3/#baby).

For training neural networks in a distributed manner, you may need a different (frequently higher) learning rate compared to training the same network on a single machine.

## Activation Function

There are two aspects to be aware of, with regard to the choice of activation function.

First, the activation function of the hidden (non-output) layers. As a general rule, 'relu' or 'leakyrelu' activations are good choices for this. Some other activation functions (tanh, sigmoid, etc) are more prone to vanishing gradient problems, which can make learning much harder in deep neural networks. However, for LSTM layers, the tanh activation function is still commonly used.

Second, regarding the activation function for the output layer: this is usually application specific. For classification problems, you generally want to use the softmax activation function, combined with the negative log likelihood / MCXENT (multi-class cross entropy). The softmax activation function gives you a probability distribution over classes (i.e., outputs sum to 1.0). For regression problems, the "identity" activation function is frequently a good choice, in conjunction with the MSE (mean squared error) loss function.

## Loss Function

Loss functions for each neural network layer can either be used in pretraining, to learn better weights, or in classification (on the output layer) for achieving some result. (In the example above, classification happens in the override section.)

Your net's purpose will determine the loss funtion you use. For pretraining, choose reconstruction entropy. For classification, use multiclass cross entropy.

## Regularization

Regularization methods can help to avoid overfitting during training. Overfitting occurs when the network predicts the training set very well, but makes poor predictions on data the network has never seen. One way to think about overfitting is that the network memorizes the training data (instead of learning the general relationships in it).

Common types of regularization include:

- l1 and l2 regularization penalizes large network weights, and avoids weights becoming too large. Some level of l2 regularization is commonly used in practice. However, note that if the l1 or l2 regularization coefficients are too high, they may over-penalize the network, and stop it from learning. Common values for l2 regularization are 1e-3 to 1e-6.
- [Dropout](../glossary.html#dropout), is a frequently used regularization method can be very effective
- Dropconnect (conceptually similar to dropout, but used much less frequently)
- Restricting the total number of network size (i.e., limit the number of layers and size of each layer)
- [Early stopping](http://deeplearning4j.org/earlystopping)

To use l1/l2/dropout regularization, use .regularization(true) followed by .l1(x), .l2(y), .dropout(z) respectively. Note that z in dropout(z) is the probability of retaining an activation.

## Minibatch Size

A minibatch refers to the number of examples used at a time, when computing gradients and parameter updates. In practice (for all but the smallest data sets), it is standard to break your data set up into a number of minibatches.

The ideal minibatch size will vary. For example, a minibatch size of 10 is frequently too small for GPUs, but can work on CPUs. A minibatch size of 1 will allow a network to train, but will not reap the benefits of parallelism. 32 may be a sensible starting point to try, with minibatches in the range of 16-128 (sometimes smaller or larger, depending on the application and type of network) being common.

## Updater and Optimization Algorithm

In DL4J, the term 'updater' refers to training mechanisms such as momentum, RMSProp, adagrad, and others. Using one of these methods can result in much faster network training companed to 'vanilla' stochastic gradient descent. You can set the updater using the .updater(Updater) configuration option.

The optimization algorithm is how updates are made, given the gradient. The simplest (and most commonly used) method is stochastic gradient descent (SGD), however DL4J also provides SGD with line search, conjugate gradient and LBFGS optimization algorithms. These latter algorithms are more powerful compared to SGD, but considerably more costly per parameter update due to a line search component, and aren't used as much in practice. Note that you can in principle combine any updater with any optimization algorithm.

A good default choice in most cases is to use the stochastic gradient descent optimization algorithm combined with one of the momentum/rmsprop/adagrad updaters, with momentum frequently being used in practice. Note that for momentum, the updater is called NESTEROVS (a reference to the Nesterovs variant of momentum), and the momentum rate can be set by the .momentum(double) option.

## Gradient Normalization

When training a neural network, it can sometimes be helpful to apply gradient normalization, to avoid the gradients being too large (the so-called exploding gradient problem, common in recurrent neural networks) or too small. This can be applied using the .gradientNormalization(GradientNormalization) and .gradientNormalizationThreshould(double) methods. See [this, for more details](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/GradientNormalization.java).

## Recurrent Neural Networks: Truncated Backpropagation through Time

When training recurrent networks with long time series, it is generally advisable to use truncated backpropagation through time. With 'standard' backpropagation through time (the default in DL4J) the cost per parameter update can become prohibative. For more details, see [this page](http://deeplearning4j.org/usingrnns) and [this glossary entry](../glossary.html#backprop).

## Visible/Hidden Unit

When using a deep-belief network, pay close attention here. An RBM (the component of the DBN used for feature extraction) is stochastic and will sample from different probability distributions relative to the visible or hidden units specified. 

See Geoff Hinton's definitive work, [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf), for a list of all of the different probability distributions.

## Restricted Boltzmann Machines (RBMs)

When creating hidden layers for autoencoders that perform compression, give them fewer neurons than your input data. If the hidden-layer nodes are too close to the number of input nodes, you risk reconstructing the identity function. Too many hidden-layer neurons increase the likelihood of noise and overfitting. For an input layer of 784, you might choose an initial hidden layer of 500, and a second hidden layer of 250. No hidden layer should be less than a quarter of the input layer’s nodes. And the output layer will simply be the number of labels.

Larger datasets require more hidden layers. Facebook’s Deep Face uses nine hidden layers on what we can only presume to be an immense corpus. Many smaller datasets might only require three or four hidden layers, with their accuracy decreasing beyond that depth. As a rule: larger data sets contain more variation, which require more features/neurons for the net to obtain accurate results. Typical machine learning, of course, has one hidden layer, and those shallow nets are called Perceptrons.

Large datasets require that you pretrain your RBM several times. Only with multiple pretrainings will the algorithm learn to correctly weight features in the context of the dataset. That said, you can run the data in parallel or through a cluster to speed up the pretraining.

