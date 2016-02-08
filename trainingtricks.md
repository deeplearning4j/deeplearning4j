---
title: Tricks to Train Deep Neural Networks
layout: default
---

# Tricks to Train Deep Neural Networks

There is an art to training neural networks, just as there is an art to training tigers to jump through a ring of fire. A little to the left or the right and disaster strikes.

Many of these tips have already been discussed in the academic literature. Our purpose is to consolidate them in one site and express them as clearly as possible. 

First, know the problem you are trying to solve. This [list of questions](http://deeplearning4j.org/questions.html) will help you clarify your task if you are new to deep learning. 

* Researchers experienced with deep neural networks tend to try three different learning rates -- 1e^-1, 1^e-3, and 1e^-6 -- to get a rough idea of what it should be. If the LR is too high, learning will be unstable and not work at all; too low and it will take much longer than necessary.

* They use Nesterov's Momentum.

* They pair activations and loss functions. That is, for classification problems, they tend to pair the softmax activation function with the negative log likelihood loss function. For regression problems, they tend to pair linear (identity) activation functions with a loss function of root mean squared error (RMSE) cross entropy.

* Regularization is important when training neural networks to avoid overfitting. [Dropout](../glossary.html#dropout) can be very effective. l2 (ell-two) regularization is commonly used as well. [Early stopping](../earlystopping.html) can be thought of as a form of regularization. 

* When constructing distributed neural networks, for example, it’s important to lower the learning rate; that is, a smaller step size as you make your gradient descent is required. Otherwise the weights diverge, and when weights diverge, your net has ceased to learn. Why is a lower learning rate required? Distributed neural networks use parameter averaging, which speeds up learning, so you need to correct for that acceleration by slowing the algorithm down elsewhere.

* The ideal minibatch size will vary. 10 is too small for GPUs, but can work on CPUs. A minibatch size of 1 works, but will not reap the benefits of parallelism. 32 may be a sensible number. 

* Pretraining was more popular in the past, but not so much these days due to better weight initializations and optimization methods.
 
### For Recurrent Neural Networks

* If you have long time series (> 200 time steps), you should use truncated [backpropagation through time](../glossary.html#backprop).

### For Restricted Boltzmann Machines (RBMs)

* When creating hidden layers for autoencoders that perform compression, give them fewer neurons than your input data. If the hidden-layer nodes are too close to the number of input nodes, you risk reconstructing the identity function. Too many hidden-layer neurons increase the likelihood of noise and overfitting. For an input layer of 784, you might choose an initial hidden layer of 500, and a second hidden layer of 250. No hidden layer should be less than a quarter of the input layer’s nodes. And the output layer will simply be the number of labels. 

  Larger datasets require more hidden layers. Facebook's Deep Face uses nine hidden layers on what we can only presume to be an immense corpus. Many smaller datasets might only require three or four hidden layers, with their accuracy decreasing beyond that depth. As a rule: larger data sets contain more variation, which require more features/neurons for the net to obtain accurate results. Typical machine learning, of course, has one hidden layer, and those shallow nets are called Perceptrons. 

* Large datasets require that you pretrain your RBM several times. Only with multiple pretrainings will the algorithm learn to correctly weight features in the context of the dataset. That said, you can run the data in parallel or through a cluster to speed up the pretraining. 
