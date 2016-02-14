---
title: Troubleshooting Neural Net Training
layout: default
---

#Troubleshooting Neural Net Training

Neural networks are notoriously difficult to tune. When training with any neural net framework, you want to be able to understand how to use it. Below are some baseline steps you should take when tuning your network to train:

##Data

What's distribution of your data? Are you scaling it properly? With Deeplearning4j, there are a different scaling and normalization techniques to keep in mind.

Traverse the input data with the IrisDataSetInterator.

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNIrisExample.java?slice=57:62"></script>

Configure the DBN as a MultilayerNeuralNet whose layers are RBMs:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/deepbelief/DBNIrisExample.java?slice=42:98"></script>

There's a lot going on here -- I'll cover each of the facets of the configuration and how it relates to troubleshooting.

## weightInit

Deeplearning4j supports several different kinds of weight initializations with the weightInit parameter.

You need to make sure your weights are neither too big nor too small. If you're running a classifier, ensure that your weights are initialized to zero on the output layer (as with the above example).

##Iterations

Note the number of iterations you're training on. Early stopping can help prevent neural-network weights from diverging and will help a net generalize better.

##Learning Rate

The learning rate is either used as the master step size in Adagrad (default) or the step size used for calculating your gradient. With continuous data, you will want a smaller learning rate. With binary, you can often go up to a rate of 1e-1.

If, when visualizing weights and gradients, you notice the magnitudes are too high, try shrinking the learning rate or the weights.

##Activation Function

Different data has different optimal activation functions. 

Using tanh, you can get a bounded range of -1,1 -- and with a combination of zero mean and unit variance, that may be a good nonlinear transform for the data. When dealing with binary data, sigmoid is better.

##Visible/Hidden Unit

When using a deep-belief network, pay close attention here. An RBM (the component of the DBN used for feature extraction) is stochastic and will sample from different probability distributions relative to the visible or hidden units specified. 

See Geoff Hinton's definitive work, [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf), for a list of all of the different probability distributions.

##Loss Function

Loss functions for each neural network layer can either be used in pretraining, to learn better weights, or in classification (on the output layer) for achieving some result. (In the example above, classification happens in the override section.)

Your net's purpose will determine the loss funtion you use. For pretraining, choose reconstruction entropy. For classification, use multiclass cross entropy.

##Gradient Constraining

When training a neural network, it can sometimes be in your interest to constrain the gradient to unit norm. This will prevent overfitting and has a very similar effect to regularization. If your neural network isn't working, this can help. However, over penalizing can also prevent learning from happening. I would advise enabling this when the magnitude of your gradient is large (this can be seen if you add render(n) to your neural network configuration)
or the neural network at the beginning (due to high weights among other things) has near all white activiations (the grey/white/black plot you see pop up after the histograms).

You can enable this by adding:
 
      .constrainGradientToUnitNorm(true)

to your configuration.
