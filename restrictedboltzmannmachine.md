---
title: 
layout: default
---

*previous* - [a neural nets overview](../index.html)
# restricted Boltzmann machine (rbm)

To quote Geoff Hinton, a Boltzmann machine is "a network of symmetrically connected, neuron-like units that make [stochastic](../glossary.html#downpoursgd) decisions about whether to be on or off." 

A restricted Boltzmann machine "consists of a layer of visible units and a layer of hidden units with no visible-visible or hidden-hidden connections." That is, its nodes must form a symmetrical bipartite graph where data passes through the visible layer (v0-v3) at the bottom to the hidden layer (h0-h2) at the top: 

![Alt text](../img/bipartite_graph.png)

A trained restricted Boltzmann machine will learn the structure of the data fed into it via the visible layer; it does so through the very act of reconstructing the data again and again, with its reconstructions increasing their similarity to the benchmark, original image. The RBM's ever decreasing difference between reconstruction and benchmark is measured with a loss function. The restricted Boltzmann machine takes each step closer to the original using algorithms like stochastic gradient descent. 

[RBMs](../glossary.html#restrictedboltzmannmachine) are useful for [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), [classification](https://en.wikipedia.org/wiki/Statistical_classification), [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), [feature learning](https://en.wikipedia.org/wiki/Feature_learning) and [topic modeling](https://en.wikipedia.org/wiki/Topic_model). Given their relative simplicity, restricted Boltzmann machines are the first neural network we'll tackle.

### parameters & k

See [the parameters common to all single-layer networks](../singlelayernetwork.html).

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain composing the restricted Boltzmann machine. A typical value is 1.

### initiating an RBM

Here's how you set up a single-thread restricted Boltzmann machine: 

To create it, you simply instantiate an object of the [RBM class](../doc/org/deeplearning4j/rbm/RBM.html).

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/RBMMnistExample.java?slice=21:24"></script>

The RBM uses the builder pattern to set up config; for example, this builder will handle the following parameters:

           Number of visible (input) units: 784

           Number of hidden (output) units: 600 

           useRegularization(use L2?)           

Next, create a training set for the machine. 

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/RBMMnistExample.java?slice=26:35"></script>

Now that you've instantiated the machine and created the training set, it's time to train the network. This will run contrastive divergence until convergence, with a learning rate of 0.01, a k of 1 and with the input specified above. 

The last snippet will construct a new training set and show the reconstructed input. Note that the example below only takes binary input, not a continuum of integers.

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/RBMMnistExample.java?slice=47:56"></script>

After your net has trained, you'll see an F1 score. In machine learning, that's the name for one metric used to determine how well a classifier performs. The [f1 score](https://en.wikipedia.org/wiki/F1_score) is a number between zero and one that explains how well the network performed during training. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy. It's basically the probability that your net's guesses are correct.

### continuous RBMs

A continuous restricted Boltzmann machine is a form of RBM that accepts continuous input (i.e. numbers cut finer than integers) via a different type of contrastive divergence sampling. This allows the CRBM to handle things like image pixels or word-count vectors that are normalized to decimals between zero and one.

It should be noted that every layer of a deep-learning net consists of four elements: the input, the coefficients, the bias and the transformation. 

The input is the numeric data, a vector, fed to it from the previous layer (or as the initial data). The coefficients are the weights given to various features passing through each node layer. The bias ensures that some nodes in a layer will be activated no matter what. The transformation is an additional algorithm that processes the data after it passes through each layer. 

Those additional algorithms and their combinations can vary layer by layer. The most effective continuous restricted Boltzmann machine we've found employs a Gaussian transformation on the visible (or input) layer and a rectified-linear-unit tranformation on the hidden layer. We've found this particularly useful in [facial reconstruction](../facial-reconstruction-tutorial.html). For RBMs handling binary data, simply make both transformations binary ones. 

*A brief aside: Geoff Hinton has noted and we can confirm that Gaussian transformations do not work well on RBMs' hidden layers, which are where the reconstructions happen; i.e. those are the layers that matter. The rectified-linear-unit transformations used instead are capable of representing more features than binary transformations, which we employ on [deep-belief nets](../deepbeliefnetwork.html).*

### conclusions

Once built, you can test your trained restricted Boltzmann machine by feeding it unstructured data and checking the output.

You can intrepret the output numbers as percentages. Every time the number in the reconstruct is *not zero*, that's a good indication the RBM learned the input. We'll have a better example later in the tutorials. 

To explore the mechanisms that make restricted Boltzmann machines tick, click [here](../understandingRBMs.html).

Next, we'll show you how to implement a [deep-belief network](../deepbeliefnetwork.html), which is simply many restricted Boltzmann machines strung together.
