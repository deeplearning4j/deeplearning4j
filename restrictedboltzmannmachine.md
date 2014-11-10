---
title: 
layout: default
---

# Restricted Boltzmann machines

To quote Geoff Hinton, a Google researcher and university professor, a Boltzmann machine is "a network of symmetrically connected, neuron-like units that make [stochastic](../glossary.html#stochasticgradientdescent) decisions about whether to be on or off." (Stochastic means "randomly determined.")

A restricted Boltzmann machine "consists of a layer of visible units and a layer of hidden units with no visible-visible or hidden-hidden connections." The "restricted" comes from limits imposed on how its nodes connect: intra-layer connections are not allowed, but each node of one layer connects to every node of the next, and that is called "symmetry." 

So, RBMs' nodes must form a symmetrical bipartite graph where data passes through the visible layer (v0-v3) at the bottom to the hidden layer (h0-h2) at the top, like so: 

![Alt text](../img/bipartite_graph.png)

A trained restricted Boltzmann machine will learn the structure of the data fed into it via the visible layer; it does so through the act of reconstructing the data again and again, with its reconstructions increasing their similarity to the benchmark, original data. The ever-decreasing difference between the RBM's reconstruction and the benchmark is measured with a loss function. The restricted Boltzmann machine takes each step closer to the original using algorithms like stochastic gradient descent. 

[RBMs](../glossary.html#restrictedboltzmannmachine) are useful for [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), [classification](https://en.wikipedia.org/wiki/Statistical_classification), [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), [feature learning](https://en.wikipedia.org/wiki/Feature_learning) and [topic modeling](https://en.wikipedia.org/wiki/Topic_model). Given their relative simplicity, restricted Boltzmann machines are the first neural network we'll tackle.

### Parameters & k

See [the parameters common to all single-layer networks](../singlelayernetwork.html).

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain composing the restricted Boltzmann machine. A typical value is 1.

### Initiating an RBM

 <script src="http://gist-it.appspot.com/https://github.com/SkymindIO/deeplearning4j/blob/4530b123f40645a2c34e650cbfcd6b5139638c9a/deeplearning4j-core/src/test/java/org/deeplearning4j/models/featuredetectors/rbm/RBMTests.java?slice=58:74"></script>

https://github.com/SkymindIO/deeplearning4j/blob/4530b123f40645a2c34e650cbfcd6b5139638c9a/deeplearning4j-core/src/test/java/org/deeplearning4j/models/featuredetectors/rbm/RBMTests.java

### Continuous RBMs

A continuous restricted Boltzmann machine is a form of RBM that accepts continuous input (i.e. numbers cut finer than integers) via a different type of contrastive divergence sampling. This allows the CRBM to handle things like image pixels or word-count vectors that are normalized to decimals between zero and one.

It should be noted that every layer of a deep-learning net consists of four elements: the input, the coefficients, the bias and the transformation. 

The input is the numeric data, a vector, fed to it from the previous layer (or as the initial data). The coefficients are the weights given to various features passing through each node layer. The bias ensures that some nodes in a layer will be activated no matter what. The transformation is an additional algorithm that processes the data after it passes through each layer. 

Those additional algorithms and their combinations can vary layer by layer. The most effective continuous restricted Boltzmann machine we've found employs a Gaussian transformation on the visible (or input) layer and a rectified-linear-unit tranformation on the hidden layer. We've found this particularly useful in [facial reconstruction](../facial-reconstruction-tutorial.html). For RBMs handling binary data, simply make both transformations binary ones. 

*A brief aside: Geoff Hinton has noted and we can confirm that Gaussian transformations do not work well on RBMs' hidden layers, which are where the reconstructions happen; i.e. those are the layers that matter. The rectified-linear-unit transformations used instead are capable of representing more features than binary transformations, which we employ on [deep-belief nets](../deepbeliefnetwork.html).*

### Conclusions & Next Steps

You can intrepret RBMs' output numbers as percentages. Every time the number in the reconstruct is *not zero*, that's a good indication the RBM learned the input. We'll have a better example later in the tutorials. 

To explore the mechanisms that make restricted Boltzmann machines tick, click [here](../understandingRBMs.html).

Next, we'll show you how to implement a [deep-belief network](../deepbeliefnetwork.html), which is simply many restricted Boltzmann machines strung together.
