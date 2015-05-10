---
title: 
layout: default
---

# Restricted Boltzmann Machines

A restricted Boltzmann machine consists of neuron-like units, or nodes, arranged in two layers, the visible layer and the hidden. These nodes can be imagined as places on a graph that communicate with each other. Arranged in two rows, each node of the visible layer communicates with each node of the hidden layer; i.e. they are symmetrically connected. But the nodes of the visible layer are not connected to each other, and the same goes for the nodes of the hidden layer. 

There is no intra layer communication -- this is the restriction in a restricted Boltzmann machine. Each node is a locus of computation that process input, and makes [stochastic](../glossary.html#stochasticgradientdescent) decisions about whether to transmit that input or not. (Stochastic means "randomly determined.")

So, RBMs' nodes must form a symmetrical bipartite graph where data passes through the visible layer (v0-v3) at the bottom to the hidden layer (h0-h2) at the top, like so: 

![Alt text](../img/bipartite_graph.png)

A trained restricted Boltzmann machine will learn the structure of the data fed into it via the visible layer; it does so through the act of reconstructing the data again and again, with its reconstructions increasing their similarity to the benchmark, original data. The ever-decreasing difference between the RBM's reconstruction and the benchmark is measured with a loss function. The restricted Boltzmann machine takes each step closer to the original using algorithms like stochastic gradient descent. 

[RBMs](../glossary.html#restrictedboltzmannmachine) are useful for [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), [classification](https://en.wikipedia.org/wiki/Statistical_classification), [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), [feature learning](https://en.wikipedia.org/wiki/Feature_learning) and [topic modeling](https://en.wikipedia.org/wiki/Topic_model). Given their relative simplicity, restricted Boltzmann machines are the first neural network we'll tackle.

## Parameters & k

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain composing the restricted Boltzmann machine. A typical value is 1.

### Initiating an RBM on Iris

 <script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/iris/IrisExample.java?slice=36:53"></script>

In the above example, you can see how RBMs can be created as layers with a more general MultiLayerConfiguration. After each dot you'll find an additional parameter that affects the structure and performance of a deep neural net. Most of those parameters are defined on this site. 

**weightInit**, or weightInitialization, represents the starting value of the coefficients that amplify or mute the input signal coming into each node. Proper weight initialization can save you a lot of training time, because training a net is nothing more than adjusting the coefficients to transmit the best signals, which allow the net to classify accurately.

**activationFunction** refers to one of a set of functions that determine the threshhold(s) at each node above which a signal is passed through the node, and below which it is blocked. If a node passes the signal through, it is "activated."

**optimizationAlgo** refers to the manner by which a neural net minimizes error, or finds a locus of least error, as it adjusts its coefficients step by step. LBFGS, an acronym whose letters each refer to the last names of its multiple inventors, is an optimization algorithm that makes us of second-order derivatives to calculate the slope of gradient along which coefficients are adjust.

**regularization** such as **l2** helps fight overfitting in neural nets. Regularization essentially punishes large coefficients, since large coefficients by definition mean the net has learned to pin its results to a few heavily weighted inputs. Such a strong weight can make it difficult to generalize the net's model over new data. 

**visibleUnit/hiddenUnit** refers to the layers of a neural net. The visible unit, or layer, is the layer of nodes where input goes in, and the hiddenUnit is the layer where those inputs are recombined in more complex features. Both units have their own so-called transforms, in this case Gaussian for the visible and Rectified Linear for the hidden, which map the signal coming out of their respective layers onto a new space. 

**lossFunction** is the way you measure error, or the difference between your net's guesses and the correct labels contained in the test set. Here we use RMSE_XENT, or Root-Mean-Squared-Error-Cross-Entropy.

**learningRate**, like **momentum**, affects how much the neural net adjusts the coefficients on each iteration as it corrects for error. These two parameters help determine the size of the steps the net takes down the gradient towards a local optimum. A large learning rate will make the net learn fast, and maybe overshoot the optimum. A small learning rate will slow down the learning, which can be inefficient. 

### Continuous RBMs

A continuous restricted Boltzmann machine is a form of RBM that accepts continuous input (i.e. numbers cut finer than integers) via a different type of contrastive divergence sampling. This allows the CRBM to handle things like image pixels or word-count vectors that are normalized to decimals between zero and one.

It should be noted that every layer of a deep-learning net consists of four elements: the input, the coefficients, the bias and the transformation. 

The input is the numeric data, a vector, fed to it from the previous layer (or as the initial data). The coefficients are the weights given to various features passing through each node layer. The bias ensures that some nodes in a layer will be activated no matter what. The transformation is an additional algorithm that processes the data after it passes through each layer. 

Those additional algorithms and their combinations can vary layer by layer. The most effective continuous restricted Boltzmann machine we've found employs a Gaussian transformation on the visible (or input) layer and a rectified-linear-unit tranformation on the hidden layer. We've found this particularly useful in [facial reconstruction](../facial-reconstruction-tutorial.html). For RBMs handling binary data, simply make both transformations binary ones. 

*A brief aside: Geoff Hinton has noted and we can confirm that Gaussian transformations do not work well on RBMs' hidden layers, which are where the reconstructions happen; i.e. those are the layers that matter. The rectified-linear-unit transformations used instead are capable of representing more features than binary transformations, which we employ on [deep-belief nets](../deepbeliefnetwork.html).*

### Conclusions & Next Steps

You can intrepret RBMs' output numbers as percentages. Every time the number in the reconstruct is *not zero*, that's a good indication the RBM learned the input. We'll have a better example later in the tutorials. 

To explore the mechanisms that make restricted Boltzmann machines tick, click [here](../understandingRBMs.html).

Next, we'll show you how to implement a [deep-belief network](../deepbeliefnetwork.html), which is simply many restricted Boltzmann machines stacked on top of one another.
