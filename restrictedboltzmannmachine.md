---
title: 
layout: default
---


### Restricted Boltzmann machine

To quote Hinton, a [Boltzmann machine](http://www.scholarpedia.org/article/Boltzmann_machine) is "a network of symmetrically connected, neuron-like units that make stochastic decisions about whether to be on or off." A [restricted Boltzmann machine](http://www.scholarpedia.org/article/Boltzmann_machine#Restricted_Boltzmann_machines) "consists of a layer of visible units and a layer of hidden units with no visible-visible or hidden-hidden connections." That is, its nodes must form a [bipartite graph](https://en.wikipedia.org/wiki/Bipartite_graph). [RBMs](.{{ site.baseurl }}/glossary.html#restrictedboltzmannmachine) are useful for [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), [classification](https://en.wikipedia.org/wiki/Statistical_classification), [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), [feature learning](https://en.wikipedia.org/wiki/Feature_learning) and [topic modeling](https://en.wikipedia.org/wiki/Topic_model). Given their relative simplicity, RBMs are the first neural network we'll tackle.




PARAMETERS - Please also see [the single layer network parameters common to all single networks]({{ site.baseurl }}/singlelayernetwork.html)

k - The number of times to run [contrastive divergence]({{ site.baseurl }}/glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the markov chain

composing the restricted boltzmann machine. A typical value of 1 is fine.



BIAS - HIDDEN AND VISIBLE

SHOW CONNECTION BETWEEN MATRIX AND DRAWING OF NODES, MAPPING NUMBERS TO CONNECTIONS

EXPLAIN WHAT THE WEIGHTS MEAN

### Initiating a restricted Boltzmann machine 

Setting up a single-thread restricted Boltzmann machine is easy. 

To create the machine, you simply instantiate an object of the [class](../doc/com/ccc/deeplearning/rbm/RBM.html).

CODE BLOCK MACHINE CREATION TK

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

CODE BLOCK TRAINING SET TK

Now that you have instantiated the machine and created the training set, it's time to train the network. 

CODE BLOCK TRAINING THE MACHINE TK

You can test your trained network by feeding it unstructured data and checking the output. [MORE EXPLANATION HERE?]

Here are the code blocks for a multithread restricted Boltzmann machine:

Create the machine:

CODE BLOCK MACHINE CREATION TK

Create the training set:

CODE BLOCK TRAINING SET TK

Train the machine:

CODE BLOCK TRAINING THE MACHINE TK

