---
title: 
layout: default
---

# Deep-belief networks

For our purposes, a deep-belief network can be defined as a stack of restricted Boltzmann machines in which each layer communicates with both the previous and subsequent layers. The nodes of any single layer don't communicate with each other laterally. 

With the exception of the first and final layers, each hidden layer has a double role: it serves as the hidden layer to the higher nodes that come before it, and as the input (or visible) layer to the lower nodes that come after. It is a network of single-layer networks. 

Deep-belief networks are used to recognize and generate images, video sequences and motion-capture data. A continuous deep-belief network is simply an extension of a deep-belief network that accepts a continuum of decimals, rather than binary data. 

### Parameters & k

See the [parameters common to all multilayer networks](../multinetwork.html).

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain. In composing a deep-belief network, a typical value is one.

### Initiating a Deep-Belief Network

Here's how you set up a single-thread deep-belief network: 

To create it, you instantiate a more general class, the multi-layer neural net, and tell it how many RBMs to stack upon each other:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/reconstruct/DBNExample.java?slice=28:38"></script>

That creates a deep-belief network with the specified hidden-layer sizes (three hidden layers); the number of inputs being 784; outputs 10; momentum and learning rate; the specified random number generator; sets root-means-squared cross entropy as the loss function; and implements no regularization. The hidden layer sizes are 600, 500 and 400 nodes, as you move forward through the net. 

Next, you iterate through the dataset with the MNISTDataFetcher, and then evaluate the performance of your classifier.

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/blob/master/src/main/java/org/deeplearning4j/reconstruct/DBNExample.java?slice=41:59"></script>

This will print out the f1 score of the prediction.

Note that the *eval* class combines [confusion matrices](../glossary.html#confusionmatrix) and f1 scores to allow for easy display and evaluation of data by allowing input of outcome matrices. This is useful for tracking how well your network trains over time. 

The f1 score will be a percentage. It's basically the probability that your guesses are correct. Eighty-six percent is industry standard; a deep-learning network that has trained well on a large number of examples should be capable of scores in the high 90s. If you run into trouble, try modifying the hidden-layer sizes, and tweaking other parameters such as learning rate and momentum to get the f1 score up.

Next, we'll show you how to use [distributed and multithreaded computing](../scaleout.html) to train your networks more quickly. To read about another type of deep net, the deep autoencoder, [click here](../deepautoencoder.html). 
