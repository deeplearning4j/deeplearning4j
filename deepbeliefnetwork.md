---
title: 
layout: default
---

# Deep-belief networks

For our purposes, a deep-belief network can be defined as a stack of restricted Boltzmann machines in which each layer communicates with both the previous and subsequent layers. The nodes of any single layer don't communicate with each other laterally. 

With the exception of the first and final layers, each hidden layer has a double role: it serves as the hidden layer to the higher nodes that come before it, and as the input (or visible) layer to the lower nodes that come after. It is a network of single-layer networks. 

Deep-belief networks are used to recognize and generate images, video sequences and motion-capture data. A continuous deep-belief network is simply an extension of a deep-belief network that accepts a continuum of decimals, rather than binary data. 

### parameters & k

See the [parameters common to all multilayer networks](../multinetwork.html).

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain. In composing a deep-belief network, a typical value is one.

### initiating a deep-belief network

Here's how you set up a single-thread deep-belief network: 

To create it, you instantiate your NeuralNetConfiguration as well as an object of the class [DBN](../doc/org/deeplearning4j/dbn/DBN.html).

<script src="https://github.com/SkymindIO/dl4j-examples/edit/master/src/main/java/org/deeplearning4j/mnist/MnistExample.java?slice=26:39"></script>

That creates a deep-belief network with the specified hidden-layer sizes (three hidden layers); the number of inputs being 784; outputs 10; momentum and learning rate; the specified random number generator; sets reconstruction cross entropy as the l;oss function; and implements no regularization.

Next, you iterate through the dataset with the MNISTDataFetcher.

<script src="https://github.com/SkymindIO/dl4j-examples/edit/master/src/main/java/org/deeplearning4j/mnist/MnistExample.java?slice=39:46"></script>

Finally, you evaluate the performance of your DBN:

<script src="https://github.com/SkymindIO/dl4j-examples/edit/master/src/main/java/org/deeplearning4j/mnist/MnistExample.java?slice=46:56"></script>

This will print out the f1 score of the prediction.

Note that the eval class combines [confusion matrices](../glossary.html#confusionmatrix) and f1 scores to allow for easy display and evaluation of data by allowing input of outcome matrices. This is useful for tracking how well your network trains over time. 

The f1 score will be a percentage. It's basically the probability that your guesses are correct. Eighty-six percent is industry standard; a solid deep-learning network should be capable of scores in the high 90s.

If you run into trouble, try modifying the hidden-layer sizes, and tweaking other parameters such as learning rate and momentum to get the f1 score up.

Next, we'll show you how to use [distributed and multithreaded computing](../scaleout.html) to train your networks more quickly. To read about another type of deep net, the deep autoencoder, [click here](../deepautoencoder.html). 
