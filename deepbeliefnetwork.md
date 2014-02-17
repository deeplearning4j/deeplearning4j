---
title: 
layout: default
---


# deep-belief network

For our purposes, a [deep-belief network](http://www.scholarpedia.org/article/Deep_belief_networks) can be defined as a stack of restricted Boltzmann machines in which each layer communicates with its previous and subsequent layers, while the nodes of any single layer do not communicate with each other laterally. With the exception of the  first and final layers, each hidden layer has a double role: it serves as the hidden layer to the higher nodes before, and as the input layer to the lower nodes after. It is a network of networks. 

Deep belief networks are used to recognize and generate images, video sequences and motion-capture data. 


### parameters

Please also see [the multilayer network parameters common to all multilayer networks]({{ site.baseurl }}/multinetwork.html)

#### k 
K is the number of times you run [contrastive divergence]({{ site.baseurl }}/glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the Markov chain.

composing the restricted boltzmann machine. A typical value of 1 is fine.

### initiating a deep-belief network

Setting up a single-thread deep belief network is easy. 

To create the machine, you simply instantiate an object of the [class]({{ site.baseurl }}/doc/com/ccc/deeplearning/dbn/DBN.html).

CODE BLOCK MACHINE CREATION TK

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

CODE BLOCK TRAINING SET TK

Now that you have instantiated the machine and created the training set, it's time to train the network. 

CODE BLOCK TRAINING THE MACHINE TK

You can test your trained network by feeding it unstructured data and checking the output. 

Here are the code blocks for a multithread deep-belief network:

Create the machine:

CODE BLOCK MACHINE CREATION TK

Create the training set:

CODE BLOCK TRAINING SET TK

Train the machine:

CODE BLOCK TRAINING THE MACHINE TK
