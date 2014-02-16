---
title: deeplearning4j
layout: default
---


### Single Network/Continuous Restricted Boltzmann machine


A continuous Restricted Boltzmann Machine is a form of RBM that accepts

continuous input via a different flavor of contrastive divergence sampling.

This allows it to handle things like word count vectors that are normalized to probabilities

or pixels of an image.



###PARAMETERS Please also see [the single layer network parameters common to all single networks]({{ site.baseurl }}/singlelayernetwork.html)

####k - The number of times to run [contrastive divergence]({{ site.baseurl }}/glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the markov chain

composing the restricted boltzmann machine. A typical value of 1 is fine.





### Initiating a continuous restricted Boltzmann machine

Setting up a single-thread continuous restricted Boltzmann machine is easy. 

To create the machine, you simply instantiate an object of the [class]({{ site.baseurl }}/doc/com/ccc/deeplearning/rbm/CRBM.html).

CODE BLOCK MACHINE CREATION TK

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

CODE BLOCK TRAINING SET TK

Now that you have instantiated the machine and created the training set, it's time to train the network. 

CODE BLOCK TRAINING THE MACHINE TK

You can test your trained network by feeding it unstructured data and checking the output. 

Here are the code blocks for a multithread continuous restricted Boltzmann machine:

Create the machine:

CODE BLOCK MACHINE CREATION TK

Create the training set:

CODE BLOCK TRAINING SET TK

Train the machine:

CODE BLOCK TRAINING THE MACHINE TK

