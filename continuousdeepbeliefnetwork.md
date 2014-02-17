---
title: 
layout: default
---

# continuous deep-belief network

Deep-belief networks are composed of binomial neurons; that is, they can only activated with the values one and zero. Continuous deep belief networks accept decimals. That is the only difference. 

### parameters  
Please also see [the multi layer network parameters common to all multi layer networks](../multinetwork.html).

#### k 

K is the number of times to run [contrastive divergence]({{ site.baseurl }}/glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the Markov chain. In composing the restricted Boltzmann machine, a typical value is one.
<!---
### input

### initiating a continuous deep-belief network

Setting up a single-thread continuous deep-belief network is easy. 

To create the machine, you simply instantiate an object of the [class]({{ site.baseurl }}/doc/com/ccc/deeplearning/dbn/CDBN.html).

CODE BLOCK MACHINE CREATION TK

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

CODE BLOCK TRAINING SET TK

Now that you have instantiated the machine and created the training set, it's time to train the network. 

CODE BLOCK TRAINING THE MACHINE TK

You can test your trained network by feeding it unstructured data and checking the output. 

Here are the code blocks for a multithread continuous deep-belief network:

Create the machine:

CODE BLOCK MACHINE CREATION TK

Create the training set:

CODE BLOCK TRAINING SET TK

Train the machine:

CODE BLOCK TRAINING THE MACHINE TK
