---
title: Creating deep-learning networks
layout: default
---

# Creating deep-learning networks

## Single thread

While single-threaded and multithreaded deep-learning networks are just two ways of distributing the task load, real differences arise between single network and multinetwork deep-learning systems, as well as stochastic and nonstochastic generative models.

Single-network neural nets are composed of just two layers, like a [cat's cradle](https://en.wikipedia.org/wiki/File:Cat's_cradle_soldier's_bed.png). Multinetwork nets link double-layer nets together in long chains, each one functioning as a subnetwork in its own right. 

### Continuous restricted boltzmann machine

Restricted Boltzmann machines are composed of binomial neurons; that is, they can only activated with the values one and zero. Continuous restricted Boltzmann machines accept decimals. That is the only difference. 

### Parameters

### Input

### Denoising autoencoder

An autoencoder is a neural network used for dimensionality reduction; that is, for feature selection and extraction. Autoencoders with more hidden layers than inputs run the risk of learning the [identity function](https://en.wikipedia.org/wiki/Identity_function) -- where the output simple equals the input -- thereby becoming useless. 

Denoising autoencoders are an extension of the basic autoencoder, and represent a stochastic version of the autoencoder. Denoising autoencoders attempt to address identity-function risk by randomly corrupting input (i.e. introducing noise) that the autoencoder must then reconstruct, or denoise. 
<!---
### parameters

### input

### initiating a denoising autoencoder

Setting up a single-thread denoising autoencoder is easy. 

To create the machine, you simply instantiate an object of the class [CLASS].

CODE BLOCK MACHINE CREATION TK

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

CODE BLOCK TRAINING SET TK

Now that you have instantiated the machine and created the training set, it's time to train the network. 

CODE BLOCK TRAINING THE MACHINE TK

You can test your trained network by feeding it unstructured data and checking the output. 

Here are the code blocks for a multithread denoising autoencoder:

Create the machine:

CODE BLOCK MACHINE CREATION TK

Create the training set:

CODE BLOCK TRAINING SET TK

Train the machine:

CODE BLOCK TRAINING THE MACHINE TK
