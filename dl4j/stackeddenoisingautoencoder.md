---
title: deeplearning4j
layout: default
---



### Multinetwork/Stacked Denoising Autoencoder

A [stacked denoising autoencoder](http://deeplearning.net/tutorial/SdA.html) is to a denoising autoencoder what a deep-belief network is to a [restricted Boltzmann machine](../restrictedboltzmannmachine.html). A key function of SDAs, and deep learning more generally, is their capacity for unsupervised pre-training, layer by layer, as input is fed through. Once each layer is pre-trained to conduct feature selection and extraction on the input from the preceding layer, a second stage of supervised fine tuning can follow. 

A word on stochastic corruption in SDAs: Denoising autoencoders shuffle data around and learn about that data by attempting to reconstruct it. The act of shuffling is the noise, and the job of the network is to recognize the features within the noise that will allow it to classify the input. When a network is being trained, it generates a model, and measures the distance between that model and the benchmark through a loss function. Its attempts to minimize the loss function involve resampling the shuffled inputs and re-reconstructing the data, until it finds those inputs which bring its model closest to what it has been told is true. 

The serial resamplings are based on a generative model to randomly provide data to be processed. This is known as a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain#Steady-state_analysis_and_limiting_distributions), and more specifically, a [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) algorithm that steps through the data set seeking a representative sampling of indicators that can be used to construct more and more complex features.


### PARAMETERS  Please also see [the multi layer network parameters common to all multi layer networks](../multinetwork.html)
  ####Corruption Level - The amount of noise to apply to the input. This is a percentage. Typically 30% (0.3) is fine, but if you have a small amount of data, you may want to consider adding more.

### Initiating a stacked denoising autoencoder

Setting up a single-thread stacked denoising autoencoder is easy. 

To create the machine, you simply instantiate an object of the [class](../doc/com/ccc/deeplearning/sda/StackedDenoisingAutoEncoder.html).

CODE BLOCK MACHINE CREATION TK

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

CODE BLOCK TRAINING SET TK

Now that you have instantiated the machine and created the training set, it's time to train the network. 

CODE BLOCK TRAINING THE MACHINE TK

You can test your trained network by feeding it unstructured data and checking the output. 

Here are the code blocks for a multithread stacked denoising autoencoder:

Create the machine:

CODE BLOCK MACHINE CREATION TK

Create the training set:

CODE BLOCK TRAINING SET TK

Train the machine:

CODE BLOCK TRAINING THE MACHINE TK
