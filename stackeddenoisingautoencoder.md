---
title: 
layout: default
---

*previous* - [denoising autoencoders](../denoisingautoencoder.html)
# stacked denoising autoencoders

A stacked denoising autoencoder is to a denoising autoencoder what a deep-belief network is to a [restricted Boltzmann machine](../restrictedboltzmannmachine.html). A key function of SDAs, and deep learning more generally, is their capacity for unsupervised pre-training, layer by layer, as input is fed through. Once each layer is pre-trained to conduct feature selection and extraction on the input from the preceding layer, a second stage of supervised fine tuning can follow. 

A word on stochastic corruption in SDAs: Denoising autoencoders shuffle data around and learn about that data by attempting to reconstruct it. The act of shuffling is the noise, and the job of the network is to recognize the features within the noise that will allow it to classify the input. When a network is being trained, it generates a model, and measures the distance between that model and the benchmark through a loss function. Its attempts to minimize the loss function involve resampling the shuffled inputs and re-reconstructing the data, until it finds those inputs which bring its model closest to what it has been told is true. 

The serial resamplings are based on a generative model to randomly provide data to be processed. This is known as a Markov Chain, and more specifically, a Markov Chain Monte Carlo algorithm that steps through the data set seeking a representative sampling of indicators that can be used to construct more and more complex features.

### parameters & corruption levels

See the [parameters common to all multilayer networks](../multinetwork.html). 

The amount of noise to apply to the input will take the form of a percentage. Typically 30 percent (0.3) is fine, but if you have a small amount of data, you may want to consider adding more.

### initiating a stacked denoising autoencoder

Here's how you set up a single-thread stacked denoising autoencoder: 

To create it, you simply instantiate an object of the class [StackedDenoisingAutoEncoder](../doc/org/deeplearning4j/sda/StackedDenoisingAutoEncoder.html).

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/StackedDenoisingAutoEncoderExample.java?slice=30:35"></script>

This creates a stacked denoising autoencoder with the specified parameters. 

You can test your newly trained network by feeding it unstructured data and checking the output. 

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/StackedDenoisingAutoEncoderExample.java?slice=36:41"></script>

Passing an iterator to pretrain and finetune makes them automatically iterate over the dataset. After those two tasks, all that remains is creating the output, or prediction, which is expressed as an f1 score.

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/StackedDenoisingAutoEncoderExample.java?slice=52:62"></script>

Note that the eval class combines [confusion matrices](../glossary.html#confusionmatrix) and f1 scores to allow for easy display and evaluation of data by allowing input of outcome matrices. This is useful for tracking how well your network trains over time. 

The f1 score will be a percentage. It's basically the probability that your guesses are correct. Eighty-six percent is industry standard; a solid deep-learning network should be capable of scores in the high 90s.

If you run into trouble, try modifying the hidden layer sizes, and tweaking other parameters to get the f1 score up.

Next, we'll show you how to use [distributed and multithreaded computing](../scaleout.html) to train your networks more quickly.
