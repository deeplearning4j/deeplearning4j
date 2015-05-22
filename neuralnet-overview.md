---
title: 
layout: default
---

# Neural Nets Overview

Neural networks are a set of algorithms, modeled after the human brain, that are designed to recognize patterns. They are sensors: a form of machine perception. Deep learning is a name for a certain type of stacked neural network composed of several node layers. (A node is a place where computation happens, loosely patterned on the human neuron and firing when it encounters the right stimulus; a node layer is a row of those neuronlike switches that turn on or off as the input is fed through the net.) Each layer's output is simultaneously the subsequent layer's input, starting from an initial input layer.  

Deep-learning networks are distinguished from the more commonplace single-hidden-layer neural networks by their depth; that is, the number of node layers through which data is passed in a multistep process of pattern recognition. Traditional machine learning relies on shallow nets, composed of input and output layers and a hidden layer in between. More than three layers (including input and output) is "deep" learning. 

In deep-learning networks, each layer of nodes is pre-trained on a distinct set of features based on the previous layer's outputs. The further you advance into the neural net, the more complex the features your nodes can recognize, since they aggregate features from the previous layer. This is known as feature hierarchy, and it is a hierarchy of complexity. It makes deep-learning networks capable of handling very complex data sets with billions of parameters being passed through [nonlinear functions](../glossary.html#nonlineartransformfunction). Above all, these nets are capable of discovering latent structures within unsupervised, unstructured data, which is the vast majority of data in the world. Another word for unstructured data is simply media; i.e. pictures, texts, video and audio recordings. 

Deep-learning networks perform feature creation/extraction without human intervention, unlike most off-the-shelf, machine-learning software. Each node layer of the network learns features by repeatedly trying to reconstruct a training set, attempting to minimize the difference between the network's output and an established benchmark of known, supervised data known as the test set. 

In the process, these networks learn to recognize correlations between certain relevant features and optimal results -- they draw connections between feature signals and the labels applied to data. A deep-learning network trained on supervised data can then be applied to unstructured data, giving it access to much more input than machine-learning nets. This is a recipe for higher performance: the more data a net can train on, the more accurate it is likely to be. (Bad algorithms trained on lots of data can outperform good algorithms trained on very little.) Deep learning's ability to process and learn from huge quantities of unsupervised data give it a distinct advantage over previous algorithms. 

Deep-learning networks end in an output layer: a logistic, or softmax, classifier that assigns a likelihood to a particular outcome or label. We call that predictive, but it is predictive in a broad sense. Given raw data in the form of an image, a deep-learning network may decide, for example, that the input data is 90 percent likely to represent a person. 

---
### Where to Start

Two key classes help you get started building any type of neural net you need.

- [NeuralNetConfiguration](../neuralnet-configuration.html)
- [MultiLayerConfiguration](../multilayer-configuration.html)

Additional information regarding neural net types and standard configurations can be found for the following:

* [Restricted Boltzmann Machines](../restrictedboltzmannmachine.html)
* [Deep-Belief Nets](../deepbeliefnetwork.html)
* [Deep Autoencoders](../deepautoencoder.html)
* [Denoising Autencoders](../denoisingautoencoder.html)
* [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html)
* [Convolutional Nets](../convolutionalnets.html)
* [Recursive Neural Tensor Networks](../recursiveneuraltensornetwork.html).
