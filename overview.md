---
title: 
layout: default
---

# Neural nets overview

Neural networks are a set of algorithms, modeled after the human brain, that are designed to recognize patterns. They are sensors: a form of machine perception. Deep learning is a name for a certain type of stacked neural network composed of several node layers. (A node is loosely patterned on the human neuron, firing when it encounters the right stimulus; a node layer is a row of those neuron like switches that turn on or off as the input is fed through the net.) Each layer's output is simultaneously the subsequent layer's input, starting from an initial input layer.  

Deep-learning networks are distinguished from the more commonplace single-hidden-layer neural networks by their depth; that is, the number of node layers through which data is passed in a multistep process of pattern recognition. Traditional machine learning relies on shallow net, composed of input and output layers, and a hidden layer in between. More than three layers (including input and output) is "deep" learning. 

In deep-learning networks, each layer of nodes is pre-trained on a distinct set of features based on the previous layer's outputs. The further you go into the neural net, the more complex the features your nodes can recognize, since they combine features from the previous layer. This is known as feature hierarchy. It makes deep-learning networks capable of handling very complex data sets, including [nonlinear functions](../glossary.html#nonlineartransformfunction). Above all, they are capable of discovering latent structures within unsupervised data, which is the vast majority of data in the world. 

Deep-learning networks perform feature creation/extraction without human intervention, unlike most off-the-shelf, machine-learning software. Each node layer of the network learns features by repeatedly trying to reconstruct a training set, by attempting to minimize the difference between the network's output and an established benchmark of known, structured data. 

In the process, these networks learn to recognize correlations between certain relevant features and optimal results -- they draw connections between feature signals and how they label data. A trained deep-learning network can then be applied to unstructured data, giving it access to much more input than machine learning nets. This is a recipe for performance: the more data a net can train on, the more accurate it is likely to be. Deep learning's ability to process the huge quantities of unsupervised data give it a distinct advantage over previous algorithms. 

Deep-learning networks end in an output layer: a logistic, or softmax, classifier that assigns a likelihood to a particular outcome or label. We call that predictive, but it is predictive in a broad sense. Given raw data, a deep-learning network will decide, for example, the input data is 90 percent likely to represent a person. 

Next, we'll show you how to implement the simplest machine-learning network, a [Restricted Boltzmann machine](../restrictedboltzmannmachine.html). 
