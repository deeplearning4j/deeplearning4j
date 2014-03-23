---
layout: default
---

# neural nets overview 

Go [here](../gettingstarted.html) to get started without fanfare. Our Github repo lives [here](https://github.com/agibsonccc/java-deeplearning).

### purpose

Deeplearning4j is the first commercial-grade deep-learning library for Java. It is meant to be used in business environments, rather than as a research tool for extensive data exploration. 
Deeplearning4j is most helpful in solving distinct problems (like identifying cat faces). 

Deeplearning4j aims to be cutting-edge plug and play, more convention than configuration. By following its conventions, you get an infinitely scalable deep-learning architecture. It's made for data input and neural net training at scale, and its output should be highly compact and accurate predictive models. Trained correctly, these neural nets attain accuracy levels of more than 95 percent.

By following the links at the bottom of each page, you will learn to set up, and train with sample data, several types of deep-learning networks. These include single- and multithread networks, [Restricted Boltzmann machines](../restrictedboltzmannmachine.html), [deep-belief networks](../deepbeliefnetwork.html) and [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html). 

### introduction

Neural networks are a set of algorithms, modeled after the human brain, that are designed to recognize patterns. They are sensors: a form of machine perception. Deep learning is a name for a certain type of stacked neural network composed of several node layers. Each layer's output is simultaneously the subsequent layer's input, starting from an initial input layer.  

Deep-learning networks are distinguished from the more commonplace single-hidden-layer neural networks by their depth; that is, the number of node layers through which data is passed in a multistep process of pattern recognition. Three or more intermediate layers between input and output is deep learning. Anything less is simply machine learning. 

In deep-learning networks, each layer of nodes -- which are the functional equivalent of biological neurons -- is pre-trained on a distinct set of features based on the previous layer's outputs. This makes deep-learning networks capable of  handling very complex data sets, including [nonlinear functions](https://en.wikipedia.org/wiki/Nonlinear_system). Above all, they are capable of discovering latent structures within unstructured data, which is the vast majority of data in the world. 

Deep-learning networks perform feature creation/extraction without human intervention, unlike most off-the-shelf, machine-learning software. Each node layer of the network learns features by repeatedly trying to reconstruct a training set, by attempting to minimize the difference between the network's output and an established benchmark of known, structured data. 

In the process, these networks come to recognize correlations between certain relevant features and optimal results. A trained deep-learning network can then be applied to unstructured data. This is beyond the scope of traditional machine learning. 

Deep-learning networks end in an output layer: a logistic, or softmax, classifier that assigns a likelihood to a particular outcome or label. We call that predictive, but it is predictive in a broad sense. Given raw data, a deep-learning network will decide, for example, the input data is 90 percent likely to represent a person. 

Next, we'll show you how to implement the simplest machine-learning network, a [Restricted Boltzmann machine](../restrictedboltzmannmachine.html). 

