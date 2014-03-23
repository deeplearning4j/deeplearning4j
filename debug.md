---
title: 
layout: default
---

# visualizing deep learning

Deep-learning networks can be hard to debug, but Deeplearning4j's networks have built-in support for runtime visualization of the weight matrices for each neural network and its bias. You can also visualize the activations of the network. 

To visualize is to debug, and to debug is to visualize. Weight matrices and their gradients should reflect a [Gaussian distribution](http://deeplearning4j.org/glossary.html#gaussian) most of the time. It they're flat, ragged or shaped like a barbell, there's something wrong. If they're close to a Bell curve, you're on the right track.

*Gaussian Matrices*

![Alt text](../img/weighthist.png)

Activations should start out gray. If they become black and white quickly, the weights are too large and have converged early. 

![Alt text](../img/activations.png)

For a more in-depth exploration of the topic, see [Visually Debugging Restricted Boltzmann Machine Training with a 3D Example](http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf), by Jason Yosinski and Hod Lipson.



Another typical situation is to wonder what happens over time with neural net training. One trick with the filters is to use filters to ensure each neuron later on in training learns multiple features. At later parts of training neurons will activivate on different parts of features.