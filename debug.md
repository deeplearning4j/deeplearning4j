---
title: Visualizing deep learning
layout: default
---

# Visualizing deep learning

Deep-learning networks can be hard to debug, but Deeplearning4j's networks have built-in support for runtime visualization of the weight matrices for each neural network and its bias. You can also visualize the activations of the network. 

To visualize is to debug, and to debug is to visualize. Weight matrices and their gradients should reflect a [Gaussian distribution](http://deeplearning4j.org/glossary.html#gaussian) most of the time. It they're flat, ragged or shaped like a barbell, there's something wrong. If they're close to a Bell curve, you're on the right track.

*Gaussian Matrices*

![Alt text](../img/weighthist.png)

Activations should start out gray. If they become black and white quickly, the weights are too large and have converged early. 

![Alt text](../img/activations.png)

For a more in-depth exploration of the topic, see [Visually Debugging Restricted Boltzmann Machine Training with a 3D Example](http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf), by Jason Yosinski and Hod Lipson.

You may also wonder what happens with neural-net training over time; i.e. does training change as it progresses? You can use filters to ensure that each neuron learns multiple features later on in training. That is, at later phases of training, neurons will activate on different parts of certain features.