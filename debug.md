---
title: 
layout: default
---

visualizing deep learning
====================================

Deep-learning networks can be hard to debug, but Deeplearning4j's networks have built-in support for runtime visualization of the weight matrices for each neural network and its bias. You can also visualize the activations of the network. 

To visualize is to debug. Weight matrices and their gradients should be a gaussian distribution most of the time.

![Alt text](../img/weighthist.png)


Activations should start out gray. If they're black and white, the weights are too large and have converged early. 

![Alt text](../img/activations.png)

For a more in-depth exploration of the topic, see [Visually Debugging Restricted Boltzmann Machine Training with a 3D Example](http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf), by Jason Yosinski and Hod Lipson.

