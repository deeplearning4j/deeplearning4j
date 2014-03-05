---
title: 
layout: default
---



Visualizing DeepLearning
====================================





Deep Learning neural networks can be hard to debug.

Fortunately, deeplearning4j's neural networks

have built in support for runtime visualization of

the weight matrices for each neural networks and their bias.

You can also visualize the activations of the network.

General hints are as follows. 

Weight matrices and their gradients should be a gaussian distribution most of the time.

![Alt text](../img/weighthist.png)


For the activations, they should be start out grey.

If they are black and white, this means the weights are too large and

have converged early. 

![Alt text](../img/activations.png)

References:
Visually Debugging Restricted Boltzmann Machine Training
with a 3D Example by 

Jason Yosinski yosinski@cs.cornell.edu
Hod Lipson hod.lipson@cornell.edu

