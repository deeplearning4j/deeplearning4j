---
title: 
layout: default
---

*previous* - [restricted Boltzmann machine](../restrictedboltzmannmachine.html)
# convolutional restricted boltzmann machine

[Convolutional RBMs](http://www.cs.toronto.edu/~norouzi/research/papers/masters_thesis.pdf) data, because they are able to handle many dimensions in data at once. 

They are a type of restricted Boltzmann machine, in that there is no communication between the nodes of any given layer with other nodes of the same layer. 

In a typical RBM, each node of one layer is connected to all nodes of the next, and this is known as symmetry. Convolutional nets are not symmetric. The node of one layer will connect to its direct counterpart, as well as nodes to the right and left of it, but to no others. So rather than the total overlap of connections seen in an RBM, they have a partial overlap, and there will be certain nodes of layer A that have no connection whatsoever with certain distant nodes of layer B.

In addition, convolutional nets’ algorithm analyzes images differently than typical RBMs. While RBMs learn to reconstruct and identify the features of each image as a whole, convolutional nets learn images in pieces. Picture a grid superimposed on an image, which is broken down into a series of squares. The convolutional net learns each of those squares and then weaves them together in a later stage.
