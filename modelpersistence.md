---
title: 
layout: default
---

Saving and Loading a Neural Network
========================================


Neural Networks that implement the [../doc/org/deeplearning4j/nn/Persistable.html](Persistable) interface

are persistable via java serialization.

With this, binary, but portable any neural network (or [../doc/org/deeplearning4j/datasets/DataSet.html] for
that matter)
 
are saved and can also be sent across a network. This is used in deeplearning4j for distributed training of 

of neural networks, and allows neural networks to be loaded in to memory afterwards (also resuming training!)

for use with a REST API or other services.
