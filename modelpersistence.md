---
title: 
layout: default
---

# Saving and loading a neural network

Neural networks that implement the [Persistable](../doc/org/deeplearning4j/nn/Persistable.html) interface are persistable via Java serialization.

With [this binary](../doc/org/deeplearning4j/datasets/DataSet.html), any neural network can be saved and sent across a network. Deeplearning4j uses it for the distributed training of neural nets. 

It allows neural networks to be loaded into memory afterward (to resume training), and can be used with a REST API or other tools.