---
title: 
layout: default
---

# Saving and Loading a Neural Network

Neural networks that implement the Persistable interface are persistable via Java serialization. It looks like this in the [Javadoc](http://deeplearning4j.org/doc/), where you can find it by searching for "Persistable."

![Alt text](../img/persistable.png) 

With this binary, any neural network can be saved and sent across a network. Deeplearning4j uses it for the distributed training of neural nets. 

![Alt text](../img/datasets.png) 

It allows neural networks to be loaded into memory afterward (to resume training), and can be used with a REST API or other tools.
