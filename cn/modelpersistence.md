---
title: Saving and Loading a Neural Network
layout: default
---

# Saving and Loading a Neural Network
The ModelSerializer is a class which handles loading and saving models. There are 2 methods for saving (in the examples below)
The first example saves a normal multi layer network, the second one saves a [computation graph](https://deeplearning4j.org/compgraph)

Here is a [basic Example](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/modelsaving) that provides the code to save a computation graph using the ModelSerializer class, and also an example of using ModelSerializer to save a Neural Net built using MultiLayer Configuration.  


<!---
Verify up to date before re-including
Below is an example of loading either a multilayernetwork or a computation graph:
[See the example test](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/util/ModelSerializerTest.java)
-->
