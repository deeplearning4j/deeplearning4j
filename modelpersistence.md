---
title: Saving and Loading a Neural Network
layout: default
---

# Saving and Loading a Neural Network

The `ModelSerializer` is a class which handles loading and saving models. There are two methods for saving models shown in the examples through the link. The first example saves a normal multilayer network, the second one saves a [computation graph](https://deeplearning4j.org/compgraph).

Here is a [basic example](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/modelsaving) with code to save a computation graph using the `ModelSerializer` class, as well as an example of using ModelSerializer to save a neural net built using MultiLayer configuration.

## RNG Seed

If your model uses probabilities (i.e. DropOut/DropConnect), it may make sense to save it separately, and apply it after model is restored; i.e:

```bash
 Nd4j.getRandom().setSeed(12345);
 ModelSerializer.restoreMultiLayerNetwork(modelFile);
```

This will guarantee equal results between sessions/JVMs.

<!---
Verify up to date before re-including
Below is an example of loading either a multilayernetwork or a computation graph:
[See the example test](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/util/ModelSerializerTest.java)
-->
