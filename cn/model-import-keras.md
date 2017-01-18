---
title: Deeplearning4j Keras Model Import
layout: default
---

# DL4J Keras Model Import

The `deeplearning4j-modelimport` module provides routines for importing neural network models originally configured
and trained using [Keras](https://keras.io/), the most popular python deep learning library that provides abstraction
layers on top of both [Theano](http://deeplearning.net/software/theano/) and [TensorFlow](https://www.tensorflow.org)
backends. You can learn more about saving Keras models on the Keras [FAQ Page](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

The `IncompatibleKerasConfigurationException` indicates that you are attempting to import a Keras model configuration
that is not currently supported (either because model import does not cover it, or DL4J does not implement the model,
layer, or feature).

You can inquire further by visiting the [DL4J gitter channel](https://gitter.im/deeplearning4j/deeplearning4j). You
might consider filing a [feature request via Github](https://github.com/deeplearning4j/deeplearning4j/issues) so that
this missing functionality can be placed on the DL4J development roadmap or even sending us a pull request with the
necessary changes!

Check back for frequent updates to both the model import module *and* to this page!
