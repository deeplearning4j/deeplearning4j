---
title: Keras Import Loss Functions
short_title: Losses
description: Supported Keras loss functions.
category: Keras Import
weight: 4
---

## Supported loss functions

DL4J supports all available [Keras losses](https://keras.io/losses) (except for `logcosh`), namely:

* <i class="fa fa-check-square-o"></i> mean_squared_error
* <i class="fa fa-check-square-o"></i> mean_absolute_error
* <i class="fa fa-check-square-o"></i> mean_absolute_percentage_error
* <i class="fa fa-check-square-o"></i> mean_squared_logarithmic_error
* <i class="fa fa-check-square-o"></i> squared_hinge
* <i class="fa fa-check-square-o"></i> hinge
* <i class="fa fa-check-square-o"></i> categorical_hinge
* <i class="fa fa-square-o"></i> logcosh
* <i class="fa fa-check-square-o"></i> categorical_crossentropy
* <i class="fa fa-check-square-o"></i> sparse_categorical_crossentropy
* <i class="fa fa-check-square-o"></i> binary_crossentropy
* <i class="fa fa-check-square-o"></i> kullback_leibler_divergence
* <i class="fa fa-check-square-o"></i> poisson
* <i class="fa fa-check-square-o"></i> cosine_proximity

The mapping of Keras loss functions can be found in [KerasLossUtils](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasLossUtils.java).