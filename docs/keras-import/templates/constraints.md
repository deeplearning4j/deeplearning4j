---
title: Keras Constraints
short_title: Constraints
description: Supported Keras constraints.
category: Keras Import
weight: 4
---

## Supported constraints

All [Keras constraints](https://keras.io/constraints) are supported:

* <i class="fa fa-check-square-o"></i> max_norm
* <i class="fa fa-check-square-o"></i> non_neg
* <i class="fa fa-check-square-o"></i> unit_norm
* <i class="fa fa-check-square-o"></i> min_max_norm

Mapping Keras to DL4J constraints happens in [KerasConstraintUtils](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasConstraintUtils.java).
