---
title: Deeplearning4j Keras Model Import - supported features
layout: default
redirect_from: keras
---

# Keras Model import: Supported Features

While not every concept in DL4J has an equivalent in Keras and vice versa, many of the key concepts can be matched. Importing keras models into DL4J is done in our [deeplearning4j-modelimport](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras) module. Below is a comprehensive list of currently supported features.

* [Layers](#layers)
* [Losses](#losses)
* [Activations](#activations)
* [Initializers](#initializers)
* [Regularizers](#regularizers)
* [Constraints](#constraints)
* [Metrics](#metrics)
* [Optimizers](#optimizers)


## <a name="layers">Layers</a>
Mapping keras to DL4J layers is done in the [layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers) sub-module of model import. The structure of this project loosely reflects the structure of Keras.

### [Core Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core)
* <i class="fa fa-check-square-o"></i> [Dense](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDense.java)
* <i class="fa fa-check-square-o"></i> [Activation](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasActivation.java)
* <i class="fa fa-check-square-o"></i> [Dropout](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDropout.java)
* <i class="fa fa-check-square-o"></i> [Flatten](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasFlatten.java)
* <i class="fa fa-check-square-o"></i> [Reshape](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasReshape.java)
* <i class="fa fa-check-square-o"></i> [Merge](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
* <i class="fa fa-square-o"></i> Permute
* <i class="fa fa-square-o"></i> RepeatVector
* <i class="fa fa-square-o"></i> Lambda
* <i class="fa fa-square-o"></i> ActivityRegularization
* <i class="fa fa-square-o"></i> Masking

### [Convolutional Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional)
* <i class="fa fa-check-square-o"></i> [Conv1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution1D.java)
* <i class="fa fa-check-square-o"></i> [Conv2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution2D.java)
* <i class="fa fa-square-o"></i> Conv3D
* <i class="fa fa-check-square-o"></i> [AtrousConvolution1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
* <i class="fa fa-check-square-o"></i> [AtrousConvolution2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
* <i class="fa fa-square-o"></i> SeparableConv2D
* <i class="fa fa-square-o"></i> Conv2DTranspose
* <i class="fa fa-square-o"></i> Cropping1D
* <i class="fa fa-square-o"></i> Cropping2D
* <i class="fa fa-square-o"></i> Cropping3D
* <i class="fa fa-check-square-o"></i> [UpSampling1D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling1D.java)
* <i class="fa fa-check-square-o"></i> [UpSampling2D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling2D.java)
* <i class="fa fa-square-o"></i> UpSampling3D
* <i class="fa fa-check-square-o"></i> [ZeroPadding1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding1D.java)
* <i class="fa fa-check-square-o"></i> [ZeroPadding2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding2D.java)
* <i class="fa fa-square-o"></i> ZeroPadding3D

### [Pooling Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling)
* <i class="fa fa-check-square-o"></i> [MaxPooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
* <i class="fa fa-check-square-o"></i> [MaxPooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
* <i class="fa fa-square-o"></i> MaxPooling3D
* <i class="fa fa-check-square-o"></i> [AveragePooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
* <i class="fa fa-check-square-o"></i> [AveragePooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
* <i class="fa fa-square-o"></i> AveragePooling3D
* <i class="fa fa-check-square-o"></i> [GlobalMaxPooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalMaxPooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalAveragePooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalAveragePooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)

### Locally-connected Layers
DL4J currently does not support Locally-connected layers.
* <i class="fa fa-square-o"></i> LocallyConnected1D
* <i class="fa fa-square-o"></i> LocallyConnected2D

### [Recurrent Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent)
* <i class="fa fa-square-o"></i> SimpleRNN
* <i class="fa fa-square-o"></i> GRU
* <i class="fa fa-check-square-o"></i> [LSTM](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent/KerasLstm.java)

### [Embedding Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings)
* <i class="fa fa-check-square-o"></i> [Embedding](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings/KerasEmbedding.java)

### [Merge Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
* <i class="fa fa-check-square-o"></i> Add / add
* <i class="fa fa-check-square-o"></i> Multiply / multiply
* <i class="fa fa-square-o"></i> Subtract / subtract
* <i class="fa fa-square-o"></i> Average / average
* <i class="fa fa-square-o"></i> Maximum / maximum
* <i class="fa fa-square-o"></i> Concatenate / concatenate
* <i class="fa fa-square-o"></i> Dot / dot

### [Advanced Activation Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations)
* <i class="fa fa-check-square-o"></i> [LeakyReLU](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasLeakyReLU.java)
* <i class="fa fa-square-o"></i> PReLU
* <i class="fa fa-square-o"></i> ELU
* <i class="fa fa-square-o"></i> ThresholdedReLU

### [Normalization Layers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization)
* <i class="fa fa-check-square-o"></i> [BatchNormalization](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization/KerasBatchNormalization.java)

### Noise Layers
Currently, DL4J does not support noise layers.

* <i class="fa fa-square-o"></i> GaussianNoise
* <i class="fa fa-square-o"></i> GaussianDropout
* <i class="fa fa-square-o"></i> AlphaDropout

### Layer Wrappers
DL4j does not have the concept of layer wrappers, but there is an implementation of bi-directional LSTMs available [here](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesBidirectionalLSTM.java).
* <i class="fa fa-square-o"></i> TimeDistributed
* <i class="fa fa-square-o"></i> Bidirectional

## <a name="losses">[Losses](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasLossUtils.java)</a>

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

## <a name="activations">[Activations](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasActivationUtils.java)</a>
* <i class="fa fa-check-square-o"></i> softmax
* <i class="fa fa-check-square-o"></i> elu
* <i class="fa fa-check-square-o"></i> selu
* <i class="fa fa-check-square-o"></i> softplus
* <i class="fa fa-check-square-o"></i> softsign
* <i class="fa fa-check-square-o"></i> relu
* <i class="fa fa-check-square-o"></i> tanh
* <i class="fa fa-check-square-o"></i> sigmoid
* <i class="fa fa-check-square-o"></i> hard_sigmoid
* <i class="fa fa-check-square-o"></i> linear

## <a name="initializers">[Initializers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasInitilizationUtils.java)</a>

* <i class="fa fa-check-square-o"></i> Zeros
* <i class="fa fa-check-square-o"></i> Ones
* <i class="fa fa-check-square-o"></i> Constant
* <i class="fa fa-check-square-o"></i> RandomNormal
* <i class="fa fa-check-square-o"></i> RandomUniform
* <i class="fa fa-check-square-o"></i> TruncatedNormal
* <i class="fa fa-check-square-o"></i> VarianceScaling
* <i class="fa fa-check-square-o"></i> Orthogonal
* <i class="fa fa-check-square-o"></i> Identity
* <i class="fa fa-check-square-o"></i> lecun_uniform
* <i class="fa fa-check-square-o"></i> lecun_normal
* <i class="fa fa-check-square-o"></i> glorot_normal
* <i class="fa fa-check-square-o"></i> glorot_uniform
* <i class="fa fa-check-square-o"></i> he_normal
* <i class="fa fa-check-square-o"></i> he_uniform

## <a name="regularizers">[Regularizers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasRegularizerUtils.java)</a>
* <i class="fa fa-check-square-o"></i> l1
* <i class="fa fa-check-square-o"></i> l2
* <i class="fa fa-check-square-o"></i> l1_l2

## <a name="constraints">[Constraints](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasConstraintUtils.java)</a>
* <i class="fa fa-check-square-o"></i> max_norm
* <i class="fa fa-check-square-o"></i> non_neg
* <i class="fa fa-check-square-o"></i> unit_norm
* <i class="fa fa-check-square-o"></i> min_max_norm

## <a name="metrics">Metrics</a>
* <i class="fa fa-check-square-o"></i> binary_accuracy
* <i class="fa fa-check-square-o"></i> categorical_accuracy
* <i class="fa fa-check-square-o"></i> sparse_categorical_accuracy
* <i class="fa fa-check-square-o"></i> top_k_categorical_accuracy
* <i class="fa fa-check-square-o"></i> sparse_top_k_categorical_accuracy

## <a name="optimizers">Optimizers</a>
* <i class="fa fa-check-square-o"></i> SGD
* <i class="fa fa-check-square-o"></i> RMSprop
* <i class="fa fa-check-square-o"></i> Adagrad
* <i class="fa fa-check-square-o"></i> Adadelta
* <i class="fa fa-check-square-o"></i> Adam
* <i class="fa fa-check-square-o"></i> Adamax
* <i class="fa fa-check-square-o"></i> Nadam
* <i class="fa fa-square-o"></i> TFOptimizer
