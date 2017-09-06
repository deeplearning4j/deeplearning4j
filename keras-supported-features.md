---
title: Deeplearning4j Keras Model Import - supported features
layout: default
---
# Keras model import: supported features

While not every concept in DL4J has an equivalent in Keras and vice versa, many of the key concepts can be matched. Importing keras models into DL4J is done in our [deeplearning4j-modelimport](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras) module. Below is a comprehensive list of currently supported features.

## Layers
Mapping keras to DL4J layers is done in the [layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers) sub-module of model import. The structure of this project loosely reflects the structure of Keras.

### [Core Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core)
- [x] [Dense](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDense.java)
- [x] [Activation](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasActivation.java)
- [x] [Dropout](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDropout.java)
- [x] [Flatten](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasFlatten.java)
- [x] [Reshape](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasReshape.java)
- [x] [Merge](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
- [ ] Permute
- [ ] RepeatVector
- [ ] Lambda
- [ ] ActivityRegularization
- [ ] Masking

### [Convolutional Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional)
- [x] [Conv1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution1D.java)
- [x] [Conv2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution2D.java)
- [ ] Conv3D
- [x] [AtrousConvolution1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
- [x] [AtrousConvolution2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
- [ ] SeparableConv2D
- [ ] Conv2DTranspose
- [ ] Cropping1D
- [ ] Cropping2D
- [ ] Cropping3D
- [ ] UpSampling1D
- [ ] UpSampling2D
- [ ] UpSampling3D
- [x] [ZeroPadding1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding1D.java)
- [x] [ZeroPadding2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding2D.java)
- [ ] ZeroPadding3D

### [Pooling Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling)
- [x] [MaxPooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
- [x] [MaxPooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
- [ ] MaxPooling3D
- [x] [AveragePooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
- [x] [AveragePooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
- [ ] AveragePooling3D
- [x] [GlobalMaxPooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
- [x] [GlobalMaxPooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
- [x] [GlobalAveragePooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
- [x] [GlobalAveragePooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)

### Locally-connected Layers
DL4J currently does not support Locally-connected layers.
- [ ] LocallyConnected1D
- [ ] LocallyConnected2D

### [Recurrent Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent)
- [ ] SimpleRNN
- [ ] GRU
- [x] [LSTM](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent/KerasLstm.java)

### [Embedding Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings)
- [x] [Embedding](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings/KerasEmbedding.java)

### [Merge Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
- [x] Add / add
- [x] Multiply / multiply
- [ ] Subtract / subtract
- [ ] Average / average
- [ ] Maximum / maximum
- [ ] Concatenate / concatenate
- [ ] Dot / dot

### [Advanced Activation Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations)
- [x] [LeakyReLU](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasLeakyReLU.java)
- [ ] PReLU
- [ ] ELU
- [ ] ThresholdedReLU

### [Normalization Layers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization)
- [x] [BatchNormalization](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization/KerasBatchNormalization.java)

### Noise Layers
Currently, DL4J does not support noise layers.

- [ ] GaussianNoise
- [ ] GaussianDropout
- [ ] AlphaDropout

### Layer Wrappers
DL4j does not have the concept of layer wrappers, but there is an implementation of bi-directional LSTMs available [here](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/GravesBidirectionalLSTM.java).
- [ ] TimeDistributed
- [ ] Bidirectional

## [Losses](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasLossUtils.java)

- [x] mean_squared_error
- [x] mean_absolute_error
- [x] mean_absolute_percentage_error
- [x] mean_squared_logarithmic_error
- [x] squared_hinge
- [x] hinge
- [x] categorical_hinge
- [ ] logcosh
- [x] categorical_crossentropy
- [x] sparse_categorical_crossentropy
- [x] binary_crossentropy
- [x] kullback_leibler_divergence
- [x] poisson
- [x] cosine_proximity

## [Activations](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasActivationUtils.java)
- [x] softmax
- [x] elu
- [x] selu
- [x] softplus
- [x] softsign
- [x] relu
- [x] tanh
- [x] sigmoid
- [x] hard_sigmoid
- [x] linear

## [Initializers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasInitilizationUtils.java).

- [x] Zeros
- [x] Ones
- [ ] Constant
- [ ] RandomNormal
- [x] RandomUniform
- [ ] TruncatedNormal
- [ ] VarianceScaling
- [ ] Orthogonal
- [x] Identity
- [ ] lecun_uniform
- [x] lecun_normal
- [x] glorot_normal
- [x] glorot_uniform
- [x] he_normal
- [x] he_uniform

## [Regularizers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasRegularizerUtils.java)
- [x] l1
- [x] l2
- [x] l1_l2

## [Constraints](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasConstraintUtils.java)
- [x] max_norm
- [x] non_neg
- [x] unit_norm
- [x] min_max_norm

## Metrics
- [x] binary_accuracy
- [x] categorical_accuracy
- [x] sparse_categorical_accuracy
- [x] top_k_categorical_accuracy
- [x] sparse_top_k_categorical_accuracy

## Optimizers
- [x] SGD
- [x] RMSprop
- [x] Adagrad
- [x] Adadelta
- [x] Adam
- [x] Adamax
- [x] Nadam
- [ ] TFOptimizer
