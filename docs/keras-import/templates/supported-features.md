---
title: Keras Import Supported Features
short_title: Supported Features
description: Supported Keras features.
category: Keras Import
weight: 2
---

## Keras Model Import: Supported Features

Little-known fact: Deeplearning4j's creator, Skymind, has two of the top
five [Keras contributors](https://github.com/keras-team/keras/graphs/contributors)
on our team, making it the largest contributor to Keras after Keras creator Francois
Chollet, who's at Google.

While not every concept in DL4J has an equivalent in Keras and vice versa, many of the
key concepts can be matched. Importing keras models into DL4J is done in
our [deeplearning4j-modelimport](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras)
module. Below is a comprehensive list of currently supported features.

* [Layers](#layers)
* [Losses](#losses)
* [Activations](#activations)
* [Initializers](#initializers)
* [Regularizers](#regularizers)
* [Constraints](#constraints)
* [Metrics](#metrics)
* [Optimizers](#optimizers)


## <a name="layers">Layers</a>
Mapping keras to DL4J layers is done in the [layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers) sub-module of model import. The structure of this project loosely reflects the structure of Keras.

### [Core Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core)
* <i class="far fa-check-square" style="color:#008000"></i> [Dense](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDense.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Activation](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasActivation.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Dropout](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDropout.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Flatten](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasFlatten.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Reshape](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasReshape.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Merge](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Permute](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasPermute.java)
* <i class="far fa-check-square" style="color:#008000"></i> [RepeatVector](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasRepeatVector.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Lambda](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasLambda.java)
* <i class="fas fa-times" style="color:#FF0000"></i> ActivityRegularization
* <i class="far fa-check-square" style="color:#008000"></i> [Masking](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMasking.java)
* <i class="far fa-check-square" style="color:#008000"></i> [SpatialDropout1D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasSpatialDropout.java)
* <i class="far fa-check-square" style="color:#008000"></i> [SpatialDropout2D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasSpatialDropout.java)
* <i class="far fa-check-square" style="color:#008000"></i> [SpatialDropout3D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasSpatialDropout.java)

### [Convolutional Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional)
* <i class="far fa-check-square" style="color:#008000"></i> [Conv1D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Conv2D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Conv3D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution3D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [AtrousConvolution1D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [AtrousConvolution2D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
* <i class="fas fa-times" style="color:#FF0000"></i> SeparableConv1D
* <i class="far fa-check-square" style="color:#008000"></i> [SeparableConv2D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasSeparableConvolution2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Conv2DTranspose](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasDeconvolution2D.java)
* <i class="fas fa-times" style="color:#FF0000"></i> Conv3DTranspose
* <i class="far fa-check-square" style="color:#008000"></i> [Cropping1D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasCropping1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Cropping2D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasCropping2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [Cropping3D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasCropping3D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [UpSampling1D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [UpSampling2D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [UpSampling3D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [ZeroPadding1D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [ZeroPadding2D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [ZeroPadding3D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding3D.java)

### [Pooling Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling)
* <i class="far fa-check-square" style="color:#008000"></i> [MaxPooling1D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [MaxPooling2D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [MaxPooling3D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling3D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [AveragePooling1D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [AveragePooling2D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [AveragePooling3D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling3D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [GlobalMaxPooling1D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="far fa-check-square" style="color:#008000"></i> [GlobalMaxPooling2D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="far fa-check-square" style="color:#008000"></i> [GlobalMaxPooling3D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="far fa-check-square" style="color:#008000"></i> [GlobalAveragePooling1D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="far fa-check-square" style="color:#008000"></i> [GlobalAveragePooling2D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="far fa-check-square" style="color:#008000"></i> [GlobalAveragePooling3D](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)

### [Locally-connected Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/local)
* <i class="far fa-check-square" style="color:#008000"></i> [LocallyConnected1D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/local/KerasLocallyConnected1D.java)
* <i class="far fa-check-square" style="color:#008000"></i> [LocallyConnected2D](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/local/KerasLocallyConnected2D.java)

### [Recurrent Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent)
* <i class="far fa-check-square" style="color:#008000"></i> [SimpleRNN](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent/KerasSimpleRnn.java)
* <i class="fas fa-times" style="color:#FF0000"></i> GRU
* <i class="far fa-check-square" style="color:#008000"></i> [LSTM](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent/KerasLstm.java)
* <i class="fas fa-times" style="color:#FF0000"></i> ConvLSTM2D


### [Embedding Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings)
* <i class="far fa-check-square" style="color:#008000"></i> [Embedding](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings/KerasEmbedding.java)

### [Merge Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
* <i class="far fa-check-square" style="color:#008000"></i> Add / add
* <i class="far fa-check-square" style="color:#008000"></i> Multiply / multiply
* <i class="far fa-check-square" style="color:#008000"></i> Subtract / subtract
* <i class="far fa-check-square" style="color:#008000"></i> Average / average
* <i class="far fa-check-square" style="color:#008000"></i> Maximum / maximum
* <i class="far fa-check-square" style="color:#008000"></i> Concatenate / concatenate
* <i class="fas fa-times" style="color:#FF0000"></i> Dot / dot


### [Advanced Activation Layers](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations)
* <i class="far fa-check-square" style="color:#008000"></i> [LeakyReLU](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasLeakyReLU.java)
* <i class="far fa-check-square" style="color:#008000"></i> [PReLU](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasPReLU.java)
* <i class="far fa-check-square" style="color:#008000"></i> ELU
* <i class="far fa-check-square" style="color:#008000"></i> [ThresholdedReLU](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasThresholdedReLU.java)

### [Normalization Layers](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization)
* <i class="far fa-check-square" style="color:#008000"></i> [BatchNormalization](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization/KerasBatchNormalization.java)

### Noise Layers
* <i class="far fa-check-square" style="color:#008000"></i> [GaussianNoise](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/noise/KerasGaussianNoise.java)
* <i class="far fa-check-square" style="color:#008000"></i> [GaussianDropout](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/noise/KerasGaussianDropout.java)
* <i class="far fa-check-square" style="color:#008000"></i> [AlphaDropout](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/noise/KerasAlphaDropout.java)

### Layer Wrappers
* <i class="fas fa-times" style="color:#FF0000"></i> TimeDistributed
* <i class="far fa-check-square" style="color:#008000"></i> [Bidirectional](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/wrappers/KerasBidirectional.java)

## <a name="losses">[Losses](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasLossUtils.java)</a>
* <i class="far fa-check-square" style="color:#008000"></i> mean_squared_error
* <i class="far fa-check-square" style="color:#008000"></i> mean_absolute_error
* <i class="far fa-check-square" style="color:#008000"></i> mean_absolute_percentage_error
* <i class="far fa-check-square" style="color:#008000"></i> mean_squared_logarithmic_error
* <i class="far fa-check-square" style="color:#008000"></i> squared_hinge
* <i class="far fa-check-square" style="color:#008000"></i> hinge
* <i class="far fa-check-square" style="color:#008000"></i> categorical_hinge
* <i class="fas fa-times" style="color:#FF0000"></i> logcosh
* <i class="far fa-check-square" style="color:#008000"></i> categorical_crossentropy
* <i class="far fa-check-square" style="color:#008000"></i> sparse_categorical_crossentropy
* <i class="far fa-check-square" style="color:#008000"></i> binary_crossentropy
* <i class="far fa-check-square" style="color:#008000"></i> kullback_leibler_divergence
* <i class="far fa-check-square" style="color:#008000"></i> poisson
* <i class="far fa-check-square" style="color:#008000"></i> cosine_proximity

## <a name="activations">[Activations](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasActivationUtils.java)</a>
* <i class="far fa-check-square" style="color:#008000"></i> softmax
* <i class="far fa-check-square" style="color:#008000"></i> elu
* <i class="far fa-check-square" style="color:#008000"></i> selu
* <i class="far fa-check-square" style="color:#008000"></i> softplus
* <i class="far fa-check-square" style="color:#008000"></i> softsign
* <i class="far fa-check-square" style="color:#008000"></i> relu
* <i class="far fa-check-square" style="color:#008000"></i> tanh
* <i class="far fa-check-square" style="color:#008000"></i> sigmoid
* <i class="far fa-check-square" style="color:#008000"></i> hard_sigmoid
* <i class="far fa-check-square" style="color:#008000"></i> linear

## <a name="initializers">[Initializers](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasInitilizationUtils.java)</a>
* <i class="far fa-check-square" style="color:#008000"></i> Zeros
* <i class="far fa-check-square" style="color:#008000"></i> Ones
* <i class="far fa-check-square" style="color:#008000"></i> Constant
* <i class="far fa-check-square" style="color:#008000"></i> RandomNormal
* <i class="far fa-check-square" style="color:#008000"></i> RandomUniform
* <i class="far fa-check-square" style="color:#008000"></i> TruncatedNormal
* <i class="far fa-check-square" style="color:#008000"></i> VarianceScaling
* <i class="far fa-check-square" style="color:#008000"></i> Orthogonal
* <i class="far fa-check-square" style="color:#008000"></i> Identity
* <i class="far fa-check-square" style="color:#008000"></i> lecun_uniform
* <i class="far fa-check-square" style="color:#008000"></i> lecun_normal
* <i class="far fa-check-square" style="color:#008000"></i> glorot_normal
* <i class="far fa-check-square" style="color:#008000"></i> glorot_uniform
* <i class="far fa-check-square" style="color:#008000"></i> he_normal
* <i class="far fa-check-square" style="color:#008000"></i> he_uniform

## <a name="regularizers">[Regularizers](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasRegularizerUtils.java)</a>
* <i class="far fa-check-square" style="color:#008000"></i> l1
* <i class="far fa-check-square" style="color:#008000"></i> l2
* <i class="far fa-check-square" style="color:#008000"></i> l1_l2

## <a name="constraints">[Constraints](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasConstraintUtils.java)</a>
* <i class="far fa-check-square" style="color:#008000"></i> max_norm
* <i class="far fa-check-square" style="color:#008000"></i> non_neg
* <i class="far fa-check-square" style="color:#008000"></i> unit_norm
* <i class="far fa-check-square" style="color:#008000"></i> min_max_norm

## <a name="optimizers">[Optimizers](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasOptimizerUtils.java)</a>
* <i class="far fa-check-square" style="color:#008000"></i> SGD
* <i class="far fa-check-square" style="color:#008000"></i> RMSprop
* <i class="far fa-check-square" style="color:#008000"></i> Adagrad
* <i class="far fa-check-square" style="color:#008000"></i> Adadelta
* <i class="far fa-check-square" style="color:#008000"></i> Adam
* <i class="far fa-check-square" style="color:#008000"></i> Adamax
* <i class="far fa-check-square" style="color:#008000"></i> Nadam
* <i class="fas fa-times" style="color:#FF0000"></i> TFOptimizer
