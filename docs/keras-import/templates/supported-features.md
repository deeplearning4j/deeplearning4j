# Keras Model Import: Supported Features

Little-known fact: Deeplearning4j's creator, Skymind, has two of the top 
five [Keras contributors](https://github.com/keras-team/keras/graphs/contributors) 
on our team, making it the largest contributor to Keras after Keras creator Francois 
Chollet, who's at Google.

While not every concept in DL4J has an equivalent in Keras and vice versa, many of the 
key concepts can be matched. Importing keras models into DL4J is done in 
our [deeplearning4j-modelimport](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras) 
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
Mapping keras to DL4J layers is done in the [layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers) sub-module of model import. The structure of this project loosely reflects the structure of Keras.

### [Core Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core)
* <i class="fa fa-check-square-o"></i> [Dense](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDense.java)
* <i class="fa fa-check-square-o"></i> [Activation](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasActivation.java)
* <i class="fa fa-check-square-o"></i> [Dropout](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasDropout.java)
* <i class="fa fa-check-square-o"></i> [Flatten](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasFlatten.java)
* <i class="fa fa-check-square-o"></i> [Reshape](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasReshape.java)
* <i class="fa fa-check-square-o"></i> [Merge](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
* <i class="fa fa-check-square-o"></i> [Permute](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasPermute.java)
* <i class="fa fa-check-square-o"></i> [RepeatVector](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasRepeatVector.java)
* <i class="fa fa-check-square-o"></i> [Lambda](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasLambda.java)
* <i class="fa fa-square-o"></i> ActivityRegularization
* <i class="fa fa-square-o"></i> Masking
* <i class="fa fa-check-square-o"></i> [SpatialDropout1D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasSpatialDropout.java)
* <i class="fa fa-check-square-o"></i> [SpatialDropout2D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasSpatialDropout.java)
* <i class="fa fa-check-square-o"></i> [SpatialDropout3D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasSpatialDropout.java)

### [Convolutional Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional)
* <i class="fa fa-check-square-o"></i> [Conv1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution1D.java)
* <i class="fa fa-check-square-o"></i> [Conv2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution2D.java)
* <i class="fa fa-check-square-o"></i> [Conv3D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasConvolution3D.java)
* <i class="fa fa-check-square-o"></i> [AtrousConvolution1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
* <i class="fa fa-check-square-o"></i> [AtrousConvolution2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasAtrousConvolution1D.java)
* <i class="fa fa-square-o"></i> SeparableConv1D
* <i class="fa fa-check-square-o"></i> [SeparableConv2D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasSeparableConvolution2D.java)
* <i class="fa fa-check-square-o"></i> [Conv2DTranspose](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasDeconvolution2D.java)
* <i class="fa fa-square-o"></i> Conv3DTranspose
* <i class="fa fa-check-square-o"></i> [Cropping1D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasCropping1D.java)
* <i class="fa fa-check-square-o"></i> [Cropping2D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasCropping2D.java)
* <i class="fa fa-check-square-o"></i> [Cropping3D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasCropping3D.java)
* <i class="fa fa-check-square-o"></i> [UpSampling1D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling1D.java)
* <i class="fa fa-check-square-o"></i> [UpSampling2D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling2D.java)
* <i class="fa fa-check-square-o"></i> [UpSampling3D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasUpsampling2D.java)
* <i class="fa fa-check-square-o"></i> [ZeroPadding1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding1D.java)
* <i class="fa fa-check-square-o"></i> [ZeroPadding2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding2D.java)
* <i class="fa fa-check-square-o"></i> [ZeroPadding3D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/convolutional/KerasZeroPadding3D.java)

### [Pooling Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling)
* <i class="fa fa-check-square-o"></i> [MaxPooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
* <i class="fa fa-check-square-o"></i> [MaxPooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
* <i class="fa fa-check-square-o"></i> [MaxPooling3D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling3D.java)
* <i class="fa fa-check-square-o"></i> [AveragePooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling1D.java)
* <i class="fa fa-check-square-o"></i> [AveragePooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling2D.java)
* <i class="fa fa-check-square-o"></i> [AveragePooling3D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasPooling3D.java)
* <i class="fa fa-check-square-o"></i> [GlobalMaxPooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalMaxPooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalMaxPooling3D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalAveragePooling1D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalAveragePooling2D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)
* <i class="fa fa-check-square-o"></i> [GlobalAveragePooling3D](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/pooling/KerasGlobalPooling.java)

### [Locally-connected Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/local)
* <i class="fa fa-check-square-o"></i> [LocallyConnected1D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/local/KerasLocallyConnected1D.java)
* <i class="fa fa-check-square-o"></i> [LocallyConnected2D](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/local/KerasLocallyConnected2D.java)

### [Recurrent Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent)
* <i class="fa fa-check-square-o"></i> [SimpleRNN](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent/KerasSimpleRnn.java)
* <i class="fa fa-square-o"></i> GRU
* <i class="fa fa-check-square-o"></i> [LSTM](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/recurrent/KerasLstm.java)
* <i class="fa fa-square-o"></i> ConvLSTM2D


### [Embedding Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings)
* <i class="fa fa-check-square-o"></i> [Embedding](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/embeddings/KerasEmbedding.java)

### [Merge Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/core/KerasMerge.java)
* <i class="fa fa-check-square-o"></i> Add / add
* <i class="fa fa-check-square-o"></i> Multiply / multiply
* <i class="fa fa-check-square-o"></i> Subtract / subtract
* <i class="fa fa-check-square-o"></i> Average / average
* <i class="fa fa-check-square-o"></i> Maximum / maximum
* <i class="fa fa-check-square-o"></i> Concatenate / concatenate
* <i class="fa fa-square-o"></i> Dot / dot


### [Advanced Activation Layers](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations)
* <i class="fa fa-check-square-o"></i> [LeakyReLU](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasLeakyReLU.java)
* <i class="fa fa-check-square-o"></i> [PReLU](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasPReLU.java)
* <i class="fa fa-check-square-o"></i> ELU
* <i class="fa fa-check-square-o"></i> [ThresholdedReLU](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/advanced/activations/KerasThresholdedReLU.java)

### [Normalization Layers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization)
* <i class="fa fa-check-square-o"></i> [BatchNormalization](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/normalization/KerasBatchNormalization.java)

### Noise Layers
* <i class="fa fa-check-square-o"></i> [GaussianNoise](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/noise/KerasGaussianNoise.java)
* <i class="fa fa-check-square-o"></i> [GaussianDropout](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/noise/KerasGaussianDropout.java)
* <i class="fa fa-check-square-o"></i> [AlphaDropout](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/noise/KerasAlphaDropout.java)

### Layer Wrappers
* <i class="fa fa-square-o"></i> TimeDistributed
* <i class="fa fa-check-square-o"></i> [Bidirectional](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/layers/wrappers/KerasBidirectional.java)

## <a name="losses">[Losses](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasLossUtils.java)</a>
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

## <a name="activations">[Activations](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasActivationUtils.java)</a>
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

## <a name="initializers">[Initializers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasInitilizationUtils.java)</a>
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

## <a name="regularizers">[Regularizers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasRegularizerUtils.java)</a>
* <i class="fa fa-check-square-o"></i> l1
* <i class="fa fa-check-square-o"></i> l2
* <i class="fa fa-check-square-o"></i> l1_l2

## <a name="constraints">[Constraints](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasConstraintUtils.java)</a>
* <i class="fa fa-check-square-o"></i> max_norm
* <i class="fa fa-check-square-o"></i> non_neg
* <i class="fa fa-check-square-o"></i> unit_norm
* <i class="fa fa-check-square-o"></i> min_max_norm

## <a name="optimizers">[Optimizers](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/utils/KerasOptimizerUtils.java)</a>
* <i class="fa fa-check-square-o"></i> SGD
* <i class="fa fa-check-square-o"></i> RMSprop
* <i class="fa fa-check-square-o"></i> Adagrad
* <i class="fa fa-check-square-o"></i> Adadelta
* <i class="fa fa-check-square-o"></i> Adam
* <i class="fa fa-check-square-o"></i> Adamax
* <i class="fa fa-check-square-o"></i> Nadam
* <i class="fa fa-square-o"></i> TFOptimizer