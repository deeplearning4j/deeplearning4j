/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.modelimport.keras.config;

import lombok.Data;


/**
 * All relevant property fields of keras layers.
 * <p>
 * Empty String fields mean Keras 1 and 2 implementations differ,
 * supplied fields stand for shared properties.
 *
 * @author Max Pumperla
 */
@Data
public class KerasLayerConfiguration {

    private final String LAYER_FIELD_KERAS_VERSION = "keras_version";
    private final String LAYER_FIELD_CLASS_NAME = "class_name";
    private final String LAYER_FIELD_LAYER = "layer";

    /* Basic layer names */
    // Missing Layers: ActivityRegularization, Masking
    // Conv3DTranspose, SeparableConv1D,
    // ConvRNN2D, ConvLSTM2D
    private final String LAYER_CLASS_NAME_ACTIVATION = "Activation";
    private final String LAYER_CLASS_NAME_INPUT = "InputLayer";
    private final String LAYER_CLASS_NAME_PERMUTE = "Permute";
    private final String LAYER_CLASS_NAME_DROPOUT = "Dropout";
    private final String LAYER_CLASS_NAME_REPEAT = "RepeatVector";
    private final String LAYER_CLASS_NAME_LAMBDA = "Lambda";


    private final String LAYER_CLASS_NAME_SPATIAL_DROPOUT_1D = "SpatialDropout1D";
    private final String LAYER_CLASS_NAME_SPATIAL_DROPOUT_2D = "SpatialDropout2D";
    private final String LAYER_CLASS_NAME_SPATIAL_DROPOUT_3D = "SpatialDropout3D";
    private final String LAYER_CLASS_NAME_ALPHA_DROPOUT = "AlphaDropout";
    private final String LAYER_CLASS_NAME_GAUSSIAN_DROPOUT = "GaussianDropout";
    private final String LAYER_CLASS_NAME_GAUSSIAN_NOISE = "GaussianNoise";
    private final String LAYER_CLASS_NAME_DENSE = "Dense";

    private final String LAYER_CLASS_NAME_LSTM = "LSTM";
    private final String LAYER_CLASS_NAME_SIMPLE_RNN = "SimpleRNN";

    private final String LAYER_CLASS_NAME_BIDIRECTIONAL = "Bidirectional";
    private final String LAYER_CLASS_NAME_TIME_DISTRIBUTED = "TimeDistributed";


    private final String LAYER_CLASS_NAME_MAX_POOLING_1D = "MaxPooling1D";
    private final String LAYER_CLASS_NAME_MAX_POOLING_2D = "MaxPooling2D";
    private final String LAYER_CLASS_NAME_MAX_POOLING_3D = "MaxPooling3D";
    private final String LAYER_CLASS_NAME_AVERAGE_POOLING_1D = "AveragePooling1D";
    private final String LAYER_CLASS_NAME_AVERAGE_POOLING_2D = "AveragePooling2D";
    private final String LAYER_CLASS_NAME_AVERAGE_POOLING_3D = "AveragePooling3D";
    private final String LAYER_CLASS_NAME_ZERO_PADDING_1D = "ZeroPadding1D";
    private final String LAYER_CLASS_NAME_ZERO_PADDING_2D = "ZeroPadding2D";
    private final String LAYER_CLASS_NAME_ZERO_PADDING_3D = "ZeroPadding3D";
    private final String LAYER_CLASS_NAME_CROPPING_1D = "Cropping1D";
    private final String LAYER_CLASS_NAME_CROPPING_2D = "Cropping2D";
    private final String LAYER_CLASS_NAME_CROPPING_3D = "Cropping3D";


    private final String LAYER_CLASS_NAME_FLATTEN = "Flatten";
    private final String LAYER_CLASS_NAME_RESHAPE = "Reshape";
    private final String LAYER_CLASS_NAME_MERGE = "Merge";
    private final String LAYER_CLASS_NAME_ADD = "Add";
    private final String LAYER_CLASS_NAME_FUNCTIONAL_ADD = "add";
    private final String LAYER_CLASS_NAME_SUBTRACT = "Subtract";
    private final String LAYER_CLASS_NAME_FUNCTIONAL_SUBTRACT = "subtract";
    private final String LAYER_CLASS_NAME_MULTIPLY = "Multiply";
    private final String LAYER_CLASS_NAME_FUNCTIONAL_MULTIPLY = "multiply";
    private final String LAYER_CLASS_NAME_AVERAGE = "Average";
    private final String LAYER_CLASS_NAME_FUNCTIONAL_AVERAGE = "average";
    private final String LAYER_CLASS_NAME_MAXIMUM = "Maximum";
    private final String LAYER_CLASS_NAME_FUNCTIONAL_MAXIMUM = "maximum";
    private final String LAYER_CLASS_NAME_CONCATENATE = "Concatenate";
    private final String LAYER_CLASS_NAME_FUNCTIONAL_CONCATENATE = "concatenate";
    private final String LAYER_CLASS_NAME_DOT = "Dot";
    private final String LAYER_CLASS_NAME_FUNCTIONAL_DOT = "dot";


    private final String LAYER_CLASS_NAME_BATCHNORMALIZATION = "BatchNormalization";
    private final String LAYER_CLASS_NAME_EMBEDDING = "Embedding";
    private final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D = "GlobalMaxPooling1D";
    private final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D = "GlobalMaxPooling2D";
    private final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_3D = "GlobalMaxPooling3D";
    private final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D = "GlobalAveragePooling1D";
    private final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D = "GlobalAveragePooling2D";
    private final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_3D = "GlobalAveragePooling3D";
    private final String LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE = "TimeDistributedDense"; // Keras 1 only
    private final String LAYER_CLASS_NAME_ATROUS_CONVOLUTION_1D = "AtrousConvolution1D"; // Keras 1 only
    private final String LAYER_CLASS_NAME_ATROUS_CONVOLUTION_2D = "AtrousConvolution2D"; // Keras 1 only
    private final String LAYER_CLASS_NAME_CONVOLUTION_1D = ""; // 1: Convolution1D, 2: Conv1D
    private final String LAYER_CLASS_NAME_CONVOLUTION_2D = ""; // 1: Convolution2D, 2: Conv2D
    private final String LAYER_CLASS_NAME_CONVOLUTION_3D = ""; // 1: Convolution2D, 2: Conv2D
    private final String LAYER_CLASS_NAME_LEAKY_RELU = "LeakyReLU";
    private final String LAYER_CLASS_NAME_PRELU = "PReLU";
    private final String LAYER_CLASS_NAME_THRESHOLDED_RELU = "ThresholdedReLU";
    private final String LAYER_CLASS_NAME_UPSAMPLING_1D = "UpSampling1D";
    private final String LAYER_CLASS_NAME_UPSAMPLING_2D = "UpSampling2D";
    private final String LAYER_CLASS_NAME_UPSAMPLING_3D = "UpSampling3D";
    private final String LAYER_CLASS_NAME_DEPTHWISE_CONVOLUTION_2D = "DepthwiseConv2D"; // Keras 2 only
    private final String LAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_1D = "SeparableConv1D"; // Keras 2 only
    private final String LAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_2D = ""; // 1: SeparableConvolution2D, 2: SeparableConv2D
    private final String LAYER_CLASS_NAME_DECONVOLUTION_2D = ""; // 1: Deconvolution2D, 2: Conv2DTranspose
    private final String LAYER_CLASS_NAME_DECONVOLUTION_3D = "Conv2DTranspose"; // Keras 2 only

    // Locally connected layers
    private final String LAYER_CLASS_NAME_LOCALLY_CONNECTED_2D = "LocallyConnected2D";
    private final String LAYER_CLASS_NAME_LOCALLY_CONNECTED_1D = "LocallyConnected1D";


    /* Partially shared layer configurations. */
    private final String LAYER_FIELD_INPUT_SHAPE = "input_shape";
    private final String LAYER_FIELD_CONFIG = "config";
    private final String LAYER_FIELD_NAME = "name";
    private final String LAYER_FIELD_BATCH_INPUT_SHAPE = "batch_input_shape";
    private final String LAYER_FIELD_INBOUND_NODES = "inbound_nodes";
    private final String LAYER_FIELD_OUTBOUND_NODES = "outbound_nodes";
    private final String LAYER_FIELD_DROPOUT = "dropout";
    private final String LAYER_FIELD_ACTIVITY_REGULARIZER = "activity_regularizer";
    private final String LAYER_FIELD_EMBEDDING_OUTPUT_DIM = "output_dim";
    private final String LAYER_FIELD_OUTPUT_DIM = ""; // 1: output_dim, 2: units
    private final String LAYER_FIELD_DROPOUT_RATE = ""; // 1: p, 2: rate
    private final String LAYER_FIELD_USE_BIAS = ""; // 1: bias, 2: use_bias
    private final String KERAS_PARAM_NAME_W = ""; // 1: W, 2: kernel
    private final String KERAS_PARAM_NAME_B = ""; // 1: b, 2: bias
    private final String KERAS_PARAM_NAME_RW = ""; // 1: U, 2: recurrent_kernel

    /* Utils */
    private final String LAYER_FIELD_REPEAT_MULTIPLIER = "n";

    /* Keras dimension ordering for, e.g., convolutional layersOrdered. */
    private final String LAYER_FIELD_BACKEND = "backend"; // not available in keras 1, caught in code
    private final String LAYER_FIELD_DIM_ORDERING = ""; // 1: dim_ordering, 2: data_format
    private final String DIM_ORDERING_THEANO = ""; // 1: th, 2: channels_first
    private final String DIM_ORDERING_TENSORFLOW = ""; // 1: tf, 2: channels_last

    /* Recurrent layers */
    private final String LAYER_FIELD_DROPOUT_W = ""; // 1: dropout_W, 2: dropout
    private final String LAYER_FIELD_DROPOUT_U = ""; // 2: dropout_U, 2: recurrent_dropout
    private final String LAYER_FIELD_INNER_INIT = ""; // 1: inner_init, 2: recurrent_initializer
    private final String LAYER_FIELD_RECURRENT_CONSTRAINT = "recurrent_constraint"; // keras 2 only
    private final String LAYER_FIELD_RECURRENT_DROPOUT = ""; // 1: dropout_U, 2: recurrent_dropout
    private final String LAYER_FIELD_INNER_ACTIVATION = ""; // 1: inner_activation, 2: recurrent_activation
    private final String LAYER_FIELD_FORGET_BIAS_INIT = "forget_bias_init"; // keras 1 only: string
    private final String LAYER_FIELD_UNIT_FORGET_BIAS = "unit_forget_bias";
    private final String LAYER_FIELD_RETURN_SEQUENCES = "return_sequences";
    private final String LAYER_FIELD_UNROLL = "unroll";

    /* Embedding layer properties */
    private final String LAYER_FIELD_INPUT_DIM = "input_dim";
    private final String LAYER_FIELD_EMBEDDING_INIT = ""; // 1: "init", 2: "embeddings_initializer"
    private final String LAYER_FIELD_EMBEDDING_WEIGHTS = ""; // 1: "W", 2: "embeddings"
    private final String LAYER_FIELD_EMBEDDINGS_REGULARIZER = ""; // 1: W_regularizer, 2: embeddings_regularizer
    private final String LAYER_FIELD_EMBEDDINGS_CONSTRAINT = ""; // 1: W_constraint, 2: embeddings_constraint
    private final String LAYER_FIELD_MASK_ZERO = "mask_zero";
    private final String LAYER_FIELD_INPUT_LENGTH = "input_length";

    /* Keras separable convolution types */
    private final String LAYER_PARAM_NAME_DEPTH_WISE_KERNEL = "depthwise_kernel";
    private final String LAYER_PARAM_NAME_POINT_WISE_KERNEL = "pointwise_kernel";
    private final String LAYER_FIELD_DEPTH_MULTIPLIER = "depth_multiplier";


    private final String LAYER_FIELD_DEPTH_WISE_INIT = "depthwise_initializer";
    private final String LAYER_FIELD_POINT_WISE_INIT = "pointwise_initializer";

    private final String LAYER_FIELD_DEPTH_WISE_REGULARIZER = "depthwise_regularizer";
    private final String LAYER_FIELD_POINT_WISE_REGULARIZER = "pointwise_regularizer";

    private final String LAYER_FIELD_DEPTH_WISE_CONSTRAINT = "depthwise_constraint";
    private final String LAYER_FIELD_POINT_WISE_CONSTRAINT = "pointwise_constraint";

    /* Normalisation layers */
    // Missing: keras 2 moving_mean_initializer, moving_variance_initializer
    private final String LAYER_FIELD_BATCHNORMALIZATION_MODE = "mode"; // keras 1 only
    private final String LAYER_FIELD_BATCHNORMALIZATION_BETA_INIT = ""; // 1: beta_init, 2: beta_initializer
    private final String LAYER_FIELD_BATCHNORMALIZATION_GAMMA_INIT = ""; // 1: gamma_init, 2: gamma_initializer
    private final String LAYER_FIELD_BATCHNORMALIZATION_BETA_CONSTRAINT = "beta_constraint"; // keras 2 only
    private final String LAYER_FIELD_BATCHNORMALIZATION_GAMMA_CONSTRAINT = "gamma_constraint"; // keras 2 only
    private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN = ""; // 1: running_mean, 2: moving_mean
    private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE = ""; // 1: running_std, 2: moving_variance

    /* Advanced activations */
    // Missing: LeakyReLU, PReLU, ThresholdedReLU, ParametricSoftplus, SReLu
    private final String LAYER_FIELD_PRELU_INIT = ""; // 1: init, 2: alpha_initializer

    /* Convolutional layer properties */
    private final String LAYER_FIELD_NB_FILTER = ""; // 1: nb_filter, 2: filters
    private final String LAYER_FIELD_NB_ROW = "nb_row"; // keras 1 only
    private final String LAYER_FIELD_NB_COL = "nb_col"; // keras 1 only
    private final String LAYER_FIELD_KERNEL_SIZE = "kernel_size"; // keras 2 only
    private final String LAYER_FIELD_POOL_SIZE = "pool_size";
    private final String LAYER_FIELD_CONVOLUTION_STRIDES = ""; // 1: subsample, 2: strides
    private final String LAYER_FIELD_FILTER_LENGTH = ""; // 1: filter_length, 2: kernel_size
    private final String LAYER_FIELD_SUBSAMPLE_LENGTH = ""; // 1: subsample_length, 2: strides
    private final String LAYER_FIELD_DILATION_RATE = ""; // 1: atrous_rate, 2: dilation_rate
    private final String LAYER_FIELD_ZERO_PADDING = "padding";
    private final String LAYER_FIELD_CROPPING = "cropping";
    private final String LAYER_FIELD_3D_KERNEL_1 = "kernel_dim1"; // keras 1 only
    private final String LAYER_FIELD_3D_KERNEL_2 = "kernel_dim2"; // keras 1 only
    private final String LAYER_FIELD_3D_KERNEL_3 = "kernel_dim3"; // keras 1 only


    /* Pooling / Upsampling layer properties */
    private final String LAYER_FIELD_POOL_STRIDES = "strides";
    private final String LAYER_FIELD_POOL_1D_SIZE = ""; // 1: pool_length, 2: pool_size
    private final String LAYER_FIELD_POOL_1D_STRIDES = ""; // 1: stride, 2: strides
    private final String LAYER_FIELD_UPSAMPLING_1D_SIZE = ""; // 1: length, 2: size
    private final String LAYER_FIELD_UPSAMPLING_2D_SIZE = "size";
    private final String LAYER_FIELD_UPSAMPLING_3D_SIZE = "size";


    /* Keras convolution border modes. */
    private final String LAYER_FIELD_BORDER_MODE = ""; // 1: border_mode, 2: padding
    private final String LAYER_BORDER_MODE_SAME = "same";
    private final String LAYER_BORDER_MODE_VALID = "valid";
    private final String LAYER_BORDER_MODE_FULL = "full";

    /* Noise layers */
    private final String LAYER_FIELD_RATE = "rate";
    private final String LAYER_FIELD_GAUSSIAN_VARIANCE = ""; // 1: sigma, 2: stddev

    /* Layer wrappers */
    // Missing: TimeDistributed


    /* Keras weight regularizers. */
    private final String LAYER_FIELD_W_REGULARIZER = ""; // 1: W_regularizer, 2: kernel_regularizer
    private final String LAYER_FIELD_B_REGULARIZER = ""; // 1: b_regularizer, 2: bias_regularizer
    private final String REGULARIZATION_TYPE_L1 = "l1";
    private final String REGULARIZATION_TYPE_L2 = "l2";

    /* Keras constraints */
    private final String LAYER_FIELD_MINMAX_NORM_CONSTRAINT = "MinMaxNorm";
    private final String LAYER_FIELD_MINMAX_NORM_CONSTRAINT_ALIAS = "min_max_norm";
    private final String LAYER_FIELD_MAX_NORM_CONSTRAINT = "MaxNorm";
    private final String LAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS = "max_norm";
    private final String LAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS_2 = "maxnorm";
    private final String LAYER_FIELD_NON_NEG_CONSTRAINT = "NonNeg";
    private final String LAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS = "nonneg";
    private final String LAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS_2 = "non_neg";
    private final String LAYER_FIELD_UNIT_NORM_CONSTRAINT = "UnitNorm";
    private final String LAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS = "unitnorm";
    private final String LAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS_2 = "unit_norm";
    private final String LAYER_FIELD_CONSTRAINT_NAME = ""; // 1: name, 2: class_name
    private final String LAYER_FIELD_W_CONSTRAINT = ""; // 1: W_constraint, 2: kernel_constraint
    private final String LAYER_FIELD_B_CONSTRAINT = ""; // 1: b_constraint, 2: bias_constraint
    private final String LAYER_FIELD_MAX_CONSTRAINT = ""; // 1: m, 2: max_value
    private final String LAYER_FIELD_MINMAX_MIN_CONSTRAINT = ""; // 1: low, 2: min_value
    private final String LAYER_FIELD_MINMAX_MAX_CONSTRAINT = ""; // 1: high, 2: max_value
    private final String LAYER_FIELD_CONSTRAINT_DIM = "axis";
    private final String LAYER_FIELD_CONSTRAINT_RATE = "rate";


    /* Keras weight initializers. */
    private final String LAYER_FIELD_INIT = ""; // 1: init, 2: kernel_initializer
    private final String LAYER_FIELD_BIAS_INIT = "bias_initializer"; // keras 2 only
    private final String LAYER_FIELD_INIT_MEAN = "mean";
    private final String LAYER_FIELD_INIT_STDDEV = "stddev";
    private final String LAYER_FIELD_INIT_SCALE = "scale";
    private final String LAYER_FIELD_INIT_MINVAL = "minval";
    private final String LAYER_FIELD_INIT_MAXVAL = "maxval";
    private final String LAYER_FIELD_INIT_VALUE = "value";
    private final String LAYER_FIELD_INIT_GAIN = "gain";
    private final String LAYER_FIELD_INIT_MODE = "mode";
    private final String LAYER_FIELD_INIT_DISTRIBUTION = "distribution";

    private final String INIT_UNIFORM = "uniform";
    private final String INIT_RANDOM_UNIFORM = "random_uniform";
    private final String INIT_RANDOM_UNIFORM_ALIAS = "RandomUniform";
    private final String INIT_ZERO = "zero";
    private final String INIT_ZEROS = "zeros";
    private final String INIT_ZEROS_ALIAS = "Zeros";
    private final String INIT_ONE = "one";
    private final String INIT_ONES = "ones";
    private final String INIT_ONES_ALIAS = "Ones";
    private final String INIT_CONSTANT = "constant";
    private final String INIT_CONSTANT_ALIAS = "Constant";
    private final String INIT_TRUNCATED_NORMAL = "truncated_normal";
    private final String INIT_TRUNCATED_NORMAL_ALIAS = "TruncatedNormal";
    private final String INIT_GLOROT_NORMAL = "glorot_normal";
    private final String INIT_GLOROT_UNIFORM = "glorot_uniform";
    private final String INIT_HE_NORMAL = "he_normal";
    private final String INIT_HE_UNIFORM = "he_uniform";
    private final String INIT_LECUN_UNIFORM = "lecun_uniform";
    private final String INIT_LECUN_NORMAL = "lecun_normal";
    private final String INIT_NORMAL = "normal";
    private final String INIT_RANDOM_NORMAL = "random_normal";
    private final String INIT_RANDOM_NORMAL_ALIAS = "RandomNormal";
    private final String INIT_ORTHOGONAL = "orthogonal";
    private final String INIT_ORTHOGONAL_ALIAS = "Orthogonal";
    private final String INIT_IDENTITY = "identity";
    private final String INIT_IDENTITY_ALIAS = "Identity";
    private final String INIT_VARIANCE_SCALING = "VarianceScaling"; // keras 2 only


    /* Keras and DL4J activation types. */
    private final String LAYER_FIELD_ACTIVATION = "activation";

    private final String KERAS_ACTIVATION_SOFTMAX = "softmax";
    private final String KERAS_ACTIVATION_SOFTPLUS = "softplus";
    private final String KERAS_ACTIVATION_SOFTSIGN = "softsign";
    private final String KERAS_ACTIVATION_RELU = "relu";
    private final String KERAS_ACTIVATION_RELU6 = "relu6";
    private final String KERAS_ACTIVATION_TANH = "tanh";
    private final String KERAS_ACTIVATION_SIGMOID = "sigmoid";
    private final String KERAS_ACTIVATION_HARD_SIGMOID = "hard_sigmoid";
    private final String KERAS_ACTIVATION_LINEAR = "linear";
    private final String KERAS_ACTIVATION_ELU = "elu"; // keras 2 only
    private final String KERAS_ACTIVATION_SELU = "selu"; // keras 2 only

    /* Keras loss functions. */
    private final String KERAS_LOSS_MEAN_SQUARED_ERROR = "mean_squared_error";
    private final String KERAS_LOSS_MSE = "mse";
    private final String KERAS_LOSS_MEAN_ABSOLUTE_ERROR = "mean_absolute_error";
    private final String KERAS_LOSS_MAE = "mae";
    private final String KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error";
    private final String KERAS_LOSS_MAPE = "mape";
    private final String KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR = "mean_squared_logarithmic_error";
    private final String KERAS_LOSS_MSLE = "msle";
    private final String KERAS_LOSS_SQUARED_HINGE = "squared_hinge";
    private final String KERAS_LOSS_HINGE = "hinge";
    private final String KERAS_LOSS_CATEGORICAL_HINGE = "categorical_hinge"; // keras 2 only
    private final String KERAS_LOSS_BINARY_CROSSENTROPY = "binary_crossentropy";
    private final String KERAS_LOSS_CATEGORICAL_CROSSENTROPY = "categorical_crossentropy";
    private final String KERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy";
    private final String KERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE = "kullback_leibler_divergence";
    private final String KERAS_LOSS_KLD = "kld";
    private final String KERAS_LOSS_POISSON = "poisson";
    private final String KERAS_LOSS_COSINE_PROXIMITY = "cosine_proximity";
    private final String KERAS_LOSS_LOG_COSH = "logcosh"; // keras 2 only

}