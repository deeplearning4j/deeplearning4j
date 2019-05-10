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
import lombok.EqualsAndHashCode;

/**
 * All relevant property fields of keras 2.x layers.
 *
 * @author Max Pumperla
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class Keras2LayerConfiguration extends KerasLayerConfiguration {

    /* Basic layer names */
    private final String LAYER_CLASS_NAME_CONVOLUTION_1D = "Conv1D";
    private final String LAYER_CLASS_NAME_CONVOLUTION_2D = "Conv2D";
    private final String LAYER_CLASS_NAME_CONVOLUTION_3D = "Conv3D";

    private final String LAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_2D = "SeparableConv2D";
    private final String LAYER_CLASS_NAME_DECONVOLUTION_2D = "Conv2DTranspose";

    /* Partially shared layer configurations. */
    private final String LAYER_FIELD_OUTPUT_DIM = "units";
    private final String LAYER_FIELD_DROPOUT_RATE = "rate";
    private final String LAYER_FIELD_USE_BIAS = "use_bias";
    private final String KERAS_PARAM_NAME_W = "kernel";
    private final String KERAS_PARAM_NAME_B = "bias";
    private final String KERAS_PARAM_NAME_RW = "recurrent_kernel";


    /* Keras dimension ordering for, e.g., convolutional layersOrdered. */
    private final String LAYER_FIELD_DIM_ORDERING = "data_format";
    private final String DIM_ORDERING_THEANO = "channels_first";
    private final String DIM_ORDERING_TENSORFLOW = "channels_last";

    /* Recurrent layers */
    private final String LAYER_FIELD_DROPOUT_W = "dropout";
    private final String LAYER_FIELD_DROPOUT_U = "recurrent_dropout";
    private final String LAYER_FIELD_INNER_INIT = "recurrent_initializer";
    private final String LAYER_FIELD_INNER_ACTIVATION = "recurrent_activation";

    /* Embedding layer properties */
    private final String LAYER_FIELD_EMBEDDING_INIT = "embeddings_initializer";
    private final String LAYER_FIELD_EMBEDDING_WEIGHTS = "embeddings";
    private final String LAYER_FIELD_EMBEDDINGS_REGULARIZER = "embeddings_regularizer";
    private final String LAYER_FIELD_EMBEDDINGS_CONSTRAINT = "embeddings_constraint";

    /* Normalisation layers */
    private final String LAYER_FIELD_BATCHNORMALIZATION_BETA_INIT = "beta_initializer";
    private final String LAYER_FIELD_BATCHNORMALIZATION_GAMMA_INIT = "gamma_initializer";
    private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN = "moving_mean";
    private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE = "moving_variance";

    /* Advanced activations */
    private final String LAYER_FIELD_PRELU_INIT = "alpha_initializer";

    /* Convolutional layer properties */
    private final String LAYER_FIELD_NB_FILTER = "filters";
    private final String LAYER_FIELD_CONVOLUTION_STRIDES = "strides";
    private final String LAYER_FIELD_FILTER_LENGTH = "kernel_size";
    private final String LAYER_FIELD_SUBSAMPLE_LENGTH = "strides";
    private final String LAYER_FIELD_DILATION_RATE = "dilation_rate";

    /* Pooling / Upsampling layer properties */
    private final String LAYER_FIELD_POOL_1D_SIZE = "pool_size";
    private final String LAYER_FIELD_POOL_1D_STRIDES = "strides";
    private final String LAYER_FIELD_UPSAMPLING_1D_SIZE = "size";

    /* Keras convolution border modes. */
    private final String LAYER_FIELD_BORDER_MODE = "padding";

    /* Noise layers */
    private final String LAYER_FIELD_GAUSSIAN_VARIANCE = "stddev";

    /* Keras weight regularizers. */
    private final String LAYER_FIELD_W_REGULARIZER = "kernel_regularizer";
    private final String LAYER_FIELD_B_REGULARIZER = "bias_regularizer";

    /* Keras constraints */
    private final String LAYER_FIELD_CONSTRAINT_NAME = "class_name";
    private final String LAYER_FIELD_W_CONSTRAINT = "kernel_constraint";
    private final String LAYER_FIELD_B_CONSTRAINT = "bias_constraint";
    private final String LAYER_FIELD_MAX_CONSTRAINT = "max_value";
    private final String LAYER_FIELD_MINMAX_MIN_CONSTRAINT = "min_value";
    private final String LAYER_FIELD_MINMAX_MAX_CONSTRAINT = "max_value";

    /* Keras weight initializers. */
    private final String LAYER_FIELD_INIT = "kernel_initializer";
}