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
 * All relevant property fields of keras 1.x layers.
 *
 * @author Max Pumperla
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class Keras1LayerConfiguration extends KerasLayerConfiguration {

    /* Basic layer names */
    private final String LAYER_CLASS_NAME_CONVOLUTION_1D = "Convolution1D";
    private final String LAYER_CLASS_NAME_CONVOLUTION_2D = "Convolution2D";
    private final String LAYER_CLASS_NAME_CONVOLUTION_3D = "Convolution3D";

    private final String LAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_2D = "SeparableConvolution2D";
    private final String LAYER_CLASS_NAME_DECONVOLUTION_2D = "Deconvolution2D";

    /* Partially shared layer configurations. */
    private final String LAYER_FIELD_OUTPUT_DIM = "output_dim";
    private final String LAYER_FIELD_DROPOUT_RATE = "p";
    private final String LAYER_FIELD_USE_BIAS = "bias";
    private final String KERAS_PARAM_NAME_W = "W";
    private final String KERAS_PARAM_NAME_B = "b";
    private final String KERAS_PARAM_NAME_RW = "U";


    /* Keras dimension ordering for, e.g., convolutional layersOrdered. */
    private final String LAYER_FIELD_DIM_ORDERING = "dim_ordering";
    private final String DIM_ORDERING_THEANO = "th";
    private final String DIM_ORDERING_TENSORFLOW = "tf";

    /* Recurrent layers */
    private final String LAYER_FIELD_DROPOUT_W = "dropout_W";
    private final String LAYER_FIELD_DROPOUT_U = "dropout_U";
    private final String LAYER_FIELD_INNER_INIT = "inner_init";
    private final String LAYER_FIELD_INNER_ACTIVATION = "inner_activation";

    /* Embedding layer properties */
    private final String LAYER_FIELD_EMBEDDING_INIT = "init";
    private final String LAYER_FIELD_EMBEDDING_WEIGHTS = "W";
    private final String LAYER_FIELD_EMBEDDINGS_REGULARIZER = "W_regularizer";
    private final String LAYER_FIELD_EMBEDDINGS_CONSTRAINT = "W_constraint";

    /* Normalisation layers */
    private final String LAYER_FIELD_BATCHNORMALIZATION_BETA_INIT = "beta_init";
    private final String LAYER_FIELD_BATCHNORMALIZATION_GAMMA_INIT = "gamma_init";
    private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_MEAN = "running_mean";
    private final String LAYER_FIELD_BATCHNORMALIZATION_MOVING_VARIANCE = "running_std";

    /* Advanced activations */
    private final String LAYER_FIELD_PRELU_INIT = "init";

    /* Convolutional layer properties */
    private final String LAYER_FIELD_NB_FILTER = "nb_filter";
    private final String LAYER_FIELD_CONVOLUTION_STRIDES = "subsample";
    private final String LAYER_FIELD_FILTER_LENGTH = "filter_length";
    private final String LAYER_FIELD_SUBSAMPLE_LENGTH = "subsample_length";
    private final String LAYER_FIELD_DILATION_RATE = "atrous_rate";


    /* Pooling / Upsampling layer properties */
    private final String LAYER_FIELD_POOL_1D_SIZE = "pool_length";
    private final String LAYER_FIELD_POOL_1D_STRIDES = "stride";
    private final String LAYER_FIELD_UPSAMPLING_1D_SIZE = "length";

    /* Keras convolution border modes. */
    private final String LAYER_FIELD_BORDER_MODE = "border_mode";

    /* Noise layers */
    private final String LAYER_FIELD_GAUSSIAN_VARIANCE = "sigma";

    /* Keras weight regularizers. */
    private final String LAYER_FIELD_W_REGULARIZER = "W_regularizer";
    private final String LAYER_FIELD_B_REGULARIZER = "b_regularizer";

    /* Keras constraints */
    private final String LAYER_FIELD_CONSTRAINT_NAME = "name";
    private final String LAYER_FIELD_W_CONSTRAINT = "W_constraint";
    private final String LAYER_FIELD_B_CONSTRAINT = "b_constraint";
    private final String LAYER_FIELD_MAX_CONSTRAINT = "m";
    private final String LAYER_FIELD_MINMAX_MIN_CONSTRAINT = "low";
    private final String LAYER_FIELD_MINMAX_MAX_CONSTRAINT = "high";


    /* Keras weight initializers. */
    private final String LAYER_FIELD_INIT = "init";

}
