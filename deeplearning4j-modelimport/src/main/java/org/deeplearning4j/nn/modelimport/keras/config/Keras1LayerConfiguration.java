/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.modelimport.keras.config;


import lombok.Data;

/**
 * All relevant property fields of keras 1.x layers.
 *
 * @author Max Pumperla
 */
@Data
public class Keras1LayerConfiguration extends KerasLayerConfiguration {

    private final String LAYER_FIELD_CLASS_NAME = "class_name";
    private final String LAYER_CLASS_NAME_ACTIVATION = "Activation";
    private final String LAYER_CLASS_NAME_INPUT = "InputLayer";
    private final String LAYER_CLASS_NAME_DROPOUT = "Dropout";
    private final String LAYER_CLASS_NAME_DENSE = "Dense";
    private final String LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE = "TimeDistributedDense";
    private final String LAYER_CLASS_NAME_LSTM = "LSTM";
    private final String LAYER_CLASS_NAME_CONVOLUTION_1D = "Convolution1D";
    private final String LAYER_CLASS_NAME_CONVOLUTION_2D = "Convolution2D";
    private final String LAYER_CLASS_NAME_MAX_POOLING_1D = "MaxPooling1D";
    private final String LAYER_CLASS_NAME_MAX_POOLING_2D = "MaxPooling2D";
    private final String LAYER_CLASS_NAME_AVERAGE_POOLING_1D = "AveragePooling1D";
    private final String LAYER_CLASS_NAME_AVERAGE_POOLING_2D = "AveragePooling2D";
    private final String LAYER_CLASS_NAME_ZERO_PADDING_1D = "ZeroPadding1D";
    private final String LAYER_CLASS_NAME_ZERO_PADDING_2D = "ZeroPadding2D";
    private final String LAYER_CLASS_NAME_FLATTEN = "Flatten";
    private final String LAYER_CLASS_NAME_MERGE = "Merge";
    private final String LAYER_CLASS_NAME_BATCHNORMALIZATION = "BatchNormalization";
    private final String LAYER_CLASS_NAME_TIME_DISTRIBUTED = "TimeDistributed";
    private final String LAYER_CLASS_NAME_EMBEDDING = "Embedding";
    private final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D = "GlobalMaxPooling1D";
    private final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D = "GlobalMaxPooling2D";
    private final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D = "GlobalAveragePooling1D";
    private final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D = "GlobalAveragePooling2D";
}
