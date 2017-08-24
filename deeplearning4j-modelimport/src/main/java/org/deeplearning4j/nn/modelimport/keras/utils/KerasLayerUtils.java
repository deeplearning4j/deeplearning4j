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
package org.deeplearning4j.nn.modelimport.keras.utils;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.*;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasLeakyReLU;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolution1D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolution2D;

import java.lang.reflect.Constructor;
import java.util.Map;

/**
 * Utility functionality to import keras models
 *
 * @author Max Pumperla
 */
public class KerasLayerUtils {

    /**
     * Build KerasLayer from a Keras layer configuration.
     *
     * @param layerConfig map containing Keras layer properties
     * @return KerasLayer
     * @see Layer
     */
    public static KerasLayer getKerasLayerFromConfig(Map<String, Object> layerConfig,
                                              KerasLayerConfiguration conf,
                                              Map<String, Class<? extends KerasLayer>> customLayers)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getKerasLayerFromConfig(layerConfig, false, conf, customLayers);
    }

    /**
     * Build KerasLayer from a Keras layer configuration. Building layer with
     * enforceTrainingConfig=true will throw exceptions for unsupported Keras
     * options related to training (e.g., unknown regularizers). Otherwise
     * we only generate warnings.
     *
     * @param layerConfig           map containing Keras layer properties
     * @param enforceTrainingConfig whether to enforce training-only configurations
     * @return KerasLayer
     * @see Layer
     */
    public static KerasLayer getKerasLayerFromConfig(Map<String, Object> layerConfig, boolean enforceTrainingConfig,
                                              KerasLayerConfiguration conf,
                                              Map<String, Class<? extends KerasLayer>> customLayers)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String layerClassName = getClassNameFromConfig(layerConfig, conf);
        if (layerClassName.equals(conf.getLAYER_CLASS_NAME_TIME_DISTRIBUTED())) {
            layerConfig = getTimeDistributedLayerConfig(layerConfig, conf);
            layerClassName = getClassNameFromConfig(layerConfig, conf);
        }

        KerasLayer layer;
        if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ACTIVATION())) {
            layer = new KerasActivation(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_LEAKY_RELU())) {
            layer = new KerasLeakyReLU(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_DROPOUT())) {
            layer = new KerasDropout(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_DENSE()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE())) {
            layer = new KerasDense(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_LSTM())) {
            layer = new KerasLstm(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CONVOLUTION_2D())) {
            layer = new KerasConvolution2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CONVOLUTION_1D())) {
            layer = new KerasConvolution1D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MAX_POOLING_2D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_AVERAGE_POOLING_2D())) {
            layer = new KerasPooling(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D())) {
            layer = new KerasGlobalPooling(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_BATCHNORMALIZATION())) {
            layer = new KerasBatchNormalization(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_EMBEDDING())) {
            layer = new KerasEmbedding(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_INPUT())) {
            layer = new KerasInput(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MERGE())) {
            layer = new KerasMerge(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_FLATTEN())) {
            layer = new KerasFlatten(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_RESHAPE())) {
            layer = new KerasReshape(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ZERO_PADDING_2D())) {
            layer = new KerasZeroPadding(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MAX_POOLING_1D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_AVERAGE_POOLING_1D())) {
            layer = new KerasPooling(layerConfig, enforceTrainingConfig);
        } else {
            // check if user registered a custom config
            Class<? extends KerasLayer> customConfig = customLayers.get(layerClassName);

            if (customConfig == null)
                throw new UnsupportedKerasConfigurationException("Unsupported keras layer type " + layerClassName);
            try {
                Constructor constructor = customConfig.getConstructor(Map.class);
                layer = (KerasLayer) constructor.newInstance(layerConfig);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        return layer;
    }

    /**
     * Get Keras layer class name from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static String getClassNameFromConfig(Map<String, Object> layerConfig, KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(conf.getLAYER_FIELD_CLASS_NAME()))
            throw new InvalidKerasConfigurationException(
                    "Field " + conf.getLAYER_FIELD_CLASS_NAME() + " missing from layer config");
        return (String) layerConfig.get(conf.getLAYER_FIELD_CLASS_NAME());
    }

    /**
     * Extract inner layer config from TimeDistributed configuration and merge
     * it into the outer config.
     *
     * @param layerConfig dictionary containing Keras TimeDistributed configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static Map<String, Object> getTimeDistributedLayerConfig(Map<String, Object> layerConfig,
                                                             KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(conf.getLAYER_FIELD_CLASS_NAME()))
            throw new InvalidKerasConfigurationException(
                    "Field " + conf.getLAYER_FIELD_CLASS_NAME() + " missing from layer config");
        if (!layerConfig.get(conf.getLAYER_FIELD_CLASS_NAME()).equals(conf.getLAYER_CLASS_NAME_TIME_DISTRIBUTED()))
            throw new InvalidKerasConfigurationException("Expected " + conf.getLAYER_CLASS_NAME_TIME_DISTRIBUTED()
                    + " layer, found " + (String) layerConfig.get(conf.getLAYER_FIELD_CLASS_NAME()));
        if (!layerConfig.containsKey(conf.getLAYER_FIELD_CONFIG()))
            throw new InvalidKerasConfigurationException("Field "
                    + conf.getLAYER_FIELD_CONFIG() + " missing from layer config");
        Map<String, Object> outerConfig = getInnerLayerConfigFromConfig(layerConfig, conf);
        Map<String, Object> innerLayer = (Map<String, Object>) outerConfig.get(conf.getLAYER_FIELD_LAYER());
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), innerLayer.get(conf.getLAYER_FIELD_CLASS_NAME()));
        layerConfig.put(conf.getLAYER_FIELD_NAME(), innerLayer.get(conf.getLAYER_FIELD_CLASS_NAME()));
        Map<String, Object> innerConfig = (Map<String, Object>) getInnerLayerConfigFromConfig(innerLayer, conf);
        outerConfig.putAll(innerConfig);
        outerConfig.remove(conf.getLAYER_FIELD_LAYER());
        return layerConfig;
    }

    /**
     * Get inner layer config from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static Map<String, Object> getInnerLayerConfigFromConfig(Map<String, Object> layerConfig, KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(conf.getLAYER_FIELD_CONFIG()))
            throw new InvalidKerasConfigurationException("Field "
                    + conf.getLAYER_FIELD_CONFIG() + " missing from layer config");
        return (Map<String, Object>) layerConfig.get(conf.getLAYER_FIELD_CONFIG());
    }

}
