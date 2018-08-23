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

package org.deeplearning4j.nn.modelimport.keras.utils;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasInput;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasLeakyReLU;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasPReLU;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasThresholdedReLU;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.*;
import org.deeplearning4j.nn.modelimport.keras.layers.core.*;
import org.deeplearning4j.nn.modelimport.keras.layers.embeddings.KerasEmbedding;
import org.deeplearning4j.nn.modelimport.keras.layers.noise.KerasAlphaDropout;
import org.deeplearning4j.nn.modelimport.keras.layers.noise.KerasGaussianDropout;
import org.deeplearning4j.nn.modelimport.keras.layers.noise.KerasGaussianNoise;
import org.deeplearning4j.nn.modelimport.keras.layers.normalization.KerasBatchNormalization;
import org.deeplearning4j.nn.modelimport.keras.layers.pooling.KerasGlobalPooling;
import org.deeplearning4j.nn.modelimport.keras.layers.pooling.KerasPooling1D;
import org.deeplearning4j.nn.modelimport.keras.layers.pooling.KerasPooling2D;
import org.deeplearning4j.nn.modelimport.keras.layers.pooling.KerasPooling3D;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasLSTM;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasSimpleRnn;
import org.deeplearning4j.nn.modelimport.keras.layers.wrappers.KerasBidirectional;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility functionality to import keras models
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasLayerUtils {

    /**
     * Checks whether layer config contains unsupported options.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to use Keras training configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public static void checkForUnsupportedConfigurations(Map<String, Object> layerConfig,
                                                         boolean enforceTrainingConfig,
                                                         KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        getBiasL1RegularizationFromConfig(layerConfig, enforceTrainingConfig, conf);
        getBiasL2RegularizationFromConfig(layerConfig, enforceTrainingConfig, conf);
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (innerConfig.containsKey(conf.getLAYER_FIELD_W_REGULARIZER())) {
            checkForUnknownRegularizer((Map<String, Object>) innerConfig.get(conf.getLAYER_FIELD_W_REGULARIZER()),
                    enforceTrainingConfig, conf);
        }
        if (innerConfig.containsKey(conf.getLAYER_FIELD_B_REGULARIZER())) {
            checkForUnknownRegularizer((Map<String, Object>) innerConfig.get(conf.getLAYER_FIELD_B_REGULARIZER()),
                    enforceTrainingConfig, conf);
        }
    }

    /**
     * Get L1 bias regularization (if any) from Keras bias regularization configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return L1 regularization strength (0.0 if none)
     */
    public static double getBiasL1RegularizationFromConfig(Map<String, Object> layerConfig,
                                                           boolean enforceTrainingConfig,
                                                           KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (innerConfig.containsKey(conf.getLAYER_FIELD_B_REGULARIZER())) {
            Map<String, Object> regularizerConfig =
                    (Map<String, Object>) innerConfig.get(conf.getLAYER_FIELD_B_REGULARIZER());
            if (regularizerConfig != null && regularizerConfig.containsKey(conf.getREGULARIZATION_TYPE_L1()))
                throw new UnsupportedKerasConfigurationException("L1 regularization for bias parameter not supported");
        }
        return 0.0;
    }

    /**
     * Get L2 bias regularization (if any) from Keras bias regularization configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return L1 regularization strength (0.0 if none)
     */
    private static double getBiasL2RegularizationFromConfig(Map<String, Object> layerConfig,
                                                            boolean enforceTrainingConfig,
                                                            KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (innerConfig.containsKey(conf.getLAYER_FIELD_B_REGULARIZER())) {
            Map<String, Object> regularizerConfig =
                    (Map<String, Object>) innerConfig.get(conf.getLAYER_FIELD_B_REGULARIZER());
            if (regularizerConfig != null && regularizerConfig.containsKey(conf.getREGULARIZATION_TYPE_L2()))
                throw new UnsupportedKerasConfigurationException("L2 regularization for bias parameter not supported");
        }
        return 0.0;
    }

    /**
     * Check whether Keras weight regularization is of unknown type. Currently prints a warning
     * since main use case for model import is inference, not further training. Unlikely since
     * standard Keras weight regularizers are L1 and L2.
     *
     * @param regularizerConfig Map containing Keras weight reguarlization configuration
     */
    private static void checkForUnknownRegularizer(Map<String, Object> regularizerConfig, boolean enforceTrainingConfig,
                                                   KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException {
        if (regularizerConfig != null) {
            for (String field : regularizerConfig.keySet()) {
                if (!field.equals(conf.getREGULARIZATION_TYPE_L1()) && !field.equals(conf.getREGULARIZATION_TYPE_L2())
                        && !field.equals(conf.getLAYER_FIELD_NAME())
                        && !field.equals(conf.getLAYER_FIELD_CLASS_NAME())
                        && !field.equals(conf.getLAYER_FIELD_CONFIG())) {
                    if (enforceTrainingConfig)
                        throw new UnsupportedKerasConfigurationException("Unknown regularization field " + field);
                    else
                        log.warn("Ignoring unknown regularization field " + field);
                }
            }
        }
    }


    /**
     * Build KerasLayer from a Keras layer configuration.
     *
     * @param layerConfig map containing Keras layer properties
     * @return KerasLayer
     * @see Layer
     */
    public static KerasLayer getKerasLayerFromConfig(Map<String, Object> layerConfig,
                                                     KerasLayerConfiguration conf,
                                                     Map<String, Class<? extends KerasLayer>> customLayers,
                                                     Map<String, SameDiffLambdaLayer> lambdaLayers,
                                                     Map<String, ? extends KerasLayer> previousLayers)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getKerasLayerFromConfig(layerConfig, false, conf, customLayers, lambdaLayers, previousLayers);
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
    public static KerasLayer getKerasLayerFromConfig(Map<String, Object> layerConfig,
                                                     boolean enforceTrainingConfig,
                                                     KerasLayerConfiguration conf,
                                                     Map<String, Class<? extends KerasLayer>> customLayers,
                                                     Map<String, SameDiffLambdaLayer> lambdaLayers,
                                                     Map<String, ? extends KerasLayer> previousLayers
    )
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
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MASKING())) {
            layer = new KerasMasking(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_THRESHOLDED_RELU())) {
            layer = new KerasThresholdedReLU(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_PRELU())) {
            layer = new KerasPReLU(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_DROPOUT())) {
            layer = new KerasDropout(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_SPATIAL_DROPOUT_1D())
                || layerClassName.equals(conf.getLAYER_CLASS_NAME_SPATIAL_DROPOUT_2D())
                || layerClassName.equals(conf.getLAYER_CLASS_NAME_SPATIAL_DROPOUT_3D())) {
            layer = new KerasSpatialDropout(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ALPHA_DROPOUT())) {
            layer = new KerasAlphaDropout(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_GAUSSIAN_DROPOUT())) {
            layer = new KerasGaussianDropout(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_GAUSSIAN_NOISE())) {
            layer = new KerasGaussianNoise(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_DENSE()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE())) {
            layer = new KerasDense(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_BIDIRECTIONAL())) {
            layer = new KerasBidirectional(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_LSTM())) {
            layer = new KerasLSTM(layerConfig, enforceTrainingConfig, previousLayers);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_SIMPLE_RNN())) {
            layer = new KerasSimpleRnn(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CONVOLUTION_3D())) {
            layer = new KerasConvolution3D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CONVOLUTION_2D())) {
            layer = new KerasConvolution2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_DECONVOLUTION_2D())) {
            layer = new KerasDeconvolution2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CONVOLUTION_1D())) {
            layer = new KerasConvolution1D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ATROUS_CONVOLUTION_2D())) {
            layer = new KerasAtrousConvolution2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ATROUS_CONVOLUTION_1D())) {
            layer = new KerasAtrousConvolution1D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_DEPTHWISE_CONVOLUTION_2D())) {
            layer = new KerasDepthwiseConvolution2D(layerConfig, previousLayers, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_2D())) {
            layer = new KerasSeparableConvolution2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MAX_POOLING_3D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_AVERAGE_POOLING_3D())) {
            layer = new KerasPooling3D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MAX_POOLING_2D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_AVERAGE_POOLING_2D())) {
            layer = new KerasPooling2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MAX_POOLING_1D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_AVERAGE_POOLING_1D())) {
            layer = new KerasPooling1D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_3D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_GLOBAL_MAX_POOLING_3D())) {
            layer = new KerasGlobalPooling(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_BATCHNORMALIZATION())) {
            layer = new KerasBatchNormalization(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_EMBEDDING())) {
            layer = new KerasEmbedding(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_INPUT())) {
            layer = new KerasInput(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_REPEAT())) {
            layer = new KerasRepeatVector(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_PERMUTE())) {
            layer = new KerasPermute(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MERGE())) {
            layer = new KerasMerge(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ADD()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_ADD())) {
            layer = new KerasMerge(layerConfig, ElementWiseVertex.Op.Add, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_SUBTRACT()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_FUNCTIONAL_SUBTRACT())) {
            layer = new KerasMerge(layerConfig, ElementWiseVertex.Op.Subtract, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_AVERAGE()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_FUNCTIONAL_AVERAGE())) {
            layer = new KerasMerge(layerConfig, ElementWiseVertex.Op.Average, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_MULTIPLY()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_FUNCTIONAL_MULTIPLY())) {
            layer = new KerasMerge(layerConfig, ElementWiseVertex.Op.Product, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CONCATENATE()) ||
                layerClassName.equals(conf.getLAYER_CLASS_NAME_FUNCTIONAL_CONCATENATE())) {
            layer = new KerasMerge(layerConfig, null, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_FLATTEN())) {
            layer = new KerasFlatten(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_RESHAPE())) {
            layer = new KerasReshape(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ZERO_PADDING_1D())) {
            layer = new KerasZeroPadding1D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ZERO_PADDING_2D())) {
            layer = new KerasZeroPadding2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_ZERO_PADDING_3D())) {
            layer = new KerasZeroPadding3D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_UPSAMPLING_1D())) {
            layer = new KerasUpsampling1D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_UPSAMPLING_2D())) {
            layer = new KerasUpsampling2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CROPPING_3D())) {
            layer = new KerasCropping3D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CROPPING_2D())) {
            layer = new KerasCropping2D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_CROPPING_1D())) {
            layer = new KerasCropping1D(layerConfig, enforceTrainingConfig);
        } else if (layerClassName.equals(conf.getLAYER_CLASS_NAME_LAMBDA()) && !lambdaLayers.isEmpty()) {
            String lambdaLayerName = KerasLayerUtils.getLayerNameFromConfig(layerConfig, conf);
            SameDiffLambdaLayer lambdaLayer;
            if (lambdaLayers.containsKey(lambdaLayerName)) {
                lambdaLayer = lambdaLayers.get(lambdaLayerName);
            } else {
                throw new UnsupportedKerasConfigurationException("No SameDiff Lambda layer found for Lambda" +
                        "layer " + lambdaLayerName);
            }
            layer = new KerasLambda(layerConfig, enforceTrainingConfig, lambdaLayer);
        } else {
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
     * @return Keras layer class name
     * @throws InvalidKerasConfigurationException Invalid Keras config
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
     * @return Time distributed layer config
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static Map<String, Object> getTimeDistributedLayerConfig(Map<String, Object> layerConfig,
                                                                    KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(conf.getLAYER_FIELD_CLASS_NAME()))
            throw new InvalidKerasConfigurationException(
                    "Field " + conf.getLAYER_FIELD_CLASS_NAME() + " missing from layer config");
        if (!layerConfig.get(conf.getLAYER_FIELD_CLASS_NAME()).equals(conf.getLAYER_CLASS_NAME_TIME_DISTRIBUTED()))
            throw new InvalidKerasConfigurationException("Expected " + conf.getLAYER_CLASS_NAME_TIME_DISTRIBUTED()
                    + " layer, found " + layerConfig.get(conf.getLAYER_FIELD_CLASS_NAME()));
        if (!layerConfig.containsKey(conf.getLAYER_FIELD_CONFIG()))
            throw new InvalidKerasConfigurationException("Field "
                    + conf.getLAYER_FIELD_CONFIG() + " missing from layer config");
        Map<String, Object> outerConfig = getInnerLayerConfigFromConfig(layerConfig, conf);
        Map<String, Object> innerLayer = (Map<String, Object>) outerConfig.get(conf.getLAYER_FIELD_LAYER());
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), innerLayer.get(conf.getLAYER_FIELD_CLASS_NAME()));
        layerConfig.put(conf.getLAYER_FIELD_NAME(), innerLayer.get(conf.getLAYER_FIELD_CLASS_NAME()));
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(innerLayer, conf);
        outerConfig.putAll(innerConfig);
        outerConfig.remove(conf.getLAYER_FIELD_LAYER());
        return layerConfig;
    }

    /**
     * Get inner layer config from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return Inner layer config for a nested Keras layer configuration
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static Map<String, Object> getInnerLayerConfigFromConfig(Map<String, Object> layerConfig, KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(conf.getLAYER_FIELD_CONFIG()))
            throw new InvalidKerasConfigurationException("Field "
                    + conf.getLAYER_FIELD_CONFIG() + " missing from layer config");
        return (Map<String, Object>) layerConfig.get(conf.getLAYER_FIELD_CONFIG());
    }

    /**
     * Get layer name from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return Keras layer name
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static String getLayerNameFromConfig(Map<String, Object> layerConfig,
                                                KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_NAME()))
            throw new InvalidKerasConfigurationException("Field " + conf.getLAYER_FIELD_NAME()
                    + " missing from layer config");
        return (String) innerConfig.get(conf.getLAYER_FIELD_NAME());
    }

    /**
     * Get Keras input shape from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return input shape array
     */
    public static int[] getInputShapeFromConfig(Map<String, Object> layerConfig,
                                                KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        // TODO: validate this. shouldn't we also have INPUT_SHAPE checked?
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_BATCH_INPUT_SHAPE()))
            return null;
        List<Integer> batchInputShape = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_BATCH_INPUT_SHAPE());
        int[] inputShape = new int[batchInputShape.size() - 1];
        for (int i = 1; i < batchInputShape.size(); i++) {
            inputShape[i - 1] = batchInputShape.get(i) != null ? batchInputShape.get(i) : 0;
        }
        return inputShape;
    }

    /**
     * Get Keras (backend) dimension order from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return Dimension order
     */
    public static KerasLayer.DimOrder getDimOrderFromConfig(Map<String, Object> layerConfig,
                                                            KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        KerasLayer.DimOrder dimOrder = KerasLayer.DimOrder.NONE;
        if (layerConfig.containsKey(conf.getLAYER_FIELD_BACKEND())) {
            String backend = (String) layerConfig.get(conf.getLAYER_FIELD_BACKEND());
            if (backend.equals("tensorflow") || backend.equals("cntk")) {
                dimOrder = KerasLayer.DimOrder.TENSORFLOW;
            } else if (backend.equals("theano")) {
                dimOrder = KerasLayer.DimOrder.THEANO;
            }
        }
        if (innerConfig.containsKey(conf.getLAYER_FIELD_DIM_ORDERING())) {
            String dimOrderStr = (String) innerConfig.get(conf.getLAYER_FIELD_DIM_ORDERING());
            if (dimOrderStr.equals(conf.getDIM_ORDERING_TENSORFLOW())) {
                dimOrder = KerasLayer.DimOrder.TENSORFLOW;
            } else if (dimOrderStr.equals(conf.getDIM_ORDERING_THEANO())) {
                dimOrder = KerasLayer.DimOrder.THEANO;
            } else {
                log.warn("Keras layer has unknown Keras dimension order: " + dimOrder);
            }
        }
        return dimOrder;
    }

    /**
     * Get list of inbound layers from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return List of inbound layer names
     */
    public static List<String> getInboundLayerNamesFromConfig(Map<String, Object> layerConfig, KerasLayerConfiguration conf) {
        List<String> inboundLayerNames = new ArrayList<>();
        if (layerConfig.containsKey(conf.getLAYER_FIELD_INBOUND_NODES())) {
            List<Object> inboundNodes = (List<Object>) layerConfig.get(conf.getLAYER_FIELD_INBOUND_NODES());
            if (!inboundNodes.isEmpty()) {
                inboundNodes = (List<Object>) inboundNodes.get(0);
                for (Object o : inboundNodes) {
                    String nodeName = (String) ((List<Object>) o).get(0);
                    inboundLayerNames.add(nodeName);
                }
            }
        }
        return inboundLayerNames;
    }

    /**
     * Get number of outputs from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return Number of output neurons of the Keras layer
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static int getNOutFromConfig(Map<String, Object> layerConfig,
                                        KerasLayerConfiguration conf) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        int nOut;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_OUTPUT_DIM()))
            /* Most feedforward layers: Dense, RNN, etc. */
            nOut = (int) innerConfig.get(conf.getLAYER_FIELD_OUTPUT_DIM());
        else if (innerConfig.containsKey(conf.getLAYER_FIELD_EMBEDDING_OUTPUT_DIM()))
            /* Embedding layers. */
            nOut = (int) innerConfig.get(conf.getLAYER_FIELD_EMBEDDING_OUTPUT_DIM());
        else if (innerConfig.containsKey(conf.getLAYER_FIELD_NB_FILTER()))
            /* Convolutional layers. */
            nOut = (int) innerConfig.get(conf.getLAYER_FIELD_NB_FILTER());
        else
            throw new InvalidKerasConfigurationException("Could not determine number of outputs for layer: no "
                    + conf.getLAYER_FIELD_OUTPUT_DIM() + " or " + conf.getLAYER_FIELD_NB_FILTER() + " field found");
        return nOut;
    }

    /**
     * Get dropout from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return get dropout value from Keras config
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static double getDropoutFromConfig(Map<String, Object> layerConfig,
                                              KerasLayerConfiguration conf) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        /* NOTE: Keras "dropout" parameter determines dropout probability,
         * while DL4J "dropout" parameter determines retention probability.
         */
        double dropout = 1.0;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_DROPOUT())) {
            /* For most feedforward layers. */
            try {
                dropout = 1.0 - (double) innerConfig.get(conf.getLAYER_FIELD_DROPOUT());
            } catch (Exception e) {
                int kerasDropout = (int) innerConfig.get(conf.getLAYER_FIELD_DROPOUT());
                dropout = 1.0 - kerasDropout;
            }
        } else if (innerConfig.containsKey(conf.getLAYER_FIELD_DROPOUT_W())) {
            /* For LSTMs. */
            try {
                dropout = 1.0 - (double) innerConfig.get(conf.getLAYER_FIELD_DROPOUT_W());
            } catch (Exception e) {
                int kerasDropout = (int) innerConfig.get(conf.getLAYER_FIELD_DROPOUT_W());
                dropout = 1.0 - kerasDropout;
            }
        }
        return dropout;
    }

    /**
     * Determine if layer should be instantiated with bias
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return whether layer has a bias term
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static boolean getHasBiasFromConfig(Map<String, Object> layerConfig,
                                               KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        boolean hasBias = true;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_USE_BIAS())) {
            hasBias = (boolean) innerConfig.get(conf.getLAYER_FIELD_USE_BIAS());
        }
        return hasBias;
    }

    /**
     * Get zero masking flag
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return if masking zeros or not
     * @throws InvalidKerasConfigurationException Invalid Keras configuration
     */
    public static boolean getZeroMaskingFromConfig(Map<String, Object> layerConfig,
                                                   KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        boolean hasZeroMasking = true;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_MASK_ZERO())) {
            hasZeroMasking = (boolean) innerConfig.get(conf.getLAYER_FIELD_MASK_ZERO());
        }
        return hasZeroMasking;
    }

    /**
     * Get mask value
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return mask value, defaults to 0.0
     * @throws InvalidKerasConfigurationException Invalid Keras configuration
     */
    public static double getMaskingValueFromConfig(Map<String, Object> layerConfig,
                                                   KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        double maskValue = 0.0;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_MASK_VALUE())) {
            try {
                maskValue = (double) innerConfig.get(conf.getLAYER_FIELD_MASK_VALUE());
            } catch (Exception e) {
                log.warn("Couldn't read masking value, default to 0.0");
            }
        } else {
            throw new InvalidKerasConfigurationException("No mask value found, field "
                    + conf.getLAYER_FIELD_MASK_VALUE());
        }
        return maskValue;
    }


    /**
     * Remove weights from config after weight setting.
     *
     * @param weights layer weights
     * @param conf Keras layer configuration
     */
    public static void removeDefaultWeights(Map<String, INDArray> weights, KerasLayerConfiguration conf) {
        if (weights.size() > 2) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(conf.getKERAS_PARAM_NAME_W());
            paramNames.remove(conf.getKERAS_PARAM_NAME_B());
            String unknownParamNames = paramNames.toString();
            log.warn("Attemping to set weights for unknown parameters: "
                    + unknownParamNames.substring(1, unknownParamNames.length() - 1));
        }
    }

}
