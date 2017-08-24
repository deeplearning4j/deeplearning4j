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
package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.getActivationFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getConvolutionModeFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getKernelSizeFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getStrideFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getPaddingFromBorderModeConfig;

/**
 * Imports a 1D Convolution layer from Keras.
 *
 * @author Max Pumperla
 */
@Slf4j
@Data
public class KerasConvolution1D extends KerasConvolution {

    /**
     * Pass-through constructor from KerasLayer
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution1D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution1D(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig               dictionary containing Keras layer configuration
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        hasBias = getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;

        Convolution1DLayer.Builder builder = new Convolution1DLayer.Builder().name(this.layerName)
                .nOut(getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
                .activation(getActivationFromConfig(layerConfig, conf))
                .weightInit(getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                        enforceTrainingConfig, conf, kerasMajorVersion))
                .biasInit(0.0)
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 1,  conf, kerasMajorVersion)[0])
                .hasBias(hasBias).stride(getStrideFromConfig(layerConfig, 1, conf)[0]);
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 1, conf, kerasMajorVersion);
        if (padding != null)
            builder.padding(padding[0]);
        this.layer = builder.build();
    }

    /**
     * Get DL4J ConvolutionLayer.
     *
     * @return  ConvolutionLayer
     */
    public Convolution1DLayer getConvolution1DLayer() {
        return (Convolution1DLayer) this.layer;
    }
}
