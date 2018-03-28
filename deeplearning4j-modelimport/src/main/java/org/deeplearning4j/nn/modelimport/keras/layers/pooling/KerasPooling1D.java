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
package org.deeplearning4j.nn.modelimport.keras.layers.pooling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;

/**
 * Imports a Keras 1D Pooling layer as a DL4J Subsampling layer.
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasPooling1D extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasPooling1D(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasPooling1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        Subsampling1DLayer.Builder builder = new Subsampling1DLayer.Builder(
                KerasPoolingUtils.mapPoolingType(this.className, conf)).name(this.layerName)
                .dropOut(this.dropout)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 1, conf, kerasMajorVersion)[0])
                .stride(getStrideFromConfig(layerConfig, 1, conf)[0]);
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 1, conf, kerasMajorVersion);
        if (padding != null)
            builder.padding(padding[0]);
        this.layer = builder.build();
        this.vertex = null;
    }

    /**
     * Get DL4J Subsampling1DLayer.
     *
     * @return Subsampling1DLayer
     */
    public Subsampling1DLayer getSubsampling1DLayer() {
        return (Subsampling1DLayer) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Subsampling 1D layer accepts only one input (received " + inputType.length + ")");
        return this.getSubsampling1DLayer().getOutputType(-1, inputType[0]);
    }
}
