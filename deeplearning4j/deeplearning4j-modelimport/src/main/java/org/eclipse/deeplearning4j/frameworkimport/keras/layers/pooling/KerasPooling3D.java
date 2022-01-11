/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.frameworkimport.keras.layers.pooling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Subsampling3DLayer;
import org.eclipse.deeplearning4j.frameworkimport.keras.exceptions.InvalidKerasConfigurationException;
import org.eclipse.deeplearning4j.frameworkimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.eclipse.deeplearning4j.frameworkimport.keras.KerasLayer;

import java.util.Map;

import static org.eclipse.deeplearning4j.frameworkimport.keras.layers.convolutional.KerasConvolutionUtils.*;

/**
 * Imports a Keras 3D Pooling layer as a DL4J Subsampling3D layer.
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasPooling3D extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasPooling3D(Map<String, Object> layerConfig)
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
    public KerasPooling3D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        Subsampling3DLayer.Builder builder = new Subsampling3DLayer.Builder(
                KerasPoolingUtils.mapPoolingType(this.className, conf)).name(this.layerName)
                .dropOut(this.dropout)
                .dataFormat(getCNN3DDataFormatFromConfig(layerConfig,conf))
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 3, conf, kerasMajorVersion))
                .stride(getStrideFromConfig(layerConfig, 3, conf));
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 3, conf, kerasMajorVersion);
        if (padding != null)
            builder.padding(padding);
        this.layer = builder.build();
        this.vertex = null;
    }

    /**
     * Get DL4J Subsampling3DLayer.
     *
     * @return Subsampling3DLayer
     */
    public Subsampling3DLayer getSubsampling3DLayer() {
        return (Subsampling3DLayer) this.layer;
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
                    "Keras Subsampling/Pooling 3D layer accepts only one input (received " + inputType.length + ")");
        return this.getSubsampling3DLayer().getOutputType(-1, inputType[0]);
    }
}
