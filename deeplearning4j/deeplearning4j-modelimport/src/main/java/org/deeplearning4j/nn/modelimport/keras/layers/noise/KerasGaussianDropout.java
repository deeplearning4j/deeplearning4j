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

package org.deeplearning4j.nn.modelimport.keras.layers.noise;

import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

/**
 * Keras wrapper for DL4J dropout layer with GaussianDropout.
 *
 * @author Max Pumperla
 */
public class KerasGaussianDropout extends KerasLayer {

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Invalid Keras config
     */
    public KerasGaussianDropout(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasGaussianDropout(Map<String, Object> layerConfig)
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
    public KerasGaussianDropout(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_RATE())) {
            throw new InvalidKerasConfigurationException("Keras configuration does not contain " +
                    "parameter" + conf.getLAYER_FIELD_RATE() +
                    "needed for GaussianDropout");
        }
        double rate = (double) innerConfig.get(conf.getLAYER_FIELD_RATE()); // Keras stores drop rates
        double retainRate = 1 - rate;

        this.layer = new DropoutLayer.Builder().name(this.layerName)
                .dropOut(new GaussianDropout(retainRate)).build();
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
                    "Keras Gaussian Dropout layer accepts only one input (received " + inputType.length + ")");
        return this.getGaussianDropoutLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Get DL4J DropoutLayer with Gaussian dropout.
     *
     * @return DropoutLayer
     */
    public DropoutLayer getGaussianDropoutLayer() {
        return (DropoutLayer) this.layer;
    }
}
