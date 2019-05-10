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

import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;

import java.util.Map;

/**
 * Utility functionality for Keras activation functions.
 *
 * @author Max Pumperla
 */
public class KerasActivationUtils {

    /**
     * Map Keras to DL4J activation functions.
     *
     * @param conf Keras layer configuration
     * @param kerasActivation String containing Keras activation function name
     * @return Activation enum value containing DL4J activation function name
     */
    public static Activation mapToActivation(String kerasActivation, KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException {
        Activation dl4jActivation;
        if (kerasActivation.equals(conf.getKERAS_ACTIVATION_SOFTMAX())) {
            dl4jActivation = Activation.SOFTMAX;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_SOFTPLUS())) {
            dl4jActivation = Activation.SOFTPLUS;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_SOFTSIGN())) {
            dl4jActivation = Activation.SOFTSIGN;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_RELU())) {
            dl4jActivation = Activation.RELU;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_RELU6())) {
            dl4jActivation = Activation.RELU6;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_ELU())) {
            dl4jActivation = Activation.ELU;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_SELU())) {
            dl4jActivation = Activation.SELU;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_TANH())) {
            dl4jActivation = Activation.TANH;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_SIGMOID())) {
            dl4jActivation = Activation.SIGMOID;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_HARD_SIGMOID())) {
            dl4jActivation = Activation.HARDSIGMOID;
        } else if (kerasActivation.equals(conf.getKERAS_ACTIVATION_LINEAR())) {
            dl4jActivation = Activation.IDENTITY;
        } else {
            throw new UnsupportedKerasConfigurationException(
                    "Unknown Keras activation function " + kerasActivation);
        }
        return dl4jActivation;
    }


    /**
     * Map Keras to DL4J activation functions.
     *
     * @param kerasActivation String containing Keras activation function name
     * @return DL4J activation function
     */
    public static IActivation mapToIActivation(String kerasActivation, KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException {
        Activation activation = mapToActivation(kerasActivation, conf);
        return activation.getActivationFunction();
    }

    /**
     * Get activation function from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return DL4J activation function
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public static IActivation getIActivationFromConfig(Map<String, Object> layerConfig, KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getActivationFromConfig(layerConfig, conf).getActivationFunction();
    }

    /**
     * Get activation enum value from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return DL4J activation enum value
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public static Activation getActivationFromConfig(Map<String, Object> layerConfig, KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_ACTIVATION()))
            throw new InvalidKerasConfigurationException("Keras layer is missing "
                    + conf.getLAYER_FIELD_ACTIVATION() + " field");
        return mapToActivation((String) innerConfig.get(conf.getLAYER_FIELD_ACTIVATION()), conf);
    }
}
