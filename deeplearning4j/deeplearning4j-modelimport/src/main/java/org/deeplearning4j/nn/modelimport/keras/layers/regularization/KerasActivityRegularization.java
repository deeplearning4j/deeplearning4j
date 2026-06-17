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

package org.deeplearning4j.nn.modelimport.keras.layers.regularization;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.nd4j.linalg.activations.Activation;

import java.util.Map;

/**
 * Imports a Keras ActivityRegularization layer as a DL4J ActivationLayer with identity activation.
 *
 * ActivityRegularization in Keras applies L1 and/or L2 regularization to the layer output (activity).
 * Since DL4J's regularization is typically applied to weights rather than activations,
 * this layer is imported as a pass-through identity layer. The regularization effect
 * is noted but not fully replicated.
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasActivityRegularization extends KerasLayer {

    private double l1;
    private double l2;

    private static final String LAYER_L1 = "l1";
    private static final String LAYER_L2 = "l2";

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasActivityRegularization(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasActivityRegularization(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration.
     * @param enforceTrainingConfig whether to load Keras training configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasActivityRegularization(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);

        this.l1 = Double.parseDouble(innerConfig.getOrDefault(LAYER_L1, "0.0").toString());
        this.l2 = Double.parseDouble(innerConfig.getOrDefault(LAYER_L2, "0.0").toString());

        if (this.l1 > 0 || this.l2 > 0) {
            log.warn("Keras ActivityRegularization layer with L1={} and L2={} is being imported as an identity layer. " +
                    "Activity regularization is not directly supported in DL4J and will not be applied.", l1, l2);
        }

        // ActivityRegularization is implemented as an identity pass-through layer
        // The regularization effect is noted but not fully replicated in DL4J
        this.layer = new ActivationLayer.Builder()
                .name(this.layerName)
                .activation(Activation.IDENTITY)
                .build();
    }

    /**
     * Get DL4J ActivationLayer (with identity activation).
     *
     * @return ActivationLayer
     */
    public ActivationLayer getActivityRegularizationLayer() {
        return (ActivationLayer) this.layer;
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
                    "Keras ActivityRegularization layer accepts only one input (received " + inputType.length + ")");
        // ActivityRegularization layer preserves input type
        return inputType[0];
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters (0 for activity regularization)
     */
    @Override
    public int getNumParams() {
        return 0;
    }

    /**
     * Get L1 regularization strength.
     *
     * @return L1 regularization value
     */
    public double getL1() {
        return l1;
    }

    /**
     * Get L2 regularization strength.
     *
     * @return L2 regularization value
     */
    public double getL2() {
        return l2;
    }
}
