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
package org.deeplearning4j.nn.modelimport.keras.layers.normalization;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LayerNormalization;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Imports a Keras LayerNormalization layer.
 *
 * LayerNormalization normalizes inputs across the features dimension.
 * Unlike BatchNormalization, it normalizes per sample rather than across the batch.
 *
 * Keras config fields:
 * - axis: The axis or axes to normalize across (default: -1, i.e., last axis)
 * - epsilon: Small constant for numerical stability (default: 1e-5)
 * - center: Whether to add learnable beta offset (default: True)
 * - scale: Whether to add learnable gamma scale (default: True)
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasLayerNormalization extends KerasLayer {

    private int[] axis;
    private double epsilon;
    private boolean center;
    private boolean scale;
    private int numTrainableParams;

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLayerNormalization(Map<String, Object> layerConfig)
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
    public KerasLayerNormalization(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);

        // Parse axis (default -1, can be a list)
        Object axisObj = innerConfig.get("axis");
        if (axisObj != null) {
            if (axisObj instanceof List) {
                List<?> axisList = (List<?>) axisObj;
                this.axis = new int[axisList.size()];
                for (int i = 0; i < axisList.size(); i++) {
                    this.axis[i] = ((Number) axisList.get(i)).intValue();
                }
            } else {
                this.axis = new int[]{((Number) axisObj).intValue()};
            }
        } else {
            this.axis = new int[]{-1};
        }

        // Parse epsilon (default 1e-5)
        Object epsilonObj = innerConfig.get("epsilon");
        this.epsilon = epsilonObj != null ? ((Number) epsilonObj).doubleValue() : 1e-5;

        // Parse center (default true)
        Object centerObj = innerConfig.get("center");
        this.center = centerObj != null ? (Boolean) centerObj : true;

        // Parse scale (default true)
        Object scaleObj = innerConfig.get("scale");
        this.scale = scaleObj != null ? (Boolean) scaleObj : true;

        // Count trainable params
        this.numTrainableParams = 0;
        if (scale) numTrainableParams++;
        if (center) numTrainableParams++;

        // Build LayerNormalization layer
        LayerNormalization.Builder builder = new LayerNormalization.Builder()
                .name(this.layerName)
                .epsilon(epsilon)
                .center(center)
                .scale(scale);

        this.layer = builder.build();
    }

    /**
     * Get DL4J LayerNormalization layer.
     *
     * @return LayerNormalization
     */
    public LayerNormalization getLayerNormalizationLayer() {
        return (LayerNormalization) this.layer;
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
                    "Keras LayerNormalization layer accepts only one input (received " + inputType.length + ")");
        return inputType[0];
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters
     */
    @Override
    public int getNumParams() {
        return numTrainableParams;
    }

    /**
     * Set weights for layer.
     *
     * @param weights Map of weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();

        // Gamma (scale)
        if (scale) {
            INDArray gamma = null;
            if (weights.containsKey("gamma")) {
                gamma = weights.get("gamma");
            } else if (kerasMajorVersion == 1 && weights.containsKey("g")) {
                gamma = weights.get("g");
            }
            if (gamma != null) {
                this.weights.put("gamma", gamma);
            }
        }

        // Beta (center/offset)
        if (center) {
            INDArray beta = null;
            if (weights.containsKey("beta")) {
                beta = weights.get("beta");
            } else if (kerasMajorVersion == 1 && weights.containsKey("b")) {
                beta = weights.get("b");
            }
            if (beta != null) {
                this.weights.put("beta", beta);
            }
        }
    }
}
