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

package org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.PReLULayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PReLUParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.removeDefaultWeights;

/**
 * Imports PReLU layer from Keras
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasPReLU extends KerasLayer {

    private final String ALPHA = "alpha";
    private final String ALPHA_INIT = "alpha_initializer";
    private final String ALPHA_CONSTRAINT = "alpha_constraint";
    private final String SHARED_AXES = "shared_axes";

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Invalid Keras config
     */
    public KerasPReLU(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Invalid Keras config
     */
    public KerasPReLU(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, ALPHA_CONSTRAINT, conf, kerasMajorVersion);

        Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, ALPHA_INIT,
                enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit weightInit = init.getFirst();
        Distribution distribution = init.getSecond();
        long[] axes = getSharedAxes(layerConfig);

        PReLULayer.Builder builder = new PReLULayer.Builder().sharedAxes(axes)
        .weightInit(weightInit).name(layerName);
        if (distribution != null) {
            builder.dist(distribution);
        }
        if (weightConstraint != null){
            builder.constrainWeights(weightConstraint);
        }
        this.layer = builder.build();
    }

    private long[] getSharedAxes(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        long[] axes = null;
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        try {
            @SuppressWarnings("unchecked")
            List<Integer> axesList = (List<Integer>) innerConfig.get(SHARED_AXES);
            int[] intAxes = ArrayUtil.toArray(axesList);
            axes = new long[intAxes.length];
            for (int i = 0; i < intAxes.length; i++) {
                axes[i] = (long) intAxes[i];
            }
        } catch (Exception e) {
            // no shared axes
        }
        return axes;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras PReLU layer accepts only one input (received " + inputType.length + ")");
        InputType inType = inputType[0];

        // Dynamically infer input shape of PReLU layer from input type
        PReLULayer shapedLayer = (PReLULayer) this.layer;
        shapedLayer.setInputShape(inType.getShape());
        this.layer = shapedLayer;

        return this.getPReLULayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Get DL4J ActivationLayer.
     *
     * @return ActivationLayer
     */
    public PReLULayer getPReLULayer() {
        return (PReLULayer) this.layer;
    }

    /**
     * Set weights for layer.
     *
     * @param weights Dense layer weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();
        if (weights.containsKey(ALPHA))
            this.weights.put(PReLUParamInitializer.WEIGHT_KEY, weights.get(ALPHA));
        else
            throw new InvalidKerasConfigurationException("Parameter " + ALPHA + " does not exist in weights");
        if (weights.size() > 1) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(ALPHA);
            String unknownParamNames = paramNames.toString();
            log.warn("Attemping to set weights for unknown parameters: "
                    + unknownParamNames.substring(1, unknownParamNames.length() - 1));
        }
    }

}
