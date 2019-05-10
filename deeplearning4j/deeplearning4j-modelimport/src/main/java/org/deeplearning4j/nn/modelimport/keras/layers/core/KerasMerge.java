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

package org.deeplearning4j.nn.modelimport.keras.layers.core;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

/**
 * Imports a Keras Merge layer as a DL4J Merge (graph) vertex.
 * <p>
 * TODO: handle axes arguments that alter merge behavior (requires changes to DL4J?)
 *
 * @author dave@skymind.io
 */
@Slf4j
@Data
public class KerasMerge extends KerasLayer {

    private final String LAYER_FIELD_MODE = "mode";
    private final String LAYER_MERGE_MODE_SUM = "sum";
    private final String LAYER_MERGE_MODE_MUL = "mul";
    private final String LAYER_MERGE_MODE_CONCAT = "concat";
    private final String LAYER_MERGE_MODE_AVE = "ave";
    private final String LAYER_MERGE_MODE_COS = "cos";
    private final String LAYER_MERGE_MODE_DOT = "dot";
    private final String LAYER_MERGE_MODE_MAX = "max";

    private ElementWiseVertex.Op mergeMode = null;

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMerge(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMerge(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary and merge mode passed in.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param mergeMode             ElementWiseVertex merge mode
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMerge(Map<String, Object> layerConfig, ElementWiseVertex.Op mergeMode, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.mergeMode = mergeMode;
        if (this.mergeMode == null)
            this.vertex = new MergeVertex();
        else
            this.vertex = new ElementWiseVertex(mergeMode);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMerge(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.mergeMode = getMergeMode(layerConfig);
        if (this.mergeMode == null)
            this.vertex = new MergeVertex();
        else
            this.vertex = new ElementWiseVertex(mergeMode);
    }

    private ElementWiseVertex.Op getMergeMode(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(LAYER_FIELD_MODE))
            throw new InvalidKerasConfigurationException(
                    "Keras Merge layer config missing " + LAYER_FIELD_MODE + " field");
        ElementWiseVertex.Op op = null;
        String mergeMode = (String) innerConfig.get(LAYER_FIELD_MODE);
        switch (mergeMode) {
            case LAYER_MERGE_MODE_SUM:
                op = ElementWiseVertex.Op.Add;
                break;
            case LAYER_MERGE_MODE_MUL:
                op = ElementWiseVertex.Op.Product;
                break;
            case LAYER_MERGE_MODE_CONCAT:
                // leave null
                break;
            case LAYER_MERGE_MODE_AVE:
                op = ElementWiseVertex.Op.Average;
                break;
            case LAYER_MERGE_MODE_MAX:
                op = ElementWiseVertex.Op.Max;
                break;
            case LAYER_MERGE_MODE_COS:
            case LAYER_MERGE_MODE_DOT:
            default:
                throw new UnsupportedKerasConfigurationException(
                        "Keras Merge layer mode " + mergeMode + " not supported");
        }
        return op;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     */
    @Override
    public InputType getOutputType(InputType... inputType) {
        return this.vertex.getOutputType(-1, inputType);
    }
}
