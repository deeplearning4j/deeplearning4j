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

package org.deeplearning4j.nn.modelimport.keras.layers.attention;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.graph.DotProductAttentionVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.List;
import java.util.Map;

/**
 * Imports a Keras MultiHeadAttention layer.
 *
 * MultiHeadAttention layer as described in "Attention Is All You Need" paper.
 * This layer implements multi-head attention mechanism.
 *
 * Note: DL4J uses DotProductAttentionVertex as a simpler approximation.
 * Full multi-head attention semantics may not be fully replicated.
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasMultiHeadAttention extends KerasLayer {

    private int numHeads;
    private int keyDim;
    private int valueDim;
    private double dropout;
    private boolean useBias;
    private List<String> inputNames;

    private static final String LAYER_NUM_HEADS = "num_heads";
    private static final String LAYER_KEY_DIM = "key_dim";
    private static final String LAYER_VALUE_DIM = "value_dim";
    private static final String LAYER_DROPOUT = "dropout";
    private static final String LAYER_USE_BIAS = "use_bias";

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMultiHeadAttention(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Default constructor
     *
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMultiHeadAttention() throws UnsupportedKerasConfigurationException {
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMultiHeadAttention(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, false);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration.
     * @param enforceTrainingConfig whether to load Keras training configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasMultiHeadAttention(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        log.warn("Keras MultiHeadAttention is being imported using DotProductAttentionVertex. " +
                "Full multi-head attention semantics may not be fully replicated.");

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);

        this.numHeads = ((Number) innerConfig.getOrDefault(LAYER_NUM_HEADS, 8)).intValue();
        this.keyDim = ((Number) innerConfig.getOrDefault(LAYER_KEY_DIM, 64)).intValue();
        this.valueDim = ((Number) innerConfig.getOrDefault(LAYER_VALUE_DIM, this.keyDim)).intValue();
        this.dropout = Double.parseDouble(innerConfig.getOrDefault(LAYER_DROPOUT, "0.0").toString());
        this.useBias = Boolean.parseBoolean(innerConfig.getOrDefault(LAYER_USE_BIAS, "true").toString());
        this.inputNames = KerasLayerUtils.getInboundLayerNamesFromConfig(layerConfig, conf);

        // Using DotProductAttentionVertex as approximation
        this.vertex = new DotProductAttentionVertex.Builder()
                .dropoutProbability(dropout)
                .scale(1.0 / Math.sqrt(keyDim))
                .inputNames(inputNames)
                .build();
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
        InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
        DotProductAttentionVertex attentionVertex = (DotProductAttentionVertex) vertex;

        switch (inputType[0].getType()) {
            case FF:
                InputType.InputTypeFeedForward ff = (InputType.InputTypeFeedForward) inputType[0];
                attentionVertex.setNIn(ff.getSize());
                attentionVertex.setNOut(ff.getSize());
                break;
            case CNN:
                InputType.InputTypeConvolutional cnn = (InputType.InputTypeConvolutional) inputType[0];
                attentionVertex.setNIn(cnn.getChannels());
                attentionVertex.setNOut(cnn.getChannels());
                break;
            case RNN:
                InputType.InputTypeRecurrent rnn = (InputType.InputTypeRecurrent) inputType[0];
                attentionVertex.setNIn(rnn.getSize());
                attentionVertex.setNOut(rnn.getSize());
                break;
            case CNN3D:
            case CNNFlat:
                throw new InvalidKerasConfigurationException(
                        "Unsupported input type for MultiHeadAttention layer: " + inputType[0].getType());
        }

        if (preprocessor != null) {
            return attentionVertex.getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }

        return attentionVertex.getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters
     */
    @Override
    public int getNumParams() {
        // Simplified - actual MHA has more params based on heads and dims
        return 0;
    }
}
