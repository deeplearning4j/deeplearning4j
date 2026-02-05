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
 * Imports a Keras AdditiveAttention layer (Bahdanau-style attention).
 *
 * AdditiveAttention layer implements Bahdanau-style (additive) attention mechanism.
 * The attention score is computed as: score = tanh(W_q * query + W_k * key)
 *
 * Note: DL4J uses DotProductAttentionVertex as an approximation since
 * native additive attention is not available.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasAdditiveAttention extends KerasLayer {

    private boolean useScale;
    private double dropout;
    private List<String> inputNames;

    private static final String LAYER_USE_SCALE = "use_scale";
    private static final String LAYER_DROPOUT = "dropout";

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasAdditiveAttention(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Default constructor
     *
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasAdditiveAttention() throws UnsupportedKerasConfigurationException {
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasAdditiveAttention(Map<String, Object> layerConfig)
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
    public KerasAdditiveAttention(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        log.warn("Keras AdditiveAttention (Bahdanau-style) is being imported using DotProductAttentionVertex. " +
                "Additive attention semantics are not fully replicated - dot product attention is used instead.");

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);

        this.useScale = Boolean.parseBoolean(innerConfig.getOrDefault(LAYER_USE_SCALE, "true").toString());
        this.dropout = Double.parseDouble(innerConfig.getOrDefault(LAYER_DROPOUT, "0.0").toString());
        this.inputNames = KerasLayerUtils.getInboundLayerNamesFromConfig(layerConfig, conf);

        // Using DotProductAttentionVertex as approximation since DL4J doesn't have additive attention
        this.vertex = new DotProductAttentionVertex.Builder()
                .dropoutProbability(dropout)
                .scale(useScale ? 0.2 : 1.0)
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
                        "Unsupported input type for AdditiveAttention layer: " + inputType[0].getType());
        }

        if (preprocessor != null) {
            return attentionVertex.getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }

        return attentionVertex.getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters (0 for vertex-based implementation)
     */
    @Override
    public int getNumParams() {
        return 0;
    }
}
