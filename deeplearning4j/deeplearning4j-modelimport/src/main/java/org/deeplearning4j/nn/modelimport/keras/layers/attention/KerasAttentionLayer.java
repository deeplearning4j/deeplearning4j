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
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.List;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;

/**
 * Docs from https://keras.io/api/layers/attention_layers/attention/
 * @author Adam Gibson
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasAttentionLayer extends KerasLayer {

    private boolean useScale;
    private double dropOut;
    private String scoreMode;
    private List<String> inputNames;

    /**
     * Float between 0 and 1. Fraction of the units to drop for the attention scores. Defaults to 0.0.
     */
    private final String LAYER_DROP_OUT = "dropout";
    /**
     * Function to use to compute attention scores, one of {"dot", "concat"}. "dot" refers to
     * the dot product between the query and key vectors.\
     * "concat" refers to the hyperbolic tangent of the concatenation of the query and key vectors.
     */
    private final String LAYER_SCORE_MODE = "score_mode";
    private final String LAYER_SCORE_MODE_DOT = "dot";
    private final String LAYER_SCORE_MODE_CONCAT = "concat";


    private final String LAYER_USE_SCALE = "use_scale";


    public KerasAttentionLayer(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);

    }

    public KerasAttentionLayer() throws UnsupportedKerasConfigurationException {
    }

    public KerasAttentionLayer(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig,false);
    }

    public KerasAttentionLayer(Map<String, Object> layerConfig, boolean enforceTrainingConfig) throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        this.useScale = Boolean.parseBoolean(innerConfig.getOrDefault(LAYER_USE_SCALE,"false").toString());
        this.dropOut = Double.parseDouble(innerConfig.getOrDefault(LAYER_DROP_OUT,"0.0").toString());
        this.inputNames = KerasLayerUtils.getInboundLayerNamesFromConfig(layerConfig, conf);
        String scoreMode = innerConfig.getOrDefault(LAYER_SCORE_MODE,LAYER_SCORE_MODE_DOT).toString();
        if(!scoreMode.equals(LAYER_SCORE_MODE_DOT) )
            throw new InvalidKerasConfigurationException("Invalid score mode " + scoreMode);
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
     * @throws InvalidKerasConfigurationException
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
        switch (inputType[0].getType()) {
            case FF:
                InputType.InputTypeFeedForward ff = (InputType.InputTypeFeedForward) inputType[0];
                this.getAttentionVertex().setNIn(ff.getSize());
                this.getAttentionVertex().setNOut(ff.getSize());
                break;
            case CNN:
                InputType.InputTypeConvolutional cnn = (InputType.InputTypeConvolutional) inputType[0];
                this.getAttentionVertex().setNIn(cnn.getChannels());
                this.getAttentionVertex().setNOut(cnn.getChannels());
                break;
            case RNN:
                InputType.InputTypeRecurrent rnn = (InputType.InputTypeRecurrent) inputType[0];
                this.getAttentionVertex().setNIn(rnn.getSize());
                this.getAttentionVertex().setNOut(rnn.getSize());
                break;
            case CNN3D:
            case CNNFlat:
                throw new InvalidKerasConfigurationException("Unsupported input type for attention layer: " + inputType[0].getType());
        }

        if (preprocessor != null) {
            return this.getAttentionVertex().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }

        return this.getAttentionVertex().getOutputType(-1, inputType[0]);
    }
    private DotProductAttentionVertex getAttentionVertex() {
        return (DotProductAttentionVertex) vertex;
    }


}
