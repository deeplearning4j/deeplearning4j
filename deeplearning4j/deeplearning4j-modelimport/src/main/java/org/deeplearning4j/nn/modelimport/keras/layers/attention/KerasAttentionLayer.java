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
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Docs from https://keras.io/api/layers/attention_layers/attention/
 * @author Adam Gibson
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasAttentionLayer extends KerasLayer {

    private final int NUM_TRAINABLE_PARAMS = 2;
    private boolean useScale;
    private double dropOut;
    private String scoreMode;
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
        super(layerConfig);
    }

    public KerasAttentionLayer(Map<String, Object> layerConfig, boolean enforceTrainingConfig) throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);


    }

}
