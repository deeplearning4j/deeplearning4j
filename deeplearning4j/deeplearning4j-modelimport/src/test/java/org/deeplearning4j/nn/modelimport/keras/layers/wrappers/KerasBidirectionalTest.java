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

package org.deeplearning4j.nn.modelimport.keras.layers.wrappers;

import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasBidirectionalTest {

    private final String ACTIVATION_KERAS = "linear";
    private final String ACTIVATION_DL4J = "identity";
    private final String LAYER_NAME = "bidirectional_layer";
    private final String INIT_KERAS = "glorot_normal";
    private final WeightInit INIT_DL4J = WeightInit.XAVIER;
    private final double L1_REGULARIZATION = 0.01;
    private final double L2_REGULARIZATION = 0.02;
    private final double DROPOUT_KERAS = 0.3;
    private final double DROPOUT_DL4J = 1 - DROPOUT_KERAS;
    private final int N_OUT = 13;
    private final String mode = "sum";

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testLstmLayer() throws Exception {
        buildLstmLayer(conf1, keras1);
        buildLstmLayer(conf2, keras2);
    }

    private void buildLstmLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        String innerActivation = "hard_sigmoid";
        String lstmForgetBiasString = "one";

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_LSTM());
        Map<String, Object> lstmConfig = new HashMap<>();
        lstmConfig.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS); // keras linear -> dl4j identity
        lstmConfig.put(conf.getLAYER_FIELD_INNER_ACTIVATION(), innerActivation); // keras linear -> dl4j identity
        lstmConfig.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            lstmConfig.put(conf.getLAYER_FIELD_INNER_INIT(), INIT_KERAS);
            lstmConfig.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);

        } else {
            Map<String, Object> init = new HashMap<>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            lstmConfig.put(conf.getLAYER_FIELD_INNER_INIT(), init);
            lstmConfig.put(conf.getLAYER_FIELD_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        lstmConfig.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        lstmConfig.put(conf.getLAYER_FIELD_RETURN_SEQUENCES(), true);

        lstmConfig.put(conf.getLAYER_FIELD_DROPOUT_W(), DROPOUT_KERAS);
        lstmConfig.put(conf.getLAYER_FIELD_DROPOUT_U(), 0.0);
        lstmConfig.put(conf.getLAYER_FIELD_FORGET_BIAS_INIT(), lstmForgetBiasString);
        lstmConfig.put(conf.getLAYER_FIELD_OUTPUT_DIM(), N_OUT);
        lstmConfig.put(conf.getLAYER_FIELD_UNROLL(), true);

        Map<String, Object> innerRnnConfig = new HashMap<>();
        innerRnnConfig.put("class_name", "LSTM");
        innerRnnConfig.put("config", lstmConfig);

        Map<String, Object> innerConfig = new HashMap<>();
        innerConfig.put("merge_mode", mode);
        innerConfig.put("layer", innerRnnConfig);
        innerConfig.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);

        layerConfig.put("config", innerConfig);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        KerasBidirectional kerasBidirectional = new KerasBidirectional(layerConfig);
        Bidirectional layer = kerasBidirectional.getBidirectionalLayer();

        assertEquals(Bidirectional.Mode.ADD, layer.getMode());
        assertEquals(Activation.HARDSIGMOID.toString().toLowerCase(),
                ((LSTM) kerasBidirectional.getUnderlyingRecurrentLayer()).getGateActivationFn().toString());

    }
}
