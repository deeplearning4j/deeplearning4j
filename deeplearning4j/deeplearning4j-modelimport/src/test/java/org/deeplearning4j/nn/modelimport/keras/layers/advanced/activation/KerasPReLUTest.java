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

package org.deeplearning4j.nn.modelimport.keras.layers.advanced.activation;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.PReLULayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasLeakyReLU;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasPReLU;
import org.deeplearning4j.nn.modelimport.keras.layers.local.KerasLocallyConnected2D;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasPReLUTest {

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    private final String INIT_KERAS = "glorot_normal";
    private final WeightInit INIT_DL4J = WeightInit.XAVIER;

    @Test
    public void testPReLULayer() throws Exception {
        Integer keras1 = 1;
        buildPReLULayer(conf1, keras1);
        Integer keras2 = 2;
        buildPReLULayer(conf2, keras2);
    }

    private void buildPReLULayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_LEAKY_RELU());
        Map<String, Object> config = new HashMap<>();
        String layerName = "prelu";
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        if (kerasVersion == 1) {
            config.put("alpha_initializer", INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put("alpha_initializer", init);
        }

        KerasPReLU kerasPReLU = new KerasPReLU(layerConfig);

        kerasPReLU.getOutputType(InputType.convolutional(5,4,3));

        PReLULayer layer = kerasPReLU.getPReLULayer();
        assertArrayEquals(layer.getInputShape(), new long[] {3, 5, 4});
        assertEquals(INIT_DL4J, layer.getWeightInit());

        assertEquals(layerName, layer.getLayerName());
    }
}
