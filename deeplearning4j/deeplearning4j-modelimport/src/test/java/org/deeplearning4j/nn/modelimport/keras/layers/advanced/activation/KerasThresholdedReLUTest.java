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

import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasThresholdedReLU;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasThresholdedReLUTest {

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testThresholdedReLULayer() throws Exception {
        Integer keras1 = 1;
        buildThresholdedReLULayer(conf1, keras1);
        Integer keras2 = 2;
        buildThresholdedReLULayer(conf2, keras2);
    }

    private void buildThresholdedReLULayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {

        double theta = 0.5;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_THRESHOLDED_RELU());
        Map<String, Object> config = new HashMap<>();
        String LAYER_FIELD_THRESHOLD = "theta";
        config.put(LAYER_FIELD_THRESHOLD, theta);
        String layerName = "thresholded_relu";
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ActivationLayer layer = new KerasThresholdedReLU(layerConfig).getActivationLayer();
        assertEquals("thresholdedrelu(theta=0.5)", layer.getActivationFn().toString());
        assertEquals(layerName, layer.getLayerName());
    }
}
