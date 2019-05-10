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

import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasDropoutTest {

    String LAYER_NAME = "dropout";
    private final double DROPOUT_KERAS = 0.3;
    private final double DROPOUT_DL4J = 1 - DROPOUT_KERAS;

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();


    @Test
    public void testDropoutLayer() throws Exception {
        buildDropoutLayer(conf1, keras1);
        buildDropoutLayer(conf2, keras2);
    }


    private void buildDropoutLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DROPOUT());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        DropoutLayer layer = new KerasDropout(layerConfig).getDropoutLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(new Dropout(DROPOUT_DL4J), layer.getIDropout());
    }


}
