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

package org.deeplearning4j.nn.modelimport.keras.layers.convolution;

import org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasZeroPadding1D;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasZeroPadding1DTest {

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testZeroPadding1DLayer() throws Exception {
        Integer keras1 = 1;
        buildZeroPadding1DLayer(conf1, keras1);
        Integer keras2 = 2;
        buildZeroPadding1DLayer(conf2, keras2);
    }


    private void buildZeroPadding1DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_1D());
        Map<String, Object> config = new HashMap<>();
        String layerName = "zero_padding_1D_layer";
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        int zeroPadding = 2;
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), zeroPadding);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPadding1DLayer layer = new KerasZeroPadding1D(layerConfig).getZeroPadding1DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(zeroPadding, layer.getPadding()[0]);
    }

}
