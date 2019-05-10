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

import org.deeplearning4j.nn.conf.layers.ZeroPadding3DLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasZeroPadding2D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasZeroPadding3D;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasZeroPadding3DTest {

    private final String LAYER_NAME = "zero_padding_3D_layer";
    private final int[] ZERO_PADDING = new int[]{2, 3, 4};

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testZeroPadding3DLayer() throws Exception {
        Integer keras1 = 1;
        buildZeroPadding3DLayer(conf1, keras1);
        Integer keras2 = 2;
        buildZeroPadding3DLayer(conf2, keras2);
        buildZeroPaddingSingleDim3DLayer(conf1, keras1);
        buildZeroPaddingSingleDim3DLayer(conf2, keras2);
    }


    private void buildZeroPadding3DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_3D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        ArrayList padding = new ArrayList<Integer>() {{
            for (int i : ZERO_PADDING) add(i);
        }};
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), padding);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPadding3DLayer layer = new KerasZeroPadding3D(layerConfig).getZeroPadding3DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(ZERO_PADDING[0], layer.getPadding()[0]);
        assertEquals(ZERO_PADDING[0], layer.getPadding()[1]);
        assertEquals(ZERO_PADDING[1], layer.getPadding()[2]);
        assertEquals(ZERO_PADDING[1], layer.getPadding()[3]);
        assertEquals(ZERO_PADDING[2], layer.getPadding()[4]);
        assertEquals(ZERO_PADDING[2], layer.getPadding()[5]);

    }

    private void buildZeroPaddingSingleDim3DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_3D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), ZERO_PADDING[0]);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPadding3DLayer layer = new KerasZeroPadding3D(layerConfig).getZeroPadding3DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(ZERO_PADDING[0], layer.getPadding()[0]);
        assertEquals(ZERO_PADDING[0], layer.getPadding()[1]);
    }
}
