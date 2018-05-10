/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.modelimport.keras.layers.convolution;

import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasZeroPadding2D;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasZeroPadding2DTest {

    private final String LAYER_NAME = "zero_padding_2D_layer";
    private final int[] ZERO_PADDING = new int[]{2, 3};

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testZeroPadding2DLayer() throws Exception {
        Integer keras1 = 1;
        buildZeroPadding2DLayer(conf1, keras1);
        Integer keras2 = 2;
        buildZeroPadding2DLayer(conf2, keras2);
        buildZeroPaddingSingleDim2DLayer(conf1, keras1);
        buildZeroPaddingSingleDim2DLayer(conf2, keras2);
    }


    private void buildZeroPadding2DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        ArrayList padding = new ArrayList<Integer>() {{
            for (int i : ZERO_PADDING) add(i);
        }};
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), padding);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPaddingLayer layer = new KerasZeroPadding2D(layerConfig).getZeroPadding2DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(ZERO_PADDING[0], layer.getPadding()[0]);
        assertEquals(ZERO_PADDING[0], layer.getPadding()[1]);
        assertEquals(ZERO_PADDING[1], layer.getPadding()[2]);
        assertEquals(ZERO_PADDING[1], layer.getPadding()[3]);

    }

    private void buildZeroPaddingSingleDim2DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), ZERO_PADDING[0]);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPaddingLayer layer = new KerasZeroPadding2D(layerConfig).getZeroPadding2DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(ZERO_PADDING[0], layer.getPadding()[0]);
        assertEquals(ZERO_PADDING[0], layer.getPadding()[1]);
    }
}
