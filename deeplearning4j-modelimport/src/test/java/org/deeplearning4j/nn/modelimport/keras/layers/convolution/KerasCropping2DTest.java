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

import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasCropping2D;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasCropping2DTest {

    private final String LAYER_NAME = "cropping_2D_layer";
    private final int[] CROPPING = new int[]{2, 3};

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testCropping2DLayer() throws Exception {
        Integer keras1 = 1;
        buildCropping2DLayer(conf1, keras1);
        Integer keras2 = 2;
        buildCropping2DLayer(conf2, keras2);
        buildCroppingSingleDim2DLayer(conf1, keras1);
        buildCroppingSingleDim2DLayer(conf2, keras2);
    }


    private void buildCropping2DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CROPPING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        ArrayList padding = new ArrayList<Integer>() {{
            for (int i : CROPPING) add(i);
        }};
        config.put(conf.getLAYER_FIELD_CROPPING(), padding);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Cropping2D layer = new KerasCropping2D(layerConfig).getCropping2DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(CROPPING[0], layer.getCropping()[0]);
        assertEquals(CROPPING[0], layer.getCropping()[1]);
        assertEquals(CROPPING[1], layer.getCropping()[2]);
        assertEquals(CROPPING[1], layer.getCropping()[3]);

    }

    private void buildCroppingSingleDim2DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CROPPING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put(conf.getLAYER_FIELD_CROPPING(), CROPPING[0]);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Cropping2D layer = new KerasCropping2D(layerConfig).getCropping2DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(CROPPING[0], layer.getCropping()[0]);
        assertEquals(CROPPING[0], layer.getCropping()[1]);
    }
}
