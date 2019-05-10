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

import org.deeplearning4j.nn.conf.layers.Upsampling3D;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasUpsampling3D;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasUpsampling3DTest {

    private final String LAYER_NAME = "upsampling_3D_layer";
    private int[] size = new int[]{2, 2, 2};

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testUpsampling3DLayer() throws Exception {
        buildUpsampling3DLayer(conf1, keras1);
        buildUpsampling3DLayer(conf2, keras2);
    }


    private void buildUpsampling3DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_UPSAMPLING_3D());
        Map<String, Object> config = new HashMap<>();
        List<Integer> sizeList = new ArrayList<>();
        sizeList.add(size[0]);
        sizeList.add(size[1]);
        sizeList.add(size[2]);
        config.put(conf.getLAYER_FIELD_UPSAMPLING_3D_SIZE(), sizeList);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Upsampling3D layer = new KerasUpsampling3D(layerConfig).getUpsampling3DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(size[0], layer.getSize()[0]);
        assertEquals(size[1], layer.getSize()[1]);
        assertEquals(size[2], layer.getSize()[2]);
    }

}
