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

import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;


/**
 * @author Max Pumperla
 */
public class KerasMaskingTest {


    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();


    @Test
    public void testMaskingLayer() throws Exception {
        Integer keras1 = 1;
        buildMaskingLayer(conf1, keras1);
        Integer keras2 = 2;
        buildMaskingLayer(conf2, keras2);
    }


    private void buildMaskingLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_MASKING());
        Map<String, Object> config = new HashMap<>();
        String LAYER_NAME = "masking";
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        double MASKING_VALUE = 1.0;
        config.put(conf.getLAYER_FIELD_MASK_VALUE(), MASKING_VALUE);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        MaskZeroLayer layer = new KerasMasking(layerConfig).getMaskingLayer();
        assert MASKING_VALUE == layer.getMaskingValue();
    }


}
