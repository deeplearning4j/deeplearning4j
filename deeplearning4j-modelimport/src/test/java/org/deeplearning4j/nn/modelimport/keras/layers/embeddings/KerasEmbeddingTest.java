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
package org.deeplearning4j.nn.modelimport.keras.layers.embeddings;

import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasEmbeddingTest {

    private final String LAYER_NAME = "embedding_layer";
    private final String INIT_KERAS = "glorot_normal";
    private final int[] INPUT_SHAPE = new int[]{100, 20};

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testEmbeddingLayer() throws Exception {
        buildEmbeddingLayer(conf1, keras1);
        buildEmbeddingLayer(conf2, keras2);
    }


    void buildEmbeddingLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_EMBEDDING());
        Map<String, Object> config = new HashMap<String, Object>();
        Integer inputDim = 10;
        Integer outputDim = 10;
        config.put(conf.getLAYER_FIELD_INPUT_DIM(), inputDim);
        config.put(conf.getLAYER_FIELD_OUTPUT_DIM(), outputDim);

        ArrayList inputShape = new ArrayList<Integer>() {{
            for (int i : INPUT_SHAPE) add(i);
        }};
        config.put(conf.getLAYER_FIELD_BATCH_INPUT_SHAPE(), inputShape);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_EMBEDDING_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_EMBEDDING_INIT(), init);
        }
        KerasEmbedding kerasEmbedding = new KerasEmbedding(layerConfig, false);
        assertEquals(kerasEmbedding.getNumParams(), 1);

        EmbeddingLayer layer = kerasEmbedding.getEmbeddingLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());

    }
}
