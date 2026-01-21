/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.frameworkimport.keras.layers.normalization;

import org.deeplearning4j.nn.modelimport.keras.layers.normalization.KerasLayerNormalization;
import org.deeplearning4j.nn.conf.layers.LayerNormalization;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;

/**
 * Tests for KerasLayerNormalization layer import.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@DisplayName("Keras Layer Normalization Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class KerasLayerNormalizationTest extends BaseDL4JTest {

    private final String LAYER_NAME = "layer_norm_layer";

    private Integer keras2 = 2;

    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    @DisplayName("Test LayerNormalization Layer")
    void testLayerNormalizationLayer() throws Exception {
        buildLayerNormalizationLayer(conf2, keras2);
    }

    private void buildLayerNormalizationLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        double epsilon = 1E-5;
        boolean center = true;
        boolean scale = true;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_LAYER_NORMALIZATION());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("epsilon", epsilon);
        config.put("center", center);
        config.put("scale", scale);
        config.put("axis", -1);

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        KerasLayerNormalization kerasLayer = new KerasLayerNormalization(layerConfig);
        LayerNormalization layer = kerasLayer.getLayerNormalizationLayer();

        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(epsilon, layer.getEpsilon(), 0.0);
        assertTrue(layer.isCenter());
        assertTrue(layer.isScale());
    }

    @Test
    @DisplayName("Test LayerNormalization Without Scale")
    void testLayerNormalizationWithoutScale() throws Exception {
        double epsilon = 1E-6;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_LAYER_NORMALIZATION());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("epsilon", epsilon);
        config.put("center", true);
        config.put("scale", false);
        config.put("axis", -1);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), keras2);

        KerasLayerNormalization kerasLayer = new KerasLayerNormalization(layerConfig);
        LayerNormalization layer = kerasLayer.getLayerNormalizationLayer();

        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(epsilon, layer.getEpsilon(), 0.0);
        assertTrue(layer.isCenter());
        assertTrue(!layer.isScale());
    }

    @Test
    @DisplayName("Test Set Weights")
    void testSetWeights() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_LAYER_NORMALIZATION());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("epsilon", 1e-5);
        config.put("center", true);
        config.put("scale", true);
        config.put("axis", -1);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), keras2);

        KerasLayerNormalization kerasLayer = new KerasLayerNormalization(layerConfig);

        Map<String, INDArray> weights = new HashMap<>();
        weights.put("gamma", Nd4j.ones(64));
        weights.put("beta", Nd4j.zeros(64));

        kerasLayer.setWeights(weights);

        assertEquals(2, kerasLayer.getWeights().size());
        assertTrue(kerasLayer.getWeights().containsKey("gamma"));
        assertTrue(kerasLayer.getWeights().containsKey("beta"));
    }

    @Test
    @DisplayName("Test Get Num Params")
    void testGetNumParams() throws Exception {
        // With both scale and center
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_LAYER_NORMALIZATION());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("epsilon", 1e-5);
        config.put("center", true);
        config.put("scale", true);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), keras2);

        KerasLayerNormalization kerasLayer = new KerasLayerNormalization(layerConfig);
        assertEquals(2, kerasLayer.getNumParams());

        // Without scale
        config.put("scale", false);
        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);

        KerasLayerNormalization kerasLayer2 = new KerasLayerNormalization(layerConfig);
        assertEquals(1, kerasLayer2.getNumParams());

        // Without center
        config.put("scale", true);
        config.put("center", false);
        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);

        KerasLayerNormalization kerasLayer3 = new KerasLayerNormalization(layerConfig);
        assertEquals(1, kerasLayer3.getNumParams());

        // Without both
        config.put("scale", false);
        config.put("center", false);
        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);

        KerasLayerNormalization kerasLayer4 = new KerasLayerNormalization(layerConfig);
        assertEquals(0, kerasLayer4.getNumParams());
    }
}
