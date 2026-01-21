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

package org.eclipse.deeplearning4j.frameworkimport.keras.layers;

import org.deeplearning4j.nn.conf.layers.SeparableConvolution1D;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSeparableConvolution1D;
import org.deeplearning4j.nn.modelimport.keras.layers.normalization.KerasGroupNormalization;
import org.deeplearning4j.nn.modelimport.keras.layers.normalization.KerasUnitNormalization;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasEinsumDense;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for new Keras layer implementations.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@DisplayName("Keras New Layers Test")
class KerasNewLayersTest {

    private Keras2LayerConfiguration conf = new Keras2LayerConfiguration();

    // ==================== SeparableConvolution1D Tests ====================

    @Test
    @DisplayName("Test SeparableConvolution1D Layer Configuration")
    void testSeparableConvolution1DLayerConfiguration() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_1D());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), "sep_conv1d_test");
        config.put(conf.getLAYER_FIELD_NB_FILTER(), 32);
        config.put(conf.getLAYER_FIELD_KERNEL_SIZE(), Collections.singletonList(3));
        config.put(conf.getLAYER_FIELD_CONVOLUTION_STRIDES(), Collections.singletonList(1));
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), "valid");
        config.put(conf.getLAYER_FIELD_ACTIVATION(), "relu");
        config.put(conf.getLAYER_FIELD_USE_BIAS(), true);
        config.put(conf.getLAYER_FIELD_DEPTH_MULTIPLIER(), 1);
        config.put(conf.getLAYER_FIELD_DILATION_RATE(), Collections.singletonList(1));

        // Add initializers
        Map<String, Object> kernelInit = new HashMap<>();
        kernelInit.put("class_name", "GlorotUniform");
        kernelInit.put("config", new HashMap<>());
        config.put(conf.getLAYER_FIELD_DEPTH_WISE_INIT(), kernelInit);
        config.put(conf.getLAYER_FIELD_POINT_WISE_INIT(), kernelInit);

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasSeparableConvolution1D kerasLayer = new KerasSeparableConvolution1D(layerConfig, false);

        assertNotNull(kerasLayer);
        assertEquals("sep_conv1d_test", kerasLayer.getLayerName());

        SeparableConvolution1D layer = kerasLayer.getSeparableConvolution1DLayer();
        assertNotNull(layer);
        assertEquals(32, layer.getNOut());
    }

    // ==================== GroupNormalization Tests ====================

    @Test
    @DisplayName("Test GroupNormalization Layer Configuration")
    void testGroupNormalizationLayerConfiguration() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_GROUP_NORMALIZATION());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), "group_norm_test");
        config.put("groups", 8);
        config.put("axis", -1);
        config.put("epsilon", 1e-5);
        config.put("center", true);
        config.put("scale", true);

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasGroupNormalization kerasLayer = new KerasGroupNormalization(layerConfig, false);

        assertNotNull(kerasLayer);
        assertEquals("group_norm_test", kerasLayer.getLayerName());
    }

    // ==================== UnitNormalization Tests ====================

    @Test
    @DisplayName("Test UnitNormalization Layer Configuration")
    void testUnitNormalizationLayerConfiguration() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_UNIT_NORMALIZATION());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), "unit_norm_test");
        config.put("axis", -1);

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasUnitNormalization kerasLayer = new KerasUnitNormalization(layerConfig, false);

        assertNotNull(kerasLayer);
        assertEquals("unit_norm_test", kerasLayer.getLayerName());
    }

    // ==================== EinsumDense Tests ====================

    @Test
    @DisplayName("Test EinsumDense Layer Configuration")
    void testEinsumDenseLayerConfiguration() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), "einsum_dense_test");
        config.put("equation", "abc,cd->abd");

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "d");

        Map<String, Object> kernelInit = new HashMap<>();
        kernelInit.put("class_name", "GlorotUniform");
        kernelInit.put("config", new HashMap<>());
        config.put("kernel_initializer", kernelInit);
        config.put("bias_initializer", kernelInit);

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertNotNull(kerasLayer);
        assertEquals("einsum_dense_test", kerasLayer.getLayerName());
    }

    // ==================== Layer Registration Tests ====================

    @Test
    @DisplayName("Test Layer Class Names Are Defined")
    void testLayerClassNamesAreDefined() {
        // Verify key layer class names are properly defined
        // Note: Some layer class names are defined in parent class KerasLayerConfiguration
        KerasLayerConfiguration baseConf = new KerasLayerConfiguration();

        assertNotNull(baseConf.getLAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_1D());
        assertFalse(baseConf.getLAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_1D().isEmpty());

        assertNotNull(baseConf.getLAYER_CLASS_NAME_GROUP_NORMALIZATION());
        assertFalse(baseConf.getLAYER_CLASS_NAME_GROUP_NORMALIZATION().isEmpty());

        assertNotNull(baseConf.getLAYER_CLASS_NAME_UNIT_NORMALIZATION());
        assertFalse(baseConf.getLAYER_CLASS_NAME_UNIT_NORMALIZATION().isEmpty());

        assertNotNull(baseConf.getLAYER_CLASS_NAME_EINSUM_DENSE());
        assertFalse(baseConf.getLAYER_CLASS_NAME_EINSUM_DENSE().isEmpty());

        assertNotNull(baseConf.getLAYER_CLASS_NAME_DECONVOLUTION_1D());
        assertFalse(baseConf.getLAYER_CLASS_NAME_DECONVOLUTION_1D().isEmpty());
    }

    @Test
    @DisplayName("Test Layer Class Name Values")
    void testLayerClassNameValues() {
        KerasLayerConfiguration baseConf = new KerasLayerConfiguration();

        assertEquals("SeparableConv1D", baseConf.getLAYER_CLASS_NAME_SEPARABLE_CONVOLUTION_1D());
        assertEquals("GroupNormalization", baseConf.getLAYER_CLASS_NAME_GROUP_NORMALIZATION());
        assertEquals("UnitNormalization", baseConf.getLAYER_CLASS_NAME_UNIT_NORMALIZATION());
        assertEquals("EinsumDense", baseConf.getLAYER_CLASS_NAME_EINSUM_DENSE());
        assertEquals("Conv1DTranspose", baseConf.getLAYER_CLASS_NAME_DECONVOLUTION_1D());
    }
}
