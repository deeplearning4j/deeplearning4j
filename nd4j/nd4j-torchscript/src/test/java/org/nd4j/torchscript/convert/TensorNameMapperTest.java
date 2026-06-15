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

// TODO: Move to platform-tests/ per project convention — all tests must run from platform-tests

package org.nd4j.torchscript.convert;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("TensorNameMapper Test")
class TensorNameMapperTest {

    @Test
    @DisplayName("Test mapName with custom mappings")
    void testMapNameWithCustomMappings() {
        Map<String, String> mappings = new HashMap<>();
        mappings.put("layer1.weight", "custom_layer1_W");
        mappings.put("layer1.bias", "custom_layer1_b");

        assertEquals("custom_layer1_W", TensorNameMapper.mapName("layer1.weight", mappings));
        assertEquals("custom_layer1_b", TensorNameMapper.mapName("layer1.bias", mappings));
    }

    @Test
    @DisplayName("Test mapName falls back to normalization")
    void testMapNameFallback() {
        Map<String, String> mappings = new HashMap<>();
        mappings.put("other.weight", "other_W");

        // Not in mappings, should normalize
        assertEquals("layer1_weight", TensorNameMapper.mapName("layer1.weight", mappings));
    }

    @Test
    @DisplayName("Test mapName with null mappings")
    void testMapNameNullMappings() {
        assertEquals("layer1_weight", TensorNameMapper.mapName("layer1.weight", null));
    }

    @Test
    @DisplayName("Test normalizeName replaces dots")
    void testNormalizeNameReplacesDots() {
        assertEquals("layer1_0_conv1_weight", TensorNameMapper.normalizeName("layer1.0.conv1.weight"));
        assertEquals("encoder_layers_3_self_attn_out_proj_bias",
                TensorNameMapper.normalizeName("encoder.layers.3.self_attn.out_proj.bias"));
    }

    @Test
    @DisplayName("Test normalizeName handles special characters")
    void testNormalizeNameSpecialChars() {
        assertEquals("layer_1_conv_weight", TensorNameMapper.normalizeName("layer-1/conv.weight"));
        // Double colon becomes double underscore (each char replaced individually)
        assertEquals("module__layer_weight", TensorNameMapper.normalizeName("module::layer.weight"));
    }

    @Test
    @DisplayName("Test normalizeName handles leading digits")
    void testNormalizeNameLeadingDigits() {
        assertEquals("t_0_layer_weight", TensorNameMapper.normalizeName("0.layer.weight"));
        assertEquals("t_123_param", TensorNameMapper.normalizeName("123_param"));
    }

    @Test
    @DisplayName("Test normalizeName handles null")
    void testNormalizeNameNull() {
        assertEquals("unnamed", TensorNameMapper.normalizeName(null));
    }

    @Test
    @DisplayName("Test isWeightTensor")
    void testIsWeightTensor() {
        assertTrue(TensorNameMapper.isWeightTensor("layer1.weight"));
        assertTrue(TensorNameMapper.isWeightTensor("conv_weight"));
        assertTrue(TensorNameMapper.isWeightTensor("encoder.layers.0.weight"));
        assertFalse(TensorNameMapper.isWeightTensor("layer1.bias"));
        assertFalse(TensorNameMapper.isWeightTensor("running_mean"));
        assertFalse(TensorNameMapper.isWeightTensor("weight.layer1")); // weight not at end
    }

    @Test
    @DisplayName("Test isBiasTensor")
    void testIsBiasTensor() {
        assertTrue(TensorNameMapper.isBiasTensor("layer1.bias"));
        assertTrue(TensorNameMapper.isBiasTensor("conv_bias"));
        assertTrue(TensorNameMapper.isBiasTensor("encoder.layers.0.bias"));
        assertFalse(TensorNameMapper.isBiasTensor("layer1.weight"));
        assertFalse(TensorNameMapper.isBiasTensor("running_var"));
    }

    @Test
    @DisplayName("Test isBatchNormBuffer")
    void testIsBatchNormBuffer() {
        assertTrue(TensorNameMapper.isBatchNormBuffer("bn1.running_mean"));
        assertTrue(TensorNameMapper.isBatchNormBuffer("layer1.bn.running_var"));
        assertTrue(TensorNameMapper.isBatchNormBuffer("norm.num_batches_tracked"));
        assertTrue(TensorNameMapper.isBatchNormBuffer("RUNNING_MEAN"));
        assertFalse(TensorNameMapper.isBatchNormBuffer("bn1.weight"));
        assertFalse(TensorNameMapper.isBatchNormBuffer("bn1.bias"));
    }

    @Test
    @DisplayName("Test extractLayerName")
    void testExtractLayerName() {
        assertEquals("layer1.0.conv1", TensorNameMapper.extractLayerName("layer1.0.conv1.weight"));
        assertEquals("encoder.layers.3.self_attn",
                TensorNameMapper.extractLayerName("encoder.layers.3.self_attn.bias"));
        assertEquals("fc", TensorNameMapper.extractLayerName("fc.weight"));
        assertEquals("single", TensorNameMapper.extractLayerName("single"));
    }

    @Test
    @DisplayName("Test extractParamType")
    void testExtractParamType() {
        assertEquals("weight", TensorNameMapper.extractParamType("layer1.0.conv1.weight"));
        assertEquals("bias", TensorNameMapper.extractParamType("fc.bias"));
        assertEquals("running_mean", TensorNameMapper.extractParamType("bn1.running_mean"));
        assertEquals("single", TensorNameMapper.extractParamType("single"));
    }

    @Test
    @DisplayName("Test getWeightKey")
    void testGetWeightKey() {
        assertEquals("layer1_conv1_W", TensorNameMapper.getWeightKey("layer1.conv1"));
        assertEquals("fc_W", TensorNameMapper.getWeightKey("fc"));
    }

    @Test
    @DisplayName("Test getBiasKey")
    void testGetBiasKey() {
        assertEquals("layer1_conv1_b", TensorNameMapper.getBiasKey("layer1.conv1"));
        assertEquals("fc_b", TensorNameMapper.getBiasKey("fc"));
    }

    @Test
    @DisplayName("Test inferLayerType for convolutions")
    void testInferLayerTypeConv() {
        assertEquals("conv2d", TensorNameMapper.inferLayerType("layer1.conv.weight"));
        assertEquals("conv2d", TensorNameMapper.inferLayerType("conv2d_1.weight"));
        assertEquals("conv1d", TensorNameMapper.inferLayerType("conv1d.weight"));
        assertEquals("conv3d", TensorNameMapper.inferLayerType("conv3d.weight"));
    }

    @Test
    @DisplayName("Test inferLayerType for batch norm")
    void testInferLayerTypeBatchNorm() {
        assertEquals("batch_norm", TensorNameMapper.inferLayerType("bn1.weight"));
        assertEquals("batch_norm", TensorNameMapper.inferLayerType("batch_norm.weight"));
        assertEquals("batch_norm", TensorNameMapper.inferLayerType("layer1.batchnorm.running_mean"));
    }

    @Test
    @DisplayName("Test inferLayerType for linear layers")
    void testInferLayerTypeLinear() {
        assertEquals("linear", TensorNameMapper.inferLayerType("fc.weight"));
        assertEquals("linear", TensorNameMapper.inferLayerType("linear.weight"));
        assertEquals("linear", TensorNameMapper.inferLayerType("dense_1.weight"));
    }

    @Test
    @DisplayName("Test inferLayerType for recurrent layers")
    void testInferLayerTypeRecurrent() {
        assertEquals("lstm", TensorNameMapper.inferLayerType("lstm.weight_ih_l0"));
        assertEquals("gru", TensorNameMapper.inferLayerType("gru.weight_hh_l0"));
    }

    @Test
    @DisplayName("Test inferLayerType for embedding")
    void testInferLayerTypeEmbedding() {
        assertEquals("embedding", TensorNameMapper.inferLayerType("embedding.weight"));
        assertEquals("embedding", TensorNameMapper.inferLayerType("token_embedding.weight"));
    }

    @Test
    @DisplayName("Test inferLayerType for normalization layers")
    void testInferLayerTypeNorm() {
        assertEquals("layer_norm", TensorNameMapper.inferLayerType("layernorm.weight"));
        assertEquals("layer_norm", TensorNameMapper.inferLayerType("layer_norm.weight"));
        assertEquals("group_norm", TensorNameMapper.inferLayerType("groupnorm.weight"));
        assertEquals("group_norm", TensorNameMapper.inferLayerType("group_norm.weight"));
    }

    @Test
    @DisplayName("Test inferLayerType unknown")
    void testInferLayerTypeUnknown() {
        assertEquals("unknown", TensorNameMapper.inferLayerType("random_param"));
        assertEquals("unknown", TensorNameMapper.inferLayerType("some.other.tensor"));
    }
}
