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

package org.eclipse.deeplearning4j.ggml.export;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLFormat;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ExportOptions Test")
class ExportOptionsTest {

    // ========== Default Values Tests ==========

    @Test
    @DisplayName("Test default builder values")
    void testDefaultBuilderValues() {
        ExportOptions options = ExportOptions.builder().build();

        assertEquals(ExportOptions.QuantizationType.F16, options.getQuantizationType());
        assertEquals(GGMLFormat.GGUF, options.getOutputFormat());
        assertEquals(3, options.getGgufVersion());
        assertEquals(32, options.getAlignment());
        assertTrue(options.isValidateBeforeExport());
        assertTrue(options.isIncludeTokenizer());
        assertFalse(options.isIncludeStatistics());
        assertNull(options.getArchitectureHint());
        assertNull(options.getModelName());
        assertNull(options.getModelAuthor());
        assertNull(options.getModelDescription());
        assertNull(options.getLayerQuantizationOverrides());
        assertNull(options.getTensorNameMapping());
    }

    // ========== QuantizationType Tests ==========

    @Test
    @DisplayName("Test QuantizationType to GGMLDataType mapping")
    void testQuantizationTypeMapping() {
        assertEquals(GGMLDataType.GGML_TYPE_F32,
                ExportOptions.QuantizationType.F32.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_F16,
                ExportOptions.QuantizationType.F16.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_BF16,
                ExportOptions.QuantizationType.BF16.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q4_0,
                ExportOptions.QuantizationType.Q4_0.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q4_1,
                ExportOptions.QuantizationType.Q4_1.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q5_0,
                ExportOptions.QuantizationType.Q5_0.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q5_1,
                ExportOptions.QuantizationType.Q5_1.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q8_0,
                ExportOptions.QuantizationType.Q8_0.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q4_K,
                ExportOptions.QuantizationType.Q4_K.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q5_K,
                ExportOptions.QuantizationType.Q5_K.toGGMLDataType());
        assertEquals(GGMLDataType.GGML_TYPE_Q6_K,
                ExportOptions.QuantizationType.Q6_K.toGGMLDataType());
    }

    @Test
    @DisplayName("Test all QuantizationType values exist")
    void testAllQuantizationTypes() {
        ExportOptions.QuantizationType[] types = ExportOptions.QuantizationType.values();
        assertEquals(11, types.length);

        // Verify all have valid GGML data type mapping
        for (ExportOptions.QuantizationType type : types) {
            assertNotNull(type.toGGMLDataType());
        }
    }

    // ========== Builder Tests ==========

    @Test
    @DisplayName("Test builder with quantization type")
    void testBuilderQuantizationType() {
        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .build();

        assertEquals(ExportOptions.QuantizationType.Q4_K, options.getQuantizationType());
    }

    @Test
    @DisplayName("Test builder with output format")
    void testBuilderOutputFormat() {
        ExportOptions options = ExportOptions.builder()
                .outputFormat(GGMLFormat.GGML)
                .build();

        assertEquals(GGMLFormat.GGML, options.getOutputFormat());
    }

    @Test
    @DisplayName("Test builder with GGUF version")
    void testBuilderGgufVersion() {
        ExportOptions options = ExportOptions.builder()
                .ggufVersion(2)
                .build();

        assertEquals(2, options.getGgufVersion());
    }

    @Test
    @DisplayName("Test builder with alignment")
    void testBuilderAlignment() {
        ExportOptions options = ExportOptions.builder()
                .alignment(64)
                .build();

        assertEquals(64, options.getAlignment());
    }

    @Test
    @DisplayName("Test builder with architecture hint")
    void testBuilderArchitectureHint() {
        ExportOptions options = ExportOptions.builder()
                .architectureHint("llama")
                .build();

        assertEquals("llama", options.getArchitectureHint());
    }

    @Test
    @DisplayName("Test builder with model metadata")
    void testBuilderModelMetadata() {
        ExportOptions options = ExportOptions.builder()
                .modelName("test-llama")
                .modelAuthor("Test Author")
                .modelDescription("A test model")
                .build();

        assertEquals("test-llama", options.getModelName());
        assertEquals("Test Author", options.getModelAuthor());
        assertEquals("A test model", options.getModelDescription());
    }

    @Test
    @DisplayName("Test builder with layer quantization overrides")
    void testBuilderLayerOverrides() {
        Map<String, ExportOptions.QuantizationType> overrides = new HashMap<>();
        overrides.put(".*embed.*", ExportOptions.QuantizationType.F16);
        overrides.put(".*norm.*", ExportOptions.QuantizationType.F32);

        ExportOptions options = ExportOptions.builder()
                .layerQuantizationOverrides(overrides)
                .build();

        assertNotNull(options.getLayerQuantizationOverrides());
        assertEquals(2, options.getLayerQuantizationOverrides().size());
        assertEquals(ExportOptions.QuantizationType.F16,
                options.getLayerQuantizationOverrides().get(".*embed.*"));
    }

    @Test
    @DisplayName("Test builder with tensor name mapping")
    void testBuilderTensorNameMapping() {
        Map<String, String> mapping = new HashMap<>();
        mapping.put("model.embed_tokens.weight", "token_embd.weight");
        mapping.put("lm_head.weight", "output.weight");

        ExportOptions options = ExportOptions.builder()
                .tensorNameMapping(mapping)
                .build();

        assertNotNull(options.getTensorNameMapping());
        assertEquals(2, options.getTensorNameMapping().size());
        assertEquals("token_embd.weight",
                options.getTensorNameMapping().get("model.embed_tokens.weight"));
    }

    @Test
    @DisplayName("Test builder with validation flag")
    void testBuilderValidationFlag() {
        ExportOptions options = ExportOptions.builder()
                .validateBeforeExport(false)
                .build();

        assertFalse(options.isValidateBeforeExport());
    }

    @Test
    @DisplayName("Test builder with tokenizer flag")
    void testBuilderTokenizerFlag() {
        ExportOptions options = ExportOptions.builder()
                .includeTokenizer(false)
                .build();

        assertFalse(options.isIncludeTokenizer());
    }

    @Test
    @DisplayName("Test builder with statistics flag")
    void testBuilderStatisticsFlag() {
        ExportOptions options = ExportOptions.builder()
                .includeStatistics(true)
                .build();

        assertTrue(options.isIncludeStatistics());
    }

    // ========== Factory Method Tests ==========

    @Test
    @DisplayName("Test f16() factory method")
    void testF16FactoryMethod() {
        ExportOptions options = ExportOptions.f16();

        assertEquals(ExportOptions.QuantizationType.F16, options.getQuantizationType());
        assertEquals(GGMLFormat.GGUF, options.getOutputFormat());
    }

    @Test
    @DisplayName("Test f32() factory method")
    void testF32FactoryMethod() {
        ExportOptions options = ExportOptions.f32();

        assertEquals(ExportOptions.QuantizationType.F32, options.getQuantizationType());
    }

    @Test
    @DisplayName("Test q4k() factory method")
    void testQ4kFactoryMethod() {
        ExportOptions options = ExportOptions.q4k();

        assertEquals(ExportOptions.QuantizationType.Q4_K, options.getQuantizationType());
    }

    @Test
    @DisplayName("Test q8_0() factory method")
    void testQ8_0FactoryMethod() {
        ExportOptions options = ExportOptions.q8_0();

        assertEquals(ExportOptions.QuantizationType.Q8_0, options.getQuantizationType());
    }

    @Test
    @DisplayName("Test q5k() factory method")
    void testQ5kFactoryMethod() {
        ExportOptions options = ExportOptions.q5k();

        assertEquals(ExportOptions.QuantizationType.Q5_K, options.getQuantizationType());
    }

    @Test
    @DisplayName("Test q6k() factory method")
    void testQ6kFactoryMethod() {
        ExportOptions options = ExportOptions.q6k();

        assertEquals(ExportOptions.QuantizationType.Q6_K, options.getQuantizationType());
    }

    @Test
    @DisplayName("Test legacyGGML() factory method")
    void testLegacyGGMLFactoryMethod() {
        ExportOptions options = ExportOptions.legacyGGML();

        assertEquals(GGMLFormat.GGML, options.getOutputFormat());
        assertEquals(ExportOptions.QuantizationType.F16, options.getQuantizationType());
    }

    // ========== Full Builder Test ==========

    @Test
    @DisplayName("Test builder with all options")
    void testFullBuilder() {
        Map<String, ExportOptions.QuantizationType> overrides = new HashMap<>();
        overrides.put(".*embed.*", ExportOptions.QuantizationType.F16);

        Map<String, String> mapping = new HashMap<>();
        mapping.put("var1", "tensor1");

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .outputFormat(GGMLFormat.GGUF)
                .ggufVersion(3)
                .alignment(64)
                .architectureHint("llama")
                .modelName("test-model")
                .modelAuthor("Test Author")
                .modelDescription("Test Description")
                .includeTokenizer(true)
                .layerQuantizationOverrides(overrides)
                .tensorNameMapping(mapping)
                .validateBeforeExport(true)
                .includeStatistics(true)
                .build();

        assertEquals(ExportOptions.QuantizationType.Q4_K, options.getQuantizationType());
        assertEquals(GGMLFormat.GGUF, options.getOutputFormat());
        assertEquals(3, options.getGgufVersion());
        assertEquals(64, options.getAlignment());
        assertEquals("llama", options.getArchitectureHint());
        assertEquals("test-model", options.getModelName());
        assertEquals("Test Author", options.getModelAuthor());
        assertEquals("Test Description", options.getModelDescription());
        assertTrue(options.isIncludeTokenizer());
        assertNotNull(options.getLayerQuantizationOverrides());
        assertNotNull(options.getTensorNameMapping());
        assertTrue(options.isValidateBeforeExport());
        assertTrue(options.isIncludeStatistics());
    }

    // ========== Lombok Generated Methods Tests ==========

    @Test
    @DisplayName("Test equals and hashCode")
    void testEqualsAndHashCode() {
        ExportOptions options1 = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .modelName("test")
                .build();

        ExportOptions options2 = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .modelName("test")
                .build();

        ExportOptions options3 = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q8_0)
                .modelName("test")
                .build();

        assertEquals(options1, options2);
        assertEquals(options1.hashCode(), options2.hashCode());
        assertNotEquals(options1, options3);
    }

    @Test
    @DisplayName("Test toString")
    void testToString() {
        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .modelName("test-model")
                .build();

        String str = options.toString();
        assertNotNull(str);
        assertTrue(str.contains("Q4_K"));
        assertTrue(str.contains("test-model"));
    }

    @Test
    @DisplayName("Test setters")
    void testSetters() {
        ExportOptions options = ExportOptions.builder().build();

        options.setQuantizationType(ExportOptions.QuantizationType.Q8_0);
        assertEquals(ExportOptions.QuantizationType.Q8_0, options.getQuantizationType());

        options.setModelName("new-name");
        assertEquals("new-name", options.getModelName());

        options.setAlignment(128);
        assertEquals(128, options.getAlignment());
    }
}
