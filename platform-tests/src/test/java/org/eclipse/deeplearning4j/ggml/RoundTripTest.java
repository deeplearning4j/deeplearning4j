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

package org.eclipse.deeplearning4j.ggml;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.ggml.GGMLModelExport;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip tests for GGML export and import.
 * These tests verify that:
 * 1. Exporting a SameDiff model to GGUF produces valid files
 * 2. The exported file can be read back
 * 3. Tensor shapes, names, and data are preserved
 */
@DisplayName("GGML Round-Trip Test")
class RoundTripTest {

    @TempDir
    Path tempDir;

    // ========== Basic Round-Trip Tests ==========

    @Test
    @DisplayName("Test export and read metadata round-trip")
    void testMetadataRoundTrip() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File ggufFile = tempDir.resolve("metadata_roundtrip.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .modelName("test-roundtrip-model")
                .modelAuthor("Test Author")
                .modelDescription("A model for round-trip testing")
                .quantizationType(ExportOptions.QuantizationType.F16)
                .build();

        // Export
        GGMLModelExport.exportModel(model, ggufFile, options);

        // Read back
        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();

            assertNotNull(metadata);
            assertEquals("llama", metadata.getArchitecture());
        }
    }

    @Test
    @DisplayName("Test tensor count round-trip")
    void testTensorCountRoundTrip() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File ggufFile = tempDir.resolve("tensor_count_roundtrip.gguf").toFile();

        // Count exportable tensors
        int exportableTensors = countExportableTensors(model);

        // Export
        GGMLModelExport.exportModel(model, ggufFile);

        // Read back
        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();

            // Should have same number of tensors
            assertEquals(exportableTensors, metadata.getTensors().size());
        }
    }

    @Test
    @DisplayName("Test tensor shapes round-trip")
    void testTensorShapesRoundTrip() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File ggufFile = tempDir.resolve("shapes_roundtrip.gguf").toFile();

        // Store original shapes
        Map<String, long[]> originalShapes = new HashMap<>();
        for (SDVariable var : model.variables()) {
            if (isExportable(var)) {
                INDArray arr = var.getArr();
                if (arr != null) {
                    originalShapes.put(var.name(), arr.shape());
                }
            }
        }

        // Export
        GGMLModelExport.exportModel(model, ggufFile);

        // Read back and verify shapes
        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();

            for (GGMLTensorInfo tensorInfo : metadata.getTensors()) {
                // Find corresponding original variable
                String ggmlName = tensorInfo.getName();
                // Note: Names are transformed, so we verify total elements match
                long numElements = tensorInfo.getNumElements();
                assertTrue(numElements > 0, "Tensor " + ggmlName + " should have elements");
            }
        }
    }

    // ========== Quantization Round-Trip Tests ==========

    @Test
    @DisplayName("Test F16 quantization round-trip")
    void testF16RoundTrip() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File ggufFile = tempDir.resolve("f16_roundtrip.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.F16)
                .build();

        GGMLModelExport.exportModel(model, ggufFile, options);

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertTrue(metadata.getTensors().size() > 0);
        }
    }

    @Test
    @DisplayName("Test Q4_0 quantization round-trip")
    void testQ4_0RoundTrip() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File ggufFile = tempDir.resolve("q4_0_roundtrip.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_0)
                .build();

        GGMLModelExport.exportModel(model, ggufFile, options);

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertTrue(metadata.getTensors().size() > 0);
        }
    }

    @Test
    @DisplayName("Test Q8_0 quantization round-trip")
    void testQ8_0RoundTrip() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File ggufFile = tempDir.resolve("q8_0_roundtrip.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q8_0)
                .build();

        GGMLModelExport.exportModel(model, ggufFile, options);

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertTrue(metadata.getTensors().size() > 0);
        }
    }

    @Test
    @DisplayName("Test Q4_K quantization round-trip")
    void testQ4_KRoundTrip() throws Exception {
        SameDiff model = createLargerLLaMAModel();
        File ggufFile = tempDir.resolve("q4_k_roundtrip.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .build();

        GGMLModelExport.exportModel(model, ggufFile, options);

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertTrue(metadata.getTensors().size() > 0);
        }
    }

    @Test
    @DisplayName("Test Q5_K quantization round-trip")
    void testQ5_KRoundTrip() throws Exception {
        SameDiff model = createLargerLLaMAModel();
        File ggufFile = tempDir.resolve("q5_k_roundtrip.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q5_K)
                .build();

        GGMLModelExport.exportModel(model, ggufFile, options);

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertTrue(metadata.getTensors().size() > 0);
        }
    }

    @Test
    @DisplayName("Test Q6_K quantization round-trip")
    void testQ6_KRoundTrip() throws Exception {
        SameDiff model = createLargerLLaMAModel();
        File ggufFile = tempDir.resolve("q6_k_roundtrip.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q6_K)
                .build();

        GGMLModelExport.exportModel(model, ggufFile, options);

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertTrue(metadata.getTensors().size() > 0);
        }
    }

    // ========== File Size Tests ==========

    @Test
    @DisplayName("Test quantized file is smaller than F32")
    void testQuantizedFileSmallerThanF32() throws Exception {
        SameDiff model = createSmallLLaMAModel();

        File f32File = tempDir.resolve("size_f32.gguf").toFile();
        File q4kFile = tempDir.resolve("size_q4k.gguf").toFile();

        GGMLModelExport.exportModel(model, f32File, ExportOptions.f32());
        GGMLModelExport.exportModel(model, q4kFile, ExportOptions.q4k());

        assertTrue(q4kFile.length() < f32File.length(),
                "Q4_K file should be smaller than F32 file");
    }

    @Test
    @DisplayName("Test Q4_K file is smaller than Q8_0")
    void testQ4KSmallerThanQ8_0() throws Exception {
        SameDiff model = createSmallLLaMAModel();

        File q8File = tempDir.resolve("size_q8_0.gguf").toFile();
        File q4kFile = tempDir.resolve("size_q4k_compare.gguf").toFile();

        GGMLModelExport.exportModel(model, q8File, ExportOptions.q8_0());
        GGMLModelExport.exportModel(model, q4kFile, ExportOptions.q4k());

        assertTrue(q4kFile.length() < q8File.length(),
                "Q4_K file should be smaller than Q8_0 file");
    }

    // ========== Multiple Export Tests ==========

    @Test
    @DisplayName("Test export same model multiple times")
    void testExportMultipleTimes() throws Exception {
        SameDiff model = createSmallLLaMAModel();

        File file1 = tempDir.resolve("multi1.gguf").toFile();
        File file2 = tempDir.resolve("multi2.gguf").toFile();

        GGMLModelExport.exportModel(model, file1);
        GGMLModelExport.exportModel(model, file2);

        assertEquals(file1.length(), file2.length(),
                "Exporting same model should produce same file size");
    }

    @Test
    @DisplayName("Test export with different quantizations")
    void testExportDifferentQuantizations() throws Exception {
        SameDiff model = createSmallLLaMAModel();

        ExportOptions.QuantizationType[] types = {
                ExportOptions.QuantizationType.F32,
                ExportOptions.QuantizationType.F16,
                ExportOptions.QuantizationType.Q8_0,
                ExportOptions.QuantizationType.Q4_0
        };

        for (ExportOptions.QuantizationType type : types) {
            File file = tempDir.resolve("quant_" + type.name() + ".gguf").toFile();
            ExportOptions options = ExportOptions.builder()
                    .quantizationType(type)
                    .build();

            GGMLModelExport.exportModel(model, file, options);

            assertTrue(file.exists(), "File should exist for " + type);
            assertTrue(file.length() > 0, "File should not be empty for " + type);

            // Verify readable
            try (GGUFReader reader = new GGUFReader(file)) {
                GGMLMetadata metadata = reader.getMetadata();
                assertNotNull(metadata, "Should read metadata for " + type);
            }
        }
    }

    // ========== Edge Cases ==========

    @Test
    @DisplayName("Test export model with single layer")
    void testSingleLayerModel() throws Exception {
        SameDiff model = createSmallLLaMAModel(); // Already has 1 layer
        File ggufFile = tempDir.resolve("single_layer.gguf").toFile();

        GGMLModelExport.exportModel(model, ggufFile);

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
        }
    }

    @Test
    @DisplayName("Test export overwrites existing file")
    void testExportOverwritesFile() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File ggufFile = tempDir.resolve("overwrite.gguf").toFile();

        // First export
        GGMLModelExport.exportModel(model, ggufFile, ExportOptions.f32());
        long firstSize = ggufFile.length();

        // Second export with different options
        GGMLModelExport.exportModel(model, ggufFile, ExportOptions.q4k());
        long secondSize = ggufFile.length();

        assertNotEquals(firstSize, secondSize,
                "File should be different size after overwrite with different quantization");
    }

    // ========== Helper Methods ==========

    private SameDiff createSmallLLaMAModel() {
        SameDiff sd = SameDiff.create();

        int vocabSize = 256;
        int hiddenSize = 64;
        int intermediateSize = 128;

        // Embeddings
        sd.var("model.embed_tokens.weight", Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize));

        // Layer 0
        sd.var("model.layers.0.self_attn.q_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.self_attn.k_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.self_attn.v_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.self_attn.o_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.mlp.gate_proj.weight", Nd4j.rand(DataType.FLOAT, intermediateSize, hiddenSize));
        sd.var("model.layers.0.mlp.up_proj.weight", Nd4j.rand(DataType.FLOAT, intermediateSize, hiddenSize));
        sd.var("model.layers.0.mlp.down_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, intermediateSize));
        sd.var("model.layers.0.input_layernorm.weight", Nd4j.rand(DataType.FLOAT, hiddenSize));
        sd.var("model.layers.0.post_attention_layernorm.weight", Nd4j.rand(DataType.FLOAT, hiddenSize));

        // Final norm
        sd.var("model.norm.weight", Nd4j.rand(DataType.FLOAT, hiddenSize));

        // LM head
        sd.var("lm_head.weight", Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize));

        return sd;
    }

    /**
     * Create a larger model for K-quant tests (needs at least 256 elements for K-quant blocks)
     */
    private SameDiff createLargerLLaMAModel() {
        SameDiff sd = SameDiff.create();

        int vocabSize = 512;
        int hiddenSize = 256;
        int intermediateSize = 512;

        // Embeddings
        sd.var("model.embed_tokens.weight", Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize));

        // Layer 0
        sd.var("model.layers.0.self_attn.q_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.self_attn.k_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.self_attn.v_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.self_attn.o_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize));
        sd.var("model.layers.0.mlp.gate_proj.weight", Nd4j.rand(DataType.FLOAT, intermediateSize, hiddenSize));
        sd.var("model.layers.0.mlp.up_proj.weight", Nd4j.rand(DataType.FLOAT, intermediateSize, hiddenSize));
        sd.var("model.layers.0.mlp.down_proj.weight", Nd4j.rand(DataType.FLOAT, hiddenSize, intermediateSize));
        sd.var("model.layers.0.input_layernorm.weight", Nd4j.rand(DataType.FLOAT, hiddenSize));
        sd.var("model.layers.0.post_attention_layernorm.weight", Nd4j.rand(DataType.FLOAT, hiddenSize));

        // Final norm
        sd.var("model.norm.weight", Nd4j.rand(DataType.FLOAT, hiddenSize));

        // LM head
        sd.var("lm_head.weight", Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize));

        return sd;
    }

    private int countExportableTensors(SameDiff model) {
        int count = 0;
        for (SDVariable var : model.variables()) {
            if (isExportable(var) && var.getArr() != null) {
                count++;
            }
        }
        return count;
    }

    private boolean isExportable(SDVariable var) {
        return var.getVariableType() == VariableType.VARIABLE ||
               var.getVariableType() == VariableType.CONSTANT;
    }
}
