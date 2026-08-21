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

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLExportException;
import org.nd4j.ggml.GGMLModelExport;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.format.GGUFWriter;
import org.nd4j.ggml.quantization.IQ4_NLDequantizer;
import org.nd4j.ggml.quantization.Q4_KDequantizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("GGMLModelExport Test")
@Tag(TagNames.FULL_CI)
class GGMLModelExportTest {

    @TempDir
    Path tempDir;

    // ========== Supported Types Tests ==========

    @Test
    @DisplayName("Test get supported quantization types")
    void testGetSupportedQuantizationTypes() {
        List<String> types = GGMLModelExport.getSupportedQuantizationTypes();

        assertNotNull(types);
        assertFalse(types.isEmpty());
        assertTrue(types.contains("F32"));
        assertTrue(types.contains("F16"));
        assertTrue(types.contains("Q4_0"));
        assertTrue(types.contains("Q8_0"));
        assertTrue(types.contains("Q4_K"));
    }

    @Test
    @DisplayName("Test get supported architectures")
    void testGetSupportedArchitectures() {
        Set<String> architectures = GGMLModelExport.getSupportedArchitectures();

        assertNotNull(architectures);
        assertFalse(architectures.isEmpty());
        assertTrue(architectures.contains("llama"));
    }

    // ========== Model Detection Tests ==========

    @Test
    @DisplayName("Test can export LLaMA model")
    void testCanExportLLaMAModel() {
        SameDiff model = createMinimalLLaMAModel();
        assertTrue(GGMLModelExport.canExport(model));
    }

    @Test
    @DisplayName("Test cannot export unknown model")
    void testCannotExportUnknownModel() {
        SameDiff model = SameDiff.create();
        model.var("unknown_variable", DataType.FLOAT, 100, 100);
        assertFalse(GGMLModelExport.canExport(model));
    }

    @Test
    @DisplayName("Test detect architecture for LLaMA model")
    void testDetectArchitectureLLaMA() {
        SameDiff model = createMinimalLLaMAModel();
        String arch = GGMLModelExport.detectArchitecture(model);

        assertNotNull(arch);
        assertEquals("llama", arch);
    }

    @Test
    @DisplayName("Test detect architecture for unknown model")
    void testDetectArchitectureUnknown() {
        SameDiff model = SameDiff.create();
        model.var("unknown_variable", DataType.FLOAT, 100, 100);

        String arch = GGMLModelExport.detectArchitecture(model);
        assertNull(arch);
    }

    // ========== Validation Tests ==========

    @Test
    @DisplayName("Test validate complete model returns no errors")
    void testValidateCompleteModel() {
        SameDiff model = createMinimalLLaMAModel();
        List<String> errors = GGMLModelExport.validateForExport(model);

        assertTrue(errors.isEmpty(), "Expected no errors: " + errors);
    }

    @Test
    @DisplayName("Test validate incomplete model returns errors")
    void testValidateIncompleteModel() {
        SameDiff model = SameDiff.create();
        // Missing required tensors
        model.var("model.layers.0.self_attn.q_proj.weight", DataType.FLOAT, 4096, 4096);

        List<String> errors = GGMLModelExport.validateForExport(model);

        // Should have errors about missing embeddings and norm
        assertFalse(errors.isEmpty());
    }

    @Test
    @DisplayName("Test validate unknown model")
    void testValidateUnknownModel() {
        SameDiff model = SameDiff.create();
        model.var("unknown_variable", DataType.FLOAT, 100, 100);

        List<String> errors = GGMLModelExport.validateForExport(model);

        assertFalse(errors.isEmpty());
        assertTrue(errors.stream().anyMatch(e -> e.contains("Could not detect")));
    }

    // ========== Export Tests ==========

    @Test
    @DisplayName("Test export model to file path string")
    void testExportModelToPathString() throws GGMLExportException {
        SameDiff model = createSmallLLaMAModel();
        String outputPath = tempDir.resolve("test_export.gguf").toString();

        GGMLModelExport.exportModel(model, outputPath);

        File outputFile = new File(outputPath);
        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    @Test
    @DisplayName("Test export model to file")
    void testExportModelToFile() throws GGMLExportException {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_export2.gguf").toFile();

        GGMLModelExport.exportModel(model, outputFile);

        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    @Test
    @DisplayName("Test export model with options")
    void testExportModelWithOptions() throws GGMLExportException {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_export_options.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.F16)
                .modelName("test-model")
                .modelAuthor("Test Author")
                .build();

        GGMLModelExport.exportModel(model, outputFile, options);

        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    @Test
    @DisplayName("Test export model with Q4_K quantization")
    void testExportModelQ4K() throws GGMLExportException {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_q4k.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .build();

        GGMLModelExport.exportModel(model, outputFile, options);

        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    @Test
    @DisplayName("Test export model with Q8_0 quantization")
    void testExportModelQ8_0() throws GGMLExportException {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_q8_0.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q8_0)
                .build();

        GGMLModelExport.exportModel(model, outputFile, options);

        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    @Test
    @DisplayName("Test export throws for unknown architecture")
    void testExportThrowsForUnknownArch() {
        SameDiff model = SameDiff.create();
        model.var("unknown_variable", DataType.FLOAT, 100, 100);
        File outputFile = tempDir.resolve("unknown.gguf").toFile();

        assertThrows(GGMLExportException.class, () -> {
            GGMLModelExport.exportModel(model, outputFile);
        });
    }

    @Test
    @DisplayName("Test export with architecture hint")
    void testExportWithArchitectureHint() throws GGMLExportException {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_hint.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .architectureHint("llama")
                .build();

        GGMLModelExport.exportModel(model, outputFile, options);

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test export with validation disabled")
    void testExportWithValidationDisabled() throws GGMLExportException {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_no_validate.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .validateBeforeExport(false)
                .build();

        GGMLModelExport.exportModel(model, outputFile, options);

        assertTrue(outputFile.exists());
    }

    // ========== Export File Structure Tests ==========

    @Test
    @DisplayName("Test exported file has GGUF magic")
    void testExportedFileHasGGUFMagic() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_magic.gguf").toFile();

        GGMLModelExport.exportModel(model, outputFile);

        // Read first 4 bytes
        byte[] magic = new byte[4];
        try (java.io.FileInputStream fis = new java.io.FileInputStream(outputFile)) {
            fis.read(magic);
        }

        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(magic)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        assertEquals(0x46554747, buffer.getInt()); // "GGUF"
    }

    @Test
    @DisplayName("Test exported file can be read by GGUFReader")
    void testExportedFileReadable() throws Exception {
        SameDiff model = createSmallLLaMAModel();
        File outputFile = tempDir.resolve("test_readable.gguf").toFile();

        ExportOptions options = ExportOptions.builder()
                .modelName("test-readable-model")
                .build();
        GGMLModelExport.exportModel(model, outputFile, options);

        // Try to read it back
        try (org.nd4j.ggml.format.GGUFReader reader =
                     new org.nd4j.ggml.format.GGUFReader(outputFile)) {
            org.nd4j.ggml.format.GGMLMetadata metadata = reader.getMetadata();

            assertNotNull(metadata);
            assertEquals("llama", metadata.getArchitecture());
        }
    }

    // ========== GGMLExportException Tests ==========

    @Test
    @DisplayName("Test GGMLExportException with message")
    void testGGMLExportExceptionMessage() {
        GGMLExportException ex = new GGMLExportException("Test error message");
        assertEquals("Test error message", ex.getMessage());
    }

    @Test
    @DisplayName("Test GGMLExportException with cause")
    void testGGMLExportExceptionWithCause() {
        RuntimeException cause = new RuntimeException("Root cause");
        GGMLExportException ex = new GGMLExportException("Test error", cause);

        assertEquals("Test error", ex.getMessage());
        assertEquals(cause, ex.getCause());
    }

    @Test
    @DisplayName("Test bounded BF16 to Q4_K GGUF requantization")
    void testBoundedBFloat16ToQ4KRequantization() throws Exception {
        File source = tempDir.resolve("source-bf16.gguf").toFile();
        File output = tempDir.resolve("output-q4k.gguf").toFile();

        int weightElements = 256 * 4128;
        try (GGUFWriter writer = new GGUFWriter(source, 3)) {
            writer.addMetadata("general.architecture", "llama");
            writer.addMetadata("general.name", "bounded-requantization-test");
            writer.addMetadata("tokenizer.ggml.tokens", List.of("<unk>", "<s>", "</s>"));
            writer.registerTensor("blk.0.attn_q.weight", new long[]{256, 4128},
                    GGMLDataType.GGML_TYPE_BF16);
            writer.registerTensor("blk.0.attn_norm.weight", new long[]{256},
                    GGMLDataType.GGML_TYPE_BF16);
            writer.registerTensor("token_embd.weight", new long[]{256, 32},
                    GGMLDataType.GGML_TYPE_BF16);
            writer.registerTensor("blk.0.ssm_in.weight", new long[]{96, 64},
                    GGMLDataType.GGML_TYPE_BF16);
            writer.registerTensor("blk.0.ssm_conv.weight", new long[]{4, 64},
                    GGMLDataType.GGML_TYPE_BF16);
            writer.writeHeader();
            writer.writeTensorData("blk.0.attn_q.weight", bfloat16Data(weightElements));
            writer.writeTensorData("blk.0.attn_norm.weight", bfloat16Data(256));
            writer.writeTensorData("token_embd.weight", bfloat16Data(256 * 32));
            writer.writeTensorData("blk.0.ssm_in.weight", bfloat16Data(96 * 64));
            writer.writeTensorData("blk.0.ssm_conv.weight", bfloat16Data(4 * 64));
            writer.finalizeFile();
        }

        GGMLModelExport.requantize(source, output, ExportOptions.QuantizationType.Q4_K);

        assertTrue(source.isFile(), "Requantization must preserve the verified source GGUF");
        assertTrue(output.isFile());
        assertTrue(output.length() < source.length());

        try (GGUFReader reader = new GGUFReader(output)) {
            assertEquals("bounded-requantization-test",
                    reader.getHeader().getMetadata().get("general.name"));
            assertEquals(GGMLDataType.GGML_TYPE_Q4_K,
                    tensor(reader, "blk.0.attn_q.weight").getDataType());
            assertEquals(GGMLDataType.GGML_TYPE_F32,
                    tensor(reader, "blk.0.attn_norm.weight").getDataType());
            assertEquals(GGMLDataType.GGML_TYPE_Q8_0,
                    tensor(reader, "token_embd.weight").getDataType());
            assertEquals(GGMLDataType.GGML_TYPE_Q8_0,
                    tensor(reader, "blk.0.ssm_in.weight").getDataType(),
                    "A narrow row divisible by 32 must fall back from Q4_K to Q8_0");
            assertEquals(GGMLDataType.GGML_TYPE_BF16,
                    tensor(reader, "blk.0.ssm_conv.weight").getDataType(),
                    "A row narrower than every runtime quant block must remain dense");
            assertEquals(GGMLDataType.GGML_TYPE_Q4_K.calculateStorageBytes(weightElements),
                    tensor(reader, "blk.0.attn_q.weight").getDataSize());
        }
    }

    @Test
    @DisplayName("Test Q3_K to Q4_K GGUF requantization")
    void testQ3KToQ4KRequantization() throws Exception {
        File source = tempDir.resolve("source-q3k.gguf").toFile();
        File output = tempDir.resolve("output-q4k-from-q3k.gguf").toFile();
        int elements = 256 * 256;

        try (GGUFWriter writer = new GGUFWriter(source, 3)) {
            writer.addMetadata("general.architecture", "llama");
            writer.addMetadata("general.name", "q3k-requantization-test");
            writer.registerTensor("blk.0.attn_gate.weight", new long[]{256, 256},
                    GGMLDataType.GGML_TYPE_Q3_K);
            writer.writeHeader();
            writer.writeTensorData("blk.0.attn_gate.weight", q3KData(elements));
            writer.finalizeFile();
        }

        GGMLModelExport.requantize(source, output, ExportOptions.QuantizationType.Q4_K);

        try (GGUFReader reader = new GGUFReader(output)) {
            assertEquals("q3k-requantization-test",
                    reader.getHeader().getMetadata().get("general.name"));
            assertEquals(GGMLDataType.GGML_TYPE_Q4_K,
                    tensor(reader, "blk.0.attn_gate.weight").getDataType());
            assertEquals(GGMLDataType.GGML_TYPE_Q4_K.calculateStorageBytes(elements),
                    tensor(reader, "blk.0.attn_gate.weight").getDataSize());
        }
    }

    @Test
    @DisplayName("Test IQ4_NL to Q4_K GGUF requantization preserves values")
    void testIQ4NLToQ4KRequantizationFidelity() throws Exception {
        File source = tempDir.resolve("source-iq4nl.gguf").toFile();
        File output = tempDir.resolve("output-q4k-from-iq4nl.gguf").toFile();
        int elements = 256 * 256;

        float[] original = new float[elements];
        for (int i = 0; i < elements; i++) {
            original[i] = (float) (Math.sin(i * 0.0217) * Math.cos(i * 0.0053) * 1.4
                    + 0.35 * Math.sin(i * 0.061));
        }

        byte[] iq4Bytes = iq4NlData(original);
        try (GGUFWriter writer = new GGUFWriter(source, 3)) {
            writer.addMetadata("general.architecture", "llama");
            writer.addMetadata("general.name", "iq4nl-requantization-test");
            writer.registerTensor("blk.0.attn_gate.weight", new long[]{256, 256},
                    GGMLDataType.GGML_TYPE_IQ4_NL);
            writer.writeHeader();
            writer.writeTensorData("blk.0.attn_gate.weight", iq4Bytes);
            writer.finalizeFile();
        }

        GGMLModelExport.requantize(source, output, ExportOptions.QuantizationType.Q4_K);

        GGMLTensorInfo converted;
        try (GGUFReader reader = new GGUFReader(output)) {
            converted = tensor(reader, "blk.0.attn_gate.weight");
            assertEquals(GGMLDataType.GGML_TYPE_Q4_K, converted.getDataType());
            assertEquals(GGMLDataType.GGML_TYPE_Q4_K.calculateStorageBytes(elements),
                    converted.getDataSize());

            byte[] q4kBytes = new byte[Math.toIntExact(converted.getDataSize())];
            reader.readTensorDataRange(converted, 0, q4kBytes, 0, q4kBytes.length);

            float[] decodedQ4K = new Q4_KDequantizer().dequantize(q4kBytes, elements);
            float[] decodedIq4 = new IQ4_NLDequantizer().dequantize(iq4Bytes, elements);

            float range = 0f;
            for (float v : original) range = Math.max(range, Math.abs(v));

            double mae = 0;
            double maeIq4 = 0;
            for (int i = 0; i < elements; i++) {
                mae += Math.abs(decodedQ4K[i] - decodedIq4[i]);
                maeIq4 += Math.abs(decodedIq4[i] - original[i]);
            }
            mae /= elements;
            maeIq4 /= elements;
            assertTrue(Double.isFinite(mae), "Requantized output must be finite");
            assertTrue(mae <= 0.15 * range,
                    "Q4_K re-quantization must stay within 4-bit error of the "
                            + "IQ4_NL source, MAE=" + mae + " maeIq4=" + maeIq4
                            + " range=" + range);
        }
    }

    @Test
    @DisplayName("Test cached Qwen3.5 BF16 to Q4_K runtime import")
    void testCachedQwen35BFloat16ToQ4KRuntimeImport() throws Exception {
        File source = new File(System.getProperty("user.home"),
                ".cache/dl4j-llm-models/Qwen3.5-0.8B-BF16.gguf");
        Assumptions.assumeTrue(source.isFile(), "Cached Qwen3.5 BF16 GGUF is unavailable");
        File output = tempDir.resolve("qwen35-optimized-q4k.gguf").toFile();

        GGMLModelExport.requantize(source, output, ExportOptions.QuantizationType.Q4_K);

        ConversionOptions options = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                .targetDataType(DataType.HALF)
                .forTraining(false)
                .preserveTokenizerInfo(true)
                .build();
        try (SameDiff graph = GGMLModelImport.importModel(output, options)) {
            assertNotNull(graph);
            assertNotNull(graph.getVariable("lm_logits"));
            assertNotNull(graph.getVariable("gdn_q_0"),
                    "Qwen3.5 GDN Q slice must be present in the imported graph");
            assertNotNull(graph.getVariable("gdn_v_0"),
                    "Qwen3.5 GDN V slice must be present in the imported graph");
        }
    }

    // ========== Helper Methods ==========

    private static GGMLTensorInfo tensor(GGUFReader reader, String name) throws Exception {
        return reader.getTensorInfos().stream()
                .filter(tensor -> name.equals(tensor.getName()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("Missing tensor " + name));
    }

    private static byte[] bfloat16Data(int elements) {
        ByteBuffer data = ByteBuffer.allocate(Math.multiplyExact(elements, Short.BYTES))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < elements; i++) {
            float value = (float) Math.sin(i * 0.0078125);
            data.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
        }
        return data.array();
    }

    private static byte[] q3KData(int elements) {
        if (elements % 256 != 0) {
            throw new IllegalArgumentException("Q3_K data must contain complete 256-element blocks");
        }
        byte[] data = new byte[(elements / 256) * 110];
        for (int block = 0; block < elements / 256; block++) {
            data[block * 110 + 108] = 0x00;
            data[block * 110 + 109] = 0x3C; // FP16 d = 1.0
        }
        return data;
    }

    private static final int[] KVALUES_IQ4NL_TEST = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };

    /**
     * Encode floats into IQ4_NL blocks mirroring ggml's quantize_row_iq4_nl_ref:
     * per 32-element block, d = maxAbs/127 as FP16, then the nearest codebook
     * entry for each value packed low-nibble first.
     */
    private static byte[] iq4NlData(float[] values) {
        if (values.length % 32 != 0) {
            throw new IllegalArgumentException("IQ4_NL data must contain complete 32-element blocks");
        }
        byte[] data = new byte[(values.length / 32) * 18];
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        for (int block = 0; block < values.length / 32; block++) {
            float max = 0f;
            for (int j = 0; j < 32; j++) {
                float v = Math.abs(values[block * 32 + j]);
                if (v > max) max = v;
            }
            float d = max / 127f;
            buffer.putShort(floatToFp16Test(d));
            for (int j = 0; j < 16; j++) {
                int lo = nearestCodebook(values[block * 32 + j] / d);
                int hi = nearestCodebook(values[block * 32 + j + 16] / d);
                buffer.put((byte) ((lo & 0x0F) | ((hi & 0x0F) << 4)));
            }
        }
        return data;
    }

    private static int nearestCodebook(float value) {
        int best = 0;
        float bestError = Float.POSITIVE_INFINITY;
        for (int q = 0; q < KVALUES_IQ4NL_TEST.length; q++) {
            float error = Math.abs(KVALUES_IQ4NL_TEST[q] - value);
            if (error < bestError) {
                bestError = error;
                best = q;
            }
        }
        return best;
    }

    private static short floatToFp16Test(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            if (exponent < -10) return (short) sign;
            mantissa |= 0x800000;
            mantissa = mantissa >> (1 - exponent);
            return (short) (sign | (mantissa >> 13));
        } else if (exponent >= 31) {
            if ((bits & 0x7FFFFFFF) > 0x7F800000) return (short) (sign | 0x7E00);
            return (short) (sign | 0x7C00);
        }
        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }

    /**
     * Create a minimal LLaMA model for testing detection
     */
    private SameDiff createMinimalLLaMAModel() {
        SameDiff sd = SameDiff.create();

        // Embeddings
        sd.var("model.embed_tokens.weight", Nd4j.rand(DataType.FLOAT, 32000, 4096));

        // Layer 0
        sd.var("model.layers.0.self_attn.q_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.self_attn.k_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.self_attn.v_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.self_attn.o_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.mlp.gate_proj.weight", Nd4j.rand(DataType.FLOAT, 11008, 4096));
        sd.var("model.layers.0.mlp.up_proj.weight", Nd4j.rand(DataType.FLOAT, 11008, 4096));
        sd.var("model.layers.0.mlp.down_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 11008));
        sd.var("model.layers.0.input_layernorm.weight", Nd4j.rand(DataType.FLOAT, 4096));
        sd.var("model.layers.0.post_attention_layernorm.weight", Nd4j.rand(DataType.FLOAT, 4096));

        // Final norm
        sd.var("model.norm.weight", Nd4j.rand(DataType.FLOAT, 4096));

        // LM head
        sd.var("lm_head.weight", Nd4j.rand(DataType.FLOAT, 32000, 4096));

        return sd;
    }

    /**
     * Create a small LLaMA model for actual export tests (smaller to reduce test time)
     */
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
}
