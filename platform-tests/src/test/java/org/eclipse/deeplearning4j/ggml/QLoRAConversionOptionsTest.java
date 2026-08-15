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
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.ggml;

import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.config.ND4JInferenceWeightDataType;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGUFWriter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Collection;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the QLoRA (Quantized Low-Rank Adaptation) conversion path.
 *
 * <p>Validates that {@link ConversionOptions#forTrainingQuantized()} keeps quantized
 * weights packed as INT8 bytes instead of dequantizing them to FP32 (~4x memory
 * blowup), and that the companion {@code .__q__} metadata required by
 * {@code ggml_qmatmul} ops is produced for supported quantization types.</p>
 */
@DisplayName("QLoRA ConversionOptions Test")
@Tag(TagNames.SMOKE)
class QLoRAConversionOptionsTest {

    @TempDir
    Path tempDir;

    // =========================================================================
    // ConversionOptions preset contract
    // =========================================================================

    @Test
    @DisplayName("forTrainingQuantized: quantizationMode=RUNTIME_QUANTIZED_MATMUL, forTraining=true, dtype=HALF")
    void testForTrainingQuantizedPreset() {
        ConversionOptions opts = ConversionOptions.forTrainingQuantized();

        assertEquals(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL,
                opts.getQuantizationMode(),
                "forTrainingQuantized() must use RUNTIME_QUANTIZED_MATMUL to keep weights packed");
        assertTrue(opts.isForTraining(),
                "forTrainingQuantized() must set forTraining=true so weights are registered as sd.var");
        assertEquals(DataType.HALF, opts.getTargetDataType(),
                "forTrainingQuantized() must use FP16 working dtype");
        assertTrue(opts.isRuntimeQuantizedMatmul(),
                "isRuntimeQuantizedMatmul() must return true for forTrainingQuantized()");
    }

    @Test
    @DisplayName("forTrainingQuantized: forTraining and RUNTIME_QUANTIZED_MATMUL are independently true")
    void testForTrainingQuantizedBothFlagsIndependent() {
        // Verify that the two flags do not conflict — the builder must allow both.
        ConversionOptions opts = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                .forTraining(true)
                .targetDataType(DataType.HALF)
                .build();

        assertTrue(opts.isRuntimeQuantizedMatmul());
        assertTrue(opts.isForTraining());
    }

    @Test
    @DisplayName("forTraining: must use DEQUANTIZE_TO_FLOAT32 (full-param FT preset, NOT for QLoRA)")
    void testForTrainingUsesFloat32Dequant() {
        ConversionOptions opts = ConversionOptions.forTraining();

        assertEquals(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT32,
                opts.getQuantizationMode(),
                "forTraining() dequantizes everything to FP32 (50 GB blowup on 13 GB GGUF)");
        assertTrue(opts.isForTraining());
        assertFalse(opts.isRuntimeQuantizedMatmul(),
                "forTraining() must NOT be RUNTIME_QUANTIZED_MATMUL — callers need the distinction");
    }

    @Test
    @DisplayName("runtimeQuantizedMatmul: forTraining=false (inference preset, not QLoRA)")
    void testRuntimeQuantizedMatmulIsNotForTraining() {
        ConversionOptions opts = ConversionOptions.runtimeQuantizedMatmul();

        assertTrue(opts.isRuntimeQuantizedMatmul());
        assertFalse(opts.isForTraining(),
                "runtimeQuantizedMatmul() is the inference preset; forTraining must be false");
    }

    @Test
    @DisplayName("inference weight dtype policies map to explicit conversion modes")
    void testInferenceWeightDataTypePolicies() {
        assertInferencePolicy(ND4JInferenceWeightDataType.FLOAT32,
                ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT32, DataType.FLOAT, false);
        assertInferencePolicy(ND4JInferenceWeightDataType.FLOAT16,
                ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16, DataType.HALF, false);
        assertInferencePolicy(ND4JInferenceWeightDataType.BFLOAT16,
                ConversionOptions.QuantizationMode.DEQUANTIZE_TO_BFLOAT16, DataType.BFLOAT16, false);
        assertInferencePolicy(ND4JInferenceWeightDataType.FLOAT8_E4M3,
                ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT8_E4M3, DataType.FLOAT8, false);
        assertInferencePolicy(ND4JInferenceWeightDataType.FLOAT8_E5M2,
                ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT8_E5M2, DataType.FLOAT8_E5M2, false);
        assertInferencePolicy(ND4JInferenceWeightDataType.INT8,
                ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_INT8, DataType.HALF, true);
        assertInferencePolicy(ND4JInferenceWeightDataType.INT4,
                ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_INT4, DataType.HALF, true);
    }

    private static void assertInferencePolicy(ND4JInferenceWeightDataType weightType,
                                              ConversionOptions.QuantizationMode expectedMode,
                                              DataType expectedTargetType,
                                              boolean runtimeQuantized) {
        ConversionOptions options = ConversionOptions.forInference(weightType);
        assertEquals(expectedMode, options.getQuantizationMode());
        assertEquals(expectedTargetType, options.getTargetDataType());
        assertEquals(runtimeQuantized, options.isRuntimeQuantizedMatmul());
        assertFalse(options.isForTraining());
    }

    @Test
    @DisplayName("load configuration defaults to FP16 and keeps dtype controls consistent")
    void testPipelineLoadConfigWeightDataTypes() throws IOException {
        String previousWeightDtype = System.getProperty("nd4j.optimizer.weightDtype");
        String previousFp16 = System.getProperty("nd4j.optimizer.fp16");
        String previousBf16 = System.getProperty("nd4j.optimizer.bf16");
        try {
            System.clearProperty("nd4j.optimizer.weightDtype");
            System.clearProperty("nd4j.optimizer.fp16");
            System.clearProperty("nd4j.optimizer.bf16");

            PipelineLoader.LoadConfig defaults = PipelineLoader.LoadConfig.defaults();
            assertEquals("fp16", defaults.getDataType());
            assertFalse(defaults.convertToFloat32());

            PipelineLoader.LoadConfig fp8 = PipelineLoader.LoadConfig.builder()
                    .dataType("float8_e4m3")
                    .build();
            assertEquals("fp8", fp8.getDataType());
            assertFalse(fp8.convertToFloat32());
            assertEquals(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT8_E4M3,
                    GGMLPipelineLoader.conversionOptions(fp8).getQuantizationMode());

            PipelineLoader.LoadConfig int4 = PipelineLoader.LoadConfig.builder()
                    .dataType("q4_k")
                    .build();
            assertEquals("int4", int4.getDataType());
            assertEquals(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_INT4,
                    GGMLPipelineLoader.conversionOptions(int4).getQuantizationMode());

            PipelineLoader.LoadConfig fp32 = PipelineLoader.LoadConfig.builder()
                    .convertToFloat32(true)
                    .build();
            assertEquals("fp32", fp32.getDataType());
            assertTrue(fp32.convertToFloat32());
        } finally {
            restoreProperty("nd4j.optimizer.weightDtype", previousWeightDtype);
            restoreProperty("nd4j.optimizer.fp16", previousFp16);
            restoreProperty("nd4j.optimizer.bf16", previousBf16);
        }
    }

    @Test
    @DisplayName("weight dtype parser supports aliases and rejects unknown values")
    void testWeightDataTypeAliases() {
        assertEquals(ND4JInferenceWeightDataType.FLOAT16,
                ND4JInferenceWeightDataType.fromString("half"));
        assertEquals(ND4JInferenceWeightDataType.FLOAT8_E5M2,
                ND4JInferenceWeightDataType.fromString("fp8-e5m2"));
        assertEquals(ND4JInferenceWeightDataType.INT8,
                ND4JInferenceWeightDataType.fromString("q8_0"));
        assertEquals(ND4JInferenceWeightDataType.INT4,
                ND4JInferenceWeightDataType.fromString("q4-k"));
        assertThrows(IllegalArgumentException.class,
                () -> ND4JInferenceWeightDataType.fromString("automatic"));
    }

    // =========================================================================
    // Converter: packed INT8 storage for Q8_0 supported type
    // =========================================================================

    /**
     * Build a tiny GGUF file with:
     *   - one Q8_0 quantized weight  (shape [32, 32] = 1024 elements = 32 blocks × 34 bytes)
     *   - one F32 non-quantized weight (shape [8])
     * architecture=generic so GenericArchitecture registers them as sd.var/sd.constant.
     *
     * <p>Q8_0 block layout: 2 bytes F16 scale + 32 bytes INT8 quants = 34 bytes/block.</p>
     */
    private File createTinyQ8_0GGUFFile() throws IOException {
        return createTinyQ8_0GGUFFile(false);
    }

    private File createTinyQ8_0GGUFFile(boolean includeDenseFallback) throws IOException {
        // Q8_0: block=32, 34 bytes/block.  Shape [32,32]=1024 elements = 32 blocks = 1088 bytes.
        // GGUF stores shapes column-major: [innerDim, outerDim], so we write [32, 32].
        // The converter reverses this to ND4J [32, 32] (symmetric for square).
        final int BLOCK_SIZE = 32;
        final int NUM_ELEMENTS = 32 * 32;          // 1024
        final int NUM_BLOCKS = NUM_ELEMENTS / BLOCK_SIZE; // 32
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE;       // 34 (F16 scale + 32 INT8)
        final int TOTAL_BYTES = NUM_BLOCKS * BYTES_PER_BLOCK; // 1088

        byte[] q8Data = nonZeroQ8_0Data(NUM_BLOCKS, BLOCK_SIZE, BYTES_PER_BLOCK);

        File file = tempDir.resolve(includeDenseFallback
                ? "tiny_q8_0_mixed.gguf"
                : "tiny_q8_0.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(file, 2)) {
            writer.addMetadataString("general.architecture", "generic");

            // Q8_0 linear weight: shape [32, 32] in GGUF column-major
            writer.registerTensor("blk.0.attn_q.weight", new long[]{32, 32}, GGMLDataType.GGML_TYPE_Q8_0);
            // Q8_0 token embedding: same storage, but must be dequantized because gather cannot consume packed bytes.
            writer.registerTensor("token_embd.weight", new long[]{32, 32}, GGMLDataType.GGML_TYPE_Q8_0);
            // F32 bias (non-quantized): shape [8]
            writer.registerTensor("blk.0.attn_q.bias", new long[]{8}, GGMLDataType.GGML_TYPE_F32);
            if (includeDenseFallback) {
                // Hybrid models can contain rank-2 weights whose innermost row is too narrow
                // for any packed runtime block. They must remain dense.
                writer.registerTensor("blk.0.ssm_conv.weight", new long[]{4, 32},
                        GGMLDataType.GGML_TYPE_F32);
            }

            writer.writeHeader();

            writer.writeTensorData("blk.0.attn_q.weight", q8Data);
            writer.writeTensorData("token_embd.weight", q8Data);
            writer.writeTensorData("blk.0.attn_q.bias", new byte[8 * 4]); // 8 × 4 bytes F32
            if (includeDenseFallback) {
                writer.writeTensorData("blk.0.ssm_conv.weight", new byte[4 * 32 * 4]);
            }

            writer.finalizeFile();
        }
        return file;
    }

    private static byte[] nonZeroQ8_0Data(int numBlocks, int blockSize, int bytesPerBlock) {
        ByteBuffer buffer = ByteBuffer.allocate(numBlocks * bytesPerBlock).order(ByteOrder.LITTLE_ENDIAN);
        for (int block = 0; block < numBlocks; block++) {
            buffer.putShort(fp32ToFp16(0.5f));
            for (int i = 0; i < blockSize; i++) {
                buffer.put((byte) (i % 2 == 0 ? 2 : -2));
            }
        }
        return buffer.array();
    }

    private static short fp32ToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            return (short) sign;
        } else if (exponent >= 31) {
            return (short) (sign | 0x7C00);
        }

        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }

    @Test
    @DisplayName("forTrainingQuantized: Q8_0 weight is stored as INT8 (not dequantized to FP32/FP16)")
    void testQ8_0WeightStoredAsINT8WithForTrainingQuantized() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forTrainingQuantized();
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        assertNotNull(sd);

        // GenericArchitecture stores tensors under name.replace('.','_').replace('-','_').
        // So "blk.0.attn_q.weight" -> "blk_0_attn_q_weight"
        // And the companion "blk.0.attn_q.weight.__q__" -> "blk_0_attn_q_weight___q__"
        // (double underscore from ".__q__" → "_.__q__" → "__q__" after replace)
        // Actually the replace targets '.' -> '_', so:
        //   "blk.0.attn_q.weight.__q__".replace('.','_') = "blk_0_attn_q_weight___q__"
        // We verify the main weight variable.

        SDVariable weightVar = sd.getVariable("blk_0_attn_q_weight");
        assertNotNull(weightVar, "Packed Q8_0 weight variable 'blk_0_attn_q_weight' must exist in graph");

        INDArray weightArr = weightVar.getArr();
        assertNotNull(weightArr, "Weight variable must have an associated array");

        // The key assertion: Q8_0 is supported by RUNTIME_QUANTIZED_MATMUL,
        // so the weight MUST be stored as raw INT8 bytes — NOT dequantized to FP32 or FP16.
        assertEquals(DataType.INT8, weightArr.dataType(),
                "Q8_0 weight with forTrainingQuantized() must remain INT8 (packed); " +
                "got " + weightArr.dataType() + ". If FP32/FP16, dequantization happened " +
                "which defeats QLoRA's purpose (13 GB GGUF → ~50 GB).");

        // The array holds the raw packed bytes — shape is [numBytes] 1D.
        // For [32,32] Q8_0: 32 blocks × 34 bytes = 1088 bytes.
        assertEquals(1, weightArr.rank(), "Packed INT8 weight is stored as 1D byte array");
        assertEquals(1088L, weightArr.length(), "Q8_0 [32,32] should produce 1088 packed bytes");
    }

    @Test
    @DisplayName("forTrainingQuantized: Q8_0 token embedding is dequantized for gather")
    void testQ8_0TokenEmbeddingDequantizedForGather() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forTrainingQuantized();
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        assertNotNull(sd);

        SDVariable tokenVar = sd.getVariable("token_embd_weight");
        assertNotNull(tokenVar, "Quantized token embedding variable must exist in graph");
        INDArray tokenArr = tokenVar.getArr();
        assertNotNull(tokenArr, "Token embedding must have an associated array");
        assertEquals(DataType.HALF, tokenArr.dataType(),
                "token_embd.weight must be dequantized because gather cannot consume packed INT8 bytes");
        assertArrayEquals(new long[]{32, 32}, tokenArr.shape(),
                "Token embedding must keep logical [vocab, hidden] shape");
        assertEquals(1.0f, tokenArr.getFloat(0, 0), 1e-3f,
                "token_embd.weight must preserve dequantized Q8_0 values for gather");
        assertEquals(-1.0f, tokenArr.getFloat(0, 1), 1e-3f,
                "token_embd.weight must preserve signed dequantized Q8_0 values for gather");
        assertNull(sd.getVariable("token_embd_weight___q__"),
                "token_embd.weight must not receive ggml_qmatmul companion metadata");
    }

    @Test
    @DisplayName("forTrainingQuantized: companion .__q__ metadata is produced for Q8_0 weight")
    void testQ8_0CompanionMetadataExists() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forTrainingQuantized();
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        assertNotNull(sd);

        // The companion key in the raw weights map is "blk.0.attn_q.weight.__q__".
        // GenericArchitecture calls weights.entrySet() and stores each as sd.var(varName, array)
        // where varName = name.replace('.','_').replace('-','_').
        // The companion "blk.0.attn_q.weight.__q__" becomes "blk_0_attn_q_weight___q__".
        SDVariable metaVar = sd.getVariable("blk_0_attn_q_weight___q__");
        assertNotNull(metaVar,
                "Companion .__q__ metadata variable must exist for Q8_0 weight. " +
                "This is required by LLaMAArchitecture.qMatMulOrFp32Mmul() to emit ggml_qmatmul ops.");

        INDArray meta = metaVar.getArr();
        assertNotNull(meta, "Companion .__q__ metadata must have an associated array");
        assertEquals(DataType.INT64, meta.dataType(), "Companion metadata must be LONG[3] = [quantType, N, K]");
        assertEquals(3L, meta.length(), "Companion metadata must have exactly 3 elements: [quantType, N, K]");

        // Verify content: Q8_0 has toGgmlQuantType() = 4, N=32, K=32 (shape reversed to ND4J C-order)
        long quantType = meta.getLong(0);
        long N = meta.getLong(1);
        long K = meta.getLong(2);
        assertEquals(4L, quantType, "Q8_0 GgmlQuantType index must be 4");
        assertEquals(32L, N, "N dimension must match weight output dimension");
        assertEquals(32L, K, "K dimension must match weight input dimension");
    }

    @Test
    @DisplayName("forTrainingQuantized: non-quantized F32 bias is NOT stored as INT8")
    void testNonQuantizedWeightNotPackedAsINT8() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forTrainingQuantized();
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        assertNotNull(sd);

        SDVariable biasVar = sd.getVariable("blk_0_attn_q_bias");
        assertNotNull(biasVar, "F32 bias variable must exist");
        INDArray biasArr = biasVar.getArr();
        assertNotNull(biasArr);
        // Non-quantized F32 tensor: dtype stays FLOAT (targetDataType is HALF but F32 stays F32
        // because convertTensorDataDirect casts to targetType only when it differs from the tensor's
        // native type — and F32 with target HALF would cast to HALF).
        // Either way, it must NOT be INT8.
        assertNotEquals(DataType.INT8, biasArr.dataType(),
                "Non-quantized F32 tensor must not be stored as INT8");
    }

    @Test
    @DisplayName("runtime quantized matmul permits a dense rank-2 fallback weight")
    void testRuntimeQuantizedMatmulPermitsDenseRank2Fallback()
            throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile(true);

        SameDiff sd = GGMLModelImport.importModel(
                gguf, ConversionOptions.forTrainingQuantized());
        INDArray dense = sd.getVariable("blk_0_ssm_conv_weight").getArr();

        assertNotNull(dense);
        assertNotEquals(DataType.INT8, dense.dataType());
        assertArrayEquals(new long[]{32L, 4L}, dense.shape());
        assertNull(sd.getVariable("blk_0_ssm_conv_weight___q__"),
                "Dense fallback weights must not receive packed qmatmul metadata");
    }

    @Test
    @DisplayName("forTrainingQuantized: weight variables are sd.var not sd.constant (trainable)")
    void testWeightVariablesAreTrainable() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forTrainingQuantized();
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        assertNotNull(sd);

        // With forTraining=true, GenericArchitecture uses sd.var() not sd.constant().
        // sd.var() variables have VariableType.VARIABLE; sd.constant() variables have CONSTANT.
        // We check that the weight is not a constant (constants have no gradient).
        SDVariable weightVar = sd.getVariable("blk_0_attn_q_weight");
        assertNotNull(weightVar);
        // A sd.var() SDVariable reports VariableType.VARIABLE
        assertEquals(org.nd4j.autodiff.samediff.VariableType.VARIABLE,
                weightVar.getVariableType(),
                "With forTraining=true, weight must be sd.var (VARIABLE), not sd.constant (CONSTANT)");
    }

    // =========================================================================
    // Inference storage policies
    // =========================================================================

    @Test
    @DisplayName("forInference: FP16 is a dense dequantized weight")
    void testQ8_0WeightDequantizedToFp16ForInference() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forInference(ND4JInferenceWeightDataType.FLOAT16);
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        INDArray weightArr = sd.getVariable("blk_0_attn_q_weight").getArr();

        assertEquals(DataType.HALF, weightArr.dataType());
        assertArrayEquals(new long[]{32L, 32L}, weightArr.shape(),
                "Dequantized weight must have the logical tensor shape, not raw byte shape");
    }

    @Test
    @DisplayName("forInference: FP8 stores dense FP8 weights")
    void testQ8_0WeightDequantizedToFp8ForInference() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forInference(ND4JInferenceWeightDataType.FLOAT8_E4M3);
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        INDArray weightArr = sd.getVariable("blk_0_attn_q_weight").getArr();

        assertEquals(DataType.FLOAT8, weightArr.dataType());
        assertArrayEquals(new long[]{32L, 32L}, weightArr.shape());
    }

    @Test
    @DisplayName("forInference: FP8 E5M2 stores dense FP8 weights")
    void testQ8_0WeightDequantizedToFp8E5M2ForInference() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forInference(ND4JInferenceWeightDataType.FLOAT8_E5M2);
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        INDArray weightArr = sd.getVariable("blk_0_attn_q_weight").getArr();

        assertEquals(DataType.FLOAT8_E5M2, weightArr.dataType());
        assertArrayEquals(new long[]{32L, 32L}, weightArr.shape());
    }

    @Test
    @DisplayName("forInference: INT8 keeps compatible Q8_0 linear weights packed")
    void testQ8_0WeightRemainsPackedForInt8Inference() throws IOException, GGMLImportException {
        File gguf = createTinyQ8_0GGUFFile();

        ConversionOptions opts = ConversionOptions.forInference(ND4JInferenceWeightDataType.INT8);
        SameDiff sd = GGMLModelImport.importModel(gguf, opts);
        INDArray weightArr = sd.getVariable("blk_0_attn_q_weight").getArr();

        assertEquals(DataType.INT8, weightArr.dataType());
        assertEquals(1088L, weightArr.length());
        assertNotNull(sd.getVariable("blk_0_attn_q_weight___q__"));
    }

    @Test
    @DisplayName("forInference: INT4 rejects a Q8_0 linear weight instead of dequantizing it")
    void testInt4RejectsIncompatibleQ8_0Weight() throws IOException {
        File gguf = createTinyQ8_0GGUFFile();
        ConversionOptions opts = ConversionOptions.forInference(ND4JInferenceWeightDataType.INT4);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> GGMLModelImport.importModel(gguf, opts));
        assertTrue(causeChainContains(exception, "unsupported GGUF type"));
        assertTrue(causeChainContains(exception, "GGML_TYPE_Q8_0"));
    }

    private static boolean causeChainContains(Throwable failure, String text) {
        for (Throwable current = failure; current != null; current = current.getCause()) {
            if (current.getMessage() != null && current.getMessage().contains(text)) {
                return true;
            }
        }
        return false;
    }

    private static void restoreProperty(String name, String previousValue) {
        if (previousValue == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previousValue);
        }
    }
}
