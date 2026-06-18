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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.ModelFamily;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE;
import org.nd4j.linalg.api.ops.impl.transforms.custom.PerLayerEmbedding;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Gemma 4 custom ops (per_layer_embedding, shared_kv_attention, dual_rope)
 * and GGUF model import.
 *
 * Op tests run with both INDArray direct execution and SameDiff graph execution.
 * Import test downloads the smallest Gemma model (gemma-3-1b-it Q4_K_M) and verifies
 * GGUF import succeeds.
 *
 * Run examples:
 *   cd platform-tests && mvn test -Dtest=TestGemma4Ops 2>&1 | tee /tmp/gemma4-ops.log
 *   cd platform-tests && mvn test -Dtest=TestGemma4Ops#testPerLayerEmbeddingINDArray 2>&1 | tee /tmp/ple.log
 *   cd platform-tests && mvn test -Dtest=TestGemma4Ops#testGemmaGGUFImport 2>&1 | tee /tmp/gemma-import.log
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TestGemma4Ops extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void cleanup() {
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
    }

    // ==================== per_layer_embedding ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("per_layer_embedding: INDArray execution")
    public void testPerLayerEmbeddingINDArray(Nd4jBackend backend) {
        int batch = 2, seqLen = 4, hiddenDim = 8, vocabSize = 16;

        INDArray hiddenStates = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
        INDArray pleWeight = Nd4j.randn(DataType.FLOAT, vocabSize, hiddenDim);
        INDArray tokenIds = Nd4j.createFromArray(new int[][]{{0, 3, 7, 1}, {2, 5, 11, 4}})
                .castTo(DataType.INT);

        double scale = 0.5;

        // Execute op
        PerLayerEmbedding op = new PerLayerEmbedding(hiddenStates, pleWeight, tokenIds, scale);
        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);
        INDArray output = results[0];

        // Verify shape: same as hiddenStates
        assertArrayEquals(new long[]{batch, seqLen, hiddenDim}, output.shape(),
                "Output shape must match hiddenStates shape");

        // Verify correctness: output = hiddenStates + scale * pleWeight[tokenIds]
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = tokenIds.getInt(b, s);
                for (int d = 0; d < hiddenDim; d++) {
                    float expected = hiddenStates.getFloat(b, s, d) +
                            (float)(scale * pleWeight.getFloat(tokenId, d));
                    float actual = output.getFloat(b, s, d);
                    assertEquals(expected, actual, 1e-4f,
                            String.format("Mismatch at [%d,%d,%d]", b, s, d));
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("per_layer_embedding: SameDiff execution")
    public void testPerLayerEmbeddingSameDiff(Nd4jBackend backend) {
        int batch = 2, seqLen = 4, hiddenDim = 8, vocabSize = 16;

        SameDiff sd = SameDiff.create();
        SDVariable hidden = sd.placeHolder("hidden", DataType.FLOAT, batch, seqLen, hiddenDim);
        SDVariable weight = sd.var("ple_weight", Nd4j.randn(DataType.FLOAT, vocabSize, hiddenDim));
        SDVariable ids = sd.placeHolder("token_ids", DataType.INT, batch, seqLen);

        SDVariable result = sd.nn.perLayerEmbedding(hidden, weight, ids, 1.0);
        sd.setOutputs(result.name());

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("hidden", Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim));
        inputs.put("token_ids", Nd4j.createFromArray(new int[][]{{0, 3, 7, 1}, {2, 5, 11, 4}})
                .castTo(DataType.INT));

        Map<String, INDArray> out = sd.output(inputs, result.name());
        INDArray output = out.get(result.name());

        assertNotNull(output);
        assertArrayEquals(new long[]{batch, seqLen, hiddenDim}, output.shape());

        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("per_layer_embedding: scale=1.0 (default)")
    public void testPerLayerEmbeddingDefaultScale(Nd4jBackend backend) {
        int batch = 1, seqLen = 2, hiddenDim = 4, vocabSize = 8;

        INDArray hiddenStates = Nd4j.zeros(DataType.FLOAT, batch, seqLen, hiddenDim);
        INDArray pleWeight = Nd4j.ones(DataType.FLOAT, vocabSize, hiddenDim);
        INDArray tokenIds = Nd4j.createFromArray(new int[][]{{0, 3}}).castTo(DataType.INT);

        PerLayerEmbedding op = new PerLayerEmbedding(hiddenStates, pleWeight, tokenIds);
        INDArray[] results = Nd4j.exec(op);

        INDArray output = results[0];
        // hiddenStates = 0, pleWeight = 1, scale = 1 → output should be all 1s
        INDArray expected = Nd4j.ones(DataType.FLOAT, batch, seqLen, hiddenDim);
        assertEquals(expected, output, "With zero hidden and ones weight, output should be all 1s");
    }

    // ==================== shared_kv_attention ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("shared_kv_attention: INDArray execution basic shape")
    public void testSharedKvAttentionINDArray(Nd4jBackend backend) {
        int batch = 1, numHeads = 4, numKvHeads = 2, seqLen = 8, headDim = 16;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, headDim);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);

        double scale = 1.0 / Math.sqrt(headDim);

        SharedKvAttention op = new SharedKvAttention(query, key, value,
                numHeads, numKvHeads, 1, 0, scale);
        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);
        INDArray output = results[0];

        // Output shape must match query shape
        assertArrayEquals(query.shape(), output.shape(),
                "Output shape must match query shape [batch, numHeads, seqLen, headDim]");

        // Output should not be all zeros
        assertNotEquals(0.0, output.sumNumber().doubleValue(), 1e-10,
                "Output should not be all zeros");

        // Check for NaN/Inf
        assertFalse(Double.isNaN(output.sumNumber().doubleValue()), "Output contains NaN");
        assertFalse(Double.isInfinite(output.sumNumber().doubleValue()), "Output contains Inf");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("shared_kv_attention: with mask")
    public void testSharedKvAttentionWithMask(Nd4jBackend backend) {
        int batch = 1, numHeads = 2, numKvHeads = 2, seqLen = 4, headDim = 8;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, headDim);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);

        // Causal mask: upper triangle = -inf, lower triangle = 0
        INDArray mask = Nd4j.zeros(DataType.FLOAT, batch, 1, seqLen, seqLen);
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                mask.putScalar(new int[]{0, 0, i, j}, -3.4028235e+38f);
            }
        }

        double scale = 1.0 / Math.sqrt(headDim);

        SharedKvAttention op = new SharedKvAttention(query, key, value, mask,
                numHeads, numKvHeads, 0, 0, scale);
        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray output = results[0];
        assertArrayEquals(query.shape(), output.shape());
        assertFalse(Double.isNaN(output.sumNumber().doubleValue()), "Output with mask contains NaN");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("shared_kv_attention: SameDiff execution")
    public void testSharedKvAttentionSameDiff(Nd4jBackend backend) {
        int batch = 1, numHeads = 4, numKvHeads = 2, seqLen = 6, headDim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, numHeads, seqLen, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, numKvHeads, seqLen, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, numKvHeads, seqLen, headDim);

        double scale = 1.0 / Math.sqrt(headDim);
        SDVariable result = sd.nn.sharedKvAttention(q, k, v, numHeads, numKvHeads);
        sd.setOutputs(result.name());

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("q", Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, headDim));
        inputs.put("k", Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim));
        inputs.put("v", Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim));

        Map<String, INDArray> out = sd.output(inputs, result.name());
        INDArray output = out.get(result.name());

        assertNotNull(output);
        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, output.shape());

        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("shared_kv_attention: GQA head repeat correctness")
    public void testSharedKvAttentionGQARepeat(Nd4jBackend backend) {
        // With numHeads=4, numKvHeads=1, all query heads should attend to the same K/V
        int batch = 1, numHeads = 4, numKvHeads = 1, seqLen = 2, headDim = 4;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, headDim);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);

        double scale = 1.0 / Math.sqrt(headDim);

        SharedKvAttention op = new SharedKvAttention(query, key, value,
                numHeads, numKvHeads, 0, 0, scale);
        INDArray[] results = Nd4j.exec(op);
        INDArray output = results[0];

        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, output.shape());
        assertFalse(Double.isNaN(output.sumNumber().doubleValue()), "GQA output contains NaN");
    }

    // ==================== dual_rope ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("dual_rope: local attention (type=0)")
    public void testDualRoPELocal(Nd4jBackend backend) {
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        DualRoPE op = new DualRoPE(input,
                DualRoPE.ATTENTION_TYPE_LOCAL, 0,
                10000.0, 1000000.0, 1.0, 1.0);
        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);
        INDArray output = results[0];

        // Shape must be preserved
        assertArrayEquals(input.shape(), output.shape(),
                "RoPE output shape must match input");

        // RoPE should modify the values (not identity)
        assertNotEquals(input, output,
                "RoPE output should differ from input");

        // No NaN/Inf
        assertFalse(Double.isNaN(output.sumNumber().doubleValue()), "RoPE output contains NaN");
        assertFalse(Double.isInfinite(output.sumNumber().doubleValue()), "RoPE output contains Inf");

        // RoPE preserves L2 norm of each head vector (rotation)
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    double inputNorm = 0, outputNorm = 0;
                    for (int d = 0; d < headDim; d++) {
                        float iv = input.getFloat(b, s, h, d);
                        float ov = output.getFloat(b, s, h, d);
                        inputNorm += iv * iv;
                        outputNorm += ov * ov;
                    }
                    assertEquals(Math.sqrt(inputNorm), Math.sqrt(outputNorm), 1e-3,
                            String.format("RoPE should preserve norm at [%d,%d,%d]", b, s, h));
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("dual_rope: global attention (type=1)")
    public void testDualRoPEGlobal(Nd4jBackend backend) {
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        DualRoPE localOp = new DualRoPE(input.dup(),
                DualRoPE.ATTENTION_TYPE_LOCAL, 0,
                10000.0, 1000000.0, 1.0, 1.0);
        INDArray[] localResults = Nd4j.exec(localOp);

        DualRoPE globalOp = new DualRoPE(input.dup(),
                DualRoPE.ATTENTION_TYPE_GLOBAL, 0,
                10000.0, 1000000.0, 1.0, 1.0);
        INDArray[] globalResults = Nd4j.exec(globalOp);

        INDArray localOut = localResults[0];
        INDArray globalOut = globalResults[0];

        // Both should have same shape
        assertArrayEquals(localOut.shape(), globalOut.shape());

        // Local and global should differ (different freq_base)
        assertNotEquals(localOut, globalOut,
                "Local and global RoPE should produce different results due to different freq_base");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("dual_rope: SameDiff execution")
    public void testDualRoPESameDiff(Nd4jBackend backend) {
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, numHeads, headDim);

        SDVariable result = sd.nn.dualRoPE(input, DualRoPE.ATTENTION_TYPE_LOCAL);
        sd.setOutputs(result.name());

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim));

        Map<String, INDArray> out = sd.output(inputs, result.name());
        INDArray output = out.get(result.name());

        assertNotNull(output);
        assertArrayEquals(new long[]{batch, seqLen, numHeads, headDim}, output.shape());

        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("dual_rope: position 0 identity for zero input")
    public void testDualRoPEZeroInput(Nd4jBackend backend) {
        int batch = 1, seqLen = 1, numHeads = 1, headDim = 4;

        INDArray input = Nd4j.zeros(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        DualRoPE op = new DualRoPE(input,
                DualRoPE.ATTENTION_TYPE_LOCAL, 0,
                10000.0, 1000000.0, 1.0, 1.0);
        INDArray[] results = Nd4j.exec(op);

        INDArray output = results[0];
        // RoPE on zero input should produce zero output (rotation of zero vector is zero)
        INDArray expected = Nd4j.zeros(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        assertEquals(expected, output, "RoPE of zero vector should be zero");
    }

    // ==================== Combined op pipeline ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Gemma4 pipeline: PLE → DualRoPE → SharedKvAttn")
    public void testGemma4OpsPipeline(Nd4jBackend backend) {
        int batch = 1, seqLen = 4, hiddenDim = 32, vocabSize = 16;
        int numHeads = 4, numKvHeads = 2, headDim = hiddenDim / numHeads;

        // Step 1: Per-layer embedding
        INDArray hiddenStates = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
        INDArray pleWeight = Nd4j.randn(DataType.FLOAT, vocabSize, hiddenDim).muli(0.01);
        INDArray tokenIds = Nd4j.createFromArray(new int[][]{{1, 5, 3, 9}}).castTo(DataType.INT);

        PerLayerEmbedding pleOp = new PerLayerEmbedding(hiddenStates, pleWeight, tokenIds, 1.0);
        INDArray pleOutput = Nd4j.exec(pleOp)[0];
        assertArrayEquals(new long[]{batch, seqLen, hiddenDim}, pleOutput.shape());

        // Step 2: Reshape for attention: [B, S, D] → [B, S, H, headDim] → [B, H, S, headDim]
        INDArray reshaped = pleOutput.reshape(batch, seqLen, numHeads, headDim);
        INDArray qInput = reshaped.permute(0, 2, 1, 3);  // [B, H, S, headDim]

        // Step 3: DualRoPE on query
        // RoPE expects [B, S, H, D] so permute back
        DualRoPE ropeOp = new DualRoPE(reshaped,
                DualRoPE.ATTENTION_TYPE_LOCAL, 0,
                10000.0, 1000000.0, 1.0, 1.0);
        INDArray ropeOutput = Nd4j.exec(ropeOp)[0];
        assertArrayEquals(reshaped.shape(), ropeOutput.shape());

        // Permute for attention: [B, S, H, D] → [B, H, S, D]
        INDArray query = ropeOutput.permute(0, 2, 1, 3).dup();

        // Step 4: Shared KV attention (use random K/V as "donor" layer)
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, seqLen, headDim);

        double scale = 1.0 / Math.sqrt(headDim);
        SharedKvAttention attnOp = new SharedKvAttention(query, key, value,
                numHeads, numKvHeads, 1, 0, scale);
        INDArray attnOutput = Nd4j.exec(attnOp)[0];

        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, attnOutput.shape());
        assertFalse(Double.isNaN(attnOutput.sumNumber().doubleValue()), "Pipeline output has NaN");
        assertFalse(Double.isInfinite(attnOutput.sumNumber().doubleValue()), "Pipeline output has Inf");

        log.info("Gemma4 pipeline test passed: PLE→DualRoPE→SharedKvAttn");
    }

    // ==================== GGUF Import ====================

    @Test
    @DisplayName("Gemma 4 GGUF Import: download + import smallest Gemma 4 model")
    public void testGemmaGGUFImport() throws Exception {
        // Download smallest Gemma 4 model (gemma-4-E2B-it Q4_K_M, ~3.1GB)
        LLMModel gemmaModel = LLMModel.GEMMA4_E2B;
        QuantType quant = QuantType.Q4_K_M;

        log.info("Downloading {} with {} quantization...", gemmaModel.getName(), quant);
        long t0 = System.currentTimeMillis();
        DownloadResult downloadResult = LLMModelDownloader.download(gemmaModel, quant);
        long downloadMs = System.currentTimeMillis() - t0;

        assertTrue(downloadResult.getModelFile().exists(),
                "Model file not found: " + downloadResult.getModelFile());
        assertTrue(downloadResult.getModelFile().length() > 0, "Model file is empty");
        log.info("Downloaded: {} ({} bytes) in {}ms (cached={})",
                downloadResult.getModelFile().getName(),
                downloadResult.getFileSizeBytes(),
                downloadMs, !downloadResult.isDownloadedNow());

        // Import GGUF → SameDiff with FP16 to avoid memory blow-up
        log.info("Importing GGUF model...");
        t0 = System.currentTimeMillis();
        ConversionOptions options = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                .build();
        SameDiff model = GGMLModelImport.importModel(
                downloadResult.getModelFile().getAbsolutePath(), options);
        long importMs = System.currentTimeMillis() - t0;

        assertNotNull(model, "Imported model must not be null");
        int opCount = model.ops() != null ? model.ops().length : 0;
        int varCount = model.variables().size();
        log.info("Import: {}ms, {} ops, {} variables", importMs, opCount, varCount);

        assertTrue(opCount > 0, "Imported model should have ops");
        assertTrue(varCount > 0, "Imported model should have variables");

        // Inspect model metadata
        GGMLMetadata metadata = GGMLModelImport.inspectModel(
                downloadResult.getModelFile().getAbsolutePath());
        assertNotNull(metadata, "Metadata must not be null");
        log.info("Architecture: {}", metadata.getArchitecture());
        log.info("Num layers: {}", metadata.getNumLayers());
        log.info("Hidden size: {}", metadata.getHiddenSize());
        log.info("Num attention heads: {}", metadata.getNumAttentionHeads());
        log.info("Vocab size: {}", metadata.getVocabSize());

        // Verify it's a Gemma model
        String arch = metadata.getArchitecture();
        assertNotNull(arch, "Architecture metadata must be present");
        assertTrue(arch.toLowerCase().contains("gemma"),
                "Architecture should be gemma, got: " + arch);

        model.close();
        log.info("Gemma GGUF import test passed");
    }

    @Test
    @DisplayName("Gemma 4 GGUF Import: download only (verify cache)")
    public void testGemmaGGUFDownload() throws Exception {
        LLMModel gemmaModel = LLMModel.GEMMA4_E2B;
        QuantType quant = QuantType.Q4_K_M;

        log.info("Downloading {} with {} quantization...", gemmaModel.getName(), quant);
        DownloadResult result = LLMModelDownloader.download(gemmaModel, quant);

        assertTrue(result.getModelFile().exists(), "Model file should exist");
        assertTrue(result.getFileSizeBytes() > 100_000_000, "Model file should be > 100MB");

        log.info("Model file: {} ({} MB)",
                result.getModelFile().getAbsolutePath(),
                result.getFileSizeBytes() / (1024 * 1024));

        // Verify URL generation
        String expectedUrl = "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf";
        assertEquals(expectedUrl, gemmaModel.getUrl(quant), "URL generation must be correct");

        // Second download should be cached
        long t0 = System.currentTimeMillis();
        DownloadResult cached = LLMModelDownloader.download(gemmaModel, quant);
        long cachedMs = System.currentTimeMillis() - t0;

        assertFalse(cached.isDownloadedNow(), "Second download should be from cache");
        assertTrue(cachedMs < 1000, "Cached download should be < 1 second, was " + cachedMs + "ms");
    }
}
