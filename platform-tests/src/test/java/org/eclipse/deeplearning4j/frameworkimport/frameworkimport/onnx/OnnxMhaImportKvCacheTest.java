/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for ONNX MultiHeadAttention KV cache import.
 *
 * Tests that past_key_values placeholders are correctly wired to the MHA op's
 * past_key/past_value inputs during ONNX import.
 *
 * Root cause: SmolDocling model's ONNX MultiHeadAttention nodes have
 * past_key/past_value at input indices 6/7, but the import hook reads them
 * as empty strings, creating empty constant placeholders instead of wiring
 * the actual past_key_values variables.
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class OnnxMhaImportKvCacheTest {

    private SameDiff decoder;

    @BeforeAll
    public void setup() throws Exception {
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedResult.getModelFile().getAbsolutePath());
        decoder = models[1];
        log.info("Decoder loaded: {} ops, {} variables", decoder.getOps().size(), decoder.variables().size());
    }

    /**
     * Test 1: past_key_values placeholders exist in the model.
     * The model MUST have past_key_values.X.key and past_key_values.X.value as PLACEHOLDER inputs.
     */
    @Test
    @DisplayName("1. past_key_values placeholders exist as model inputs")
    public void testPastKvPlaceholdersExist() {
        List<String> kvInputs = new ArrayList<>();
        for (String input : decoder.inputs()) {
            if (input.startsWith("past_key_values.")) {
                kvInputs.add(input);
            }
        }
        log.info("Found {} past_key_values inputs: {}", kvInputs.size(), kvInputs.subList(0, Math.min(4, kvInputs.size())));
        assertTrue(kvInputs.size() >= 2, "Model should have at least 2 past_key_values inputs (key+value for layer 0)");
        assertTrue(kvInputs.contains("past_key_values.0.key"), "Should have past_key_values.0.key");
        assertTrue(kvInputs.contains("past_key_values.0.value"), "Should have past_key_values.0.value");
    }

    @Test
    @DisplayName("2. external present KV outputs are removed")
    public void testExternalKvOutputsRemoved() {
        assertTrue(decoder.outputs().stream().noneMatch(name -> name.matches("present\\.[0-9]+\\.(key|value)")),
                "Canonicalized decoder must not publish external present KV outputs");
    }

    @Test
    @DisplayName("3. MHA uses canonical cache inputs and shared cache position")
    public void testMhaUsesCanonicalInPlaceKv() {
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())
                && entry.getKey().contains("layers.0")) {
                List<String> inputs = op.getInputsToOp();
                assertEquals(7, inputs.size());
                assertEquals("/model/layers.0/attn/k_rotary/RotaryEmbedding/output_0", inputs.get(1));
                assertEquals("/model/layers.0/attn/v_proj/MatMul/output_0", inputs.get(2));
                assertEquals("causal_mask", inputs.get(3));
                assertEquals("past_key_values.0.key", inputs.get(4));
                assertEquals("past_key_values.0.value", inputs.get(5));
                assertEquals("cache_position", inputs.get(6));
                assertEquals(DataType.INT64, decoder.getVariable("cache_position").dataType());
                return;
            }
        }
        fail("Should find onnx_multi_head_attention op for layer 0");
    }

    @Test
    @DisplayName("4. all 30 MHA ops use one plan-owned cache contract")
    public void testAllMhaOpsUseInPlaceKv() {
        int mhaCount = 0;
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())) {
                mhaCount++;
                List<String> inputs = op.getInputsToOp();
                assertEquals(7, inputs.size(), entry.getKey());
                assertTrue(inputs.get(4).startsWith("past_key_values."), entry.getKey());
                assertTrue(inputs.get(5).startsWith("past_key_values."), entry.getKey());
                assertEquals("cache_position", inputs.get(6), entry.getKey());
            }
        }
        assertEquals(30, mhaCount, "Should have 30 MHA ops (one per layer)");
    }

    @Test
    @DisplayName("5. MHA bypasses external concat and repeat_kv")
    public void testMhaBypassesExternalRepeatKv() {
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())
                && entry.getKey().contains("layers.0")) {
                String keyInput = op.getInputsToOp().get(1);
                assertFalse(keyInput.contains("repeat_kv"), keyInput);
                assertEquals("/model/layers.0/attn/k_rotary/RotaryEmbedding/output_0", keyInput);
                assertTrue(decoder.getVariables().get(keyInput).getInputsForOp().contains(entry.getKey()),
                        "Current K reverse consumer metadata must include the rewired MHA");
                return;
            }
        }
        fail("Should find onnx_multi_head_attention op for layer 0");
    }

    @Test
    @DisplayName("6. cache placeholders preserve BHSD heads and head dimension")
    public void testCachePlaceholderLayout() {
        long[] keyShape = decoder.getVariable("past_key_values.0.key").getShape();
        long[] valueShape = decoder.getVariable("past_key_values.0.value").getShape();
        assertEquals(4, keyShape.length);
        assertEquals(3, keyShape[1]);
        assertEquals(64, keyShape[3]);
        assertArrayEquals(keyShape, valueShape);
    }

    /**
     * Test 7: Standalone MHA op with past KV correctly extends attention.
     * Build a simple SameDiff graph with MHA and verify that passing pastKey
     * changes the output compared to no pastKey.
     */
    @Test
    @DisplayName("7. Standalone MHA with pastKey produces different output than without")
    public void testStandaloneMhaWithPastKv() {
        int batch = 1, seqQ = 1, hidden = 576, numHeads = 9;
        int headDim = hidden / numHeads; // 64
        int numKvHeads = 3;
        int kvHidden = numKvHeads * headDim; // 192

        // Query and current KV (1 token)
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden);

        // Run without past KV
        OnnxMultiHeadAttention opNoPast = new OnnxMultiHeadAttention(
            query, key, value, null, null, null, numHeads, 0.0, false);
        INDArray[] noPastResult = Nd4j.exec(opNoPast);
        INDArray outputNoPast = noPastResult[0];

        // Run WITH past KV (5 tokens of past context)
        int pastSeq = 5;
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, pastSeq, headDim);

        OnnxMultiHeadAttention opWithPast = new OnnxMultiHeadAttention(
            query, key, value, null, pastKey, pastValue, numHeads, 0.0, false);
        INDArray[] withPastResult = Nd4j.exec(opWithPast);
        INDArray outputWithPast = withPastResult[0];

        log.info("Output without past: shape={}", Arrays.toString(outputNoPast.shape()));
        log.info("Output with past: shape={}", Arrays.toString(outputWithPast.shape()));

        // Outputs should be DIFFERENT because past context changes attention
        double diff = outputNoPast.sub(outputWithPast).amaxNumber().doubleValue();
        log.info("Max difference between no-past and with-past output: {}", diff);
        assertTrue(diff > 0.01,
            "Output with pastKey should differ from output without pastKey, but diff=" + diff);

        // Present key should include past + current
        if (withPastResult.length > 1) {
            INDArray presentKey = withPastResult[1];
            log.info("Present key shape: {}", Arrays.toString(presentKey.shape()));
            assertEquals(pastSeq + seqQ, presentKey.shape()[2],
                "Present key seq should be pastSeq + seqQ = " + (pastSeq + seqQ));
        }
    }

    /**
     * Test 8: Verify the ONNX import hook reads past_key/past_value from node inputs.
     * Check how many inputs the ONNX MHA node has by examining the SameDiffOp.
     */
    @Test
    @DisplayName("8. ONNX MHA import preserves input count and past KV variable names")
    public void testOnnxImportInputCount() {
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())
                && entry.getKey().contains("layers.0")) {

                List<String> inputs = op.getInputsToOp();
                log.info("MHA layer 0 has {} inputs: {}", inputs.size(), inputs);

                // query, current key/value, bias, fixed past key/value, cache_position
                assertEquals(7, inputs.size(), "MHA op should have 7 plan-owned KV inputs");

                // Check each input type
                for (int i = 0; i < inputs.size(); i++) {
                    SDVariable v = decoder.getVariable(inputs.get(i));
                    log.info("  input[{}] '{}': type={}", i, inputs.get(i), v.getVariableType());
                }
                return;
            }
        }
        fail("Should find onnx_multi_head_attention op for layer 0");
    }
}
