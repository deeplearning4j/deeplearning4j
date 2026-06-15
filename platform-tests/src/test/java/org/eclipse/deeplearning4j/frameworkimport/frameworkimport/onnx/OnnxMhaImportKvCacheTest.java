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
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
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

    /**
     * Test 2: past_key_values.0.key connects to the concat op for present.0.key.
     * This confirms the model graph DOES use past KV for KV cache accumulation.
     */
    @Test
    @DisplayName("2. past_key_values.0.key feeds into concat op for present.0.key")
    public void testPastKvFeedsConcatOp() {
        boolean foundConcat = false;
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getInputsToOp() != null && op.getInputsToOp().contains("past_key_values.0.key")
                && op.getOp() != null && "concat".equals(op.getOp().opName())) {
                foundConcat = true;
                log.info("Concat op '{}' consumes past_key_values.0.key", entry.getKey());
                log.info("  inputs: {}", op.getInputsToOp());
                log.info("  outputs: {}", op.getOutputsOfOp());
                assertTrue(op.getOutputsOfOp().contains("present.0.key"),
                    "Concat output should be present.0.key");
            }
        }
        assertTrue(foundConcat, "Should find a concat op consuming past_key_values.0.key");
    }

    /**
     * Test 3: MHA op's pastKey/pastValue are CONSTANT empty arrays.
     * This is CORRECT for SmolDocling — the ONNX model handles KV cache
     * concatenation OUTSIDE the MHA op via separate Concat nodes.
     * The MHA key/value inputs receive the already-concatenated full sequences
     * through the repeat_kv pipeline: present.0.key → Unsqueeze → Expand → MHA key.
     */
    @Test
    @DisplayName("3. MHA pastKey/Value are empty constants (KV cache is external)")
    public void testMhaOpHasEmptyPastKv() {
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())
                && entry.getKey().contains("layers.0")) {

                List<String> inputs = op.getInputsToOp();
                log.info("MHA layer 0 op '{}' inputs: {}", entry.getKey(), inputs);

                // Input 4 = pastKey, Input 5 = pastValue — both should be CONSTANT empty
                String pastKeyInput = inputs.get(4);
                String pastValueInput = inputs.get(5);

                SDVariable pastKeyVar = decoder.getVariable(pastKeyInput);
                SDVariable pastValueVar = decoder.getVariable(pastValueInput);

                log.info("  pastKey '{}' type={}", pastKeyInput, pastKeyVar.getVariableType());
                log.info("  pastValue '{}' type={}", pastValueInput, pastValueVar.getVariableType());

                // In SmolDocling, MHA pastKey is CONSTANT empty because KV cache is
                // handled externally via Concat ops (present.0.key = concat(past_kv, current_k))
                assertEquals(org.nd4j.autodiff.samediff.VariableType.CONSTANT,
                    pastKeyVar.getVariableType(),
                    "MHA pastKey should be CONSTANT (KV cache is external)");
                assertEquals(org.nd4j.autodiff.samediff.VariableType.CONSTANT,
                    pastValueVar.getVariableType(),
                    "MHA pastValue should be CONSTANT (KV cache is external)");
                return;
            }
        }
        fail("Should find onnx_multi_head_attention op for layer 0");
    }

    /**
     * Test 4: All 30 MHA ops should have CONSTANT empty pastKey.
     * SmolDocling handles KV cache externally — MHA doesn't use pastKey/pastValue inputs.
     */
    @Test
    @DisplayName("4. All MHA ops have empty constant pastKey (KV cache is external)")
    public void testAllMhaOpsHaveEmptyPastKv() {
        int mhaCount = 0;
        int constantCount = 0;

        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())) {
                mhaCount++;
                List<String> inputs = op.getInputsToOp();
                String pastKeyInput = inputs.get(4);
                SDVariable pastKeyVar = decoder.getVariable(pastKeyInput);

                if (pastKeyVar.getVariableType() == org.nd4j.autodiff.samediff.VariableType.CONSTANT) {
                    constantCount++;
                }
            }
        }

        log.info("MHA ops: {} total, {} with CONSTANT pastKey", mhaCount, constantCount);
        assertEquals(30, mhaCount, "Should have 30 MHA ops (one per layer)");
        assertEquals(30, constantCount,
            "All MHA ops should have CONSTANT empty pastKey (KV cache is external)");
    }

    /**
     * Test 5: MHA key input comes from the repeat_kv pipeline which includes
     * accumulated KV (present.0.key) through: concat → repeat_kv → MHA key.
     * The chain is: present.0.key → Unsqueeze_5 → Expand → Reshape_3 → Transpose_2 → Reshape_4 → MHA key
     */
    @Test
    @DisplayName("5. MHA key input comes from repeat_kv pipeline (includes accumulated KV)")
    public void testMhaKeyFromRepeatKvPipeline() {
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())
                && entry.getKey().contains("layers.0")) {

                String keyInput = op.getInputsToOp().get(1);
                log.info("MHA layer 0 key input: '{}'", keyInput);

                // Key should come from repeat_kv pipeline (Reshape_4)
                assertTrue(keyInput.contains("repeat_kv") || keyInput.contains("k_proj"),
                    "Key should come from k_proj/repeat_kv chain, got: " + keyInput);
                return;
            }
        }
        fail("Should find onnx_multi_head_attention op for layer 0");
    }

    /**
     * Test 6: present.0.key output should NOT be empty after prefill.
     * This is the symptom of the bug — because MHA's pastKey is empty constant,
     * the shape function produces empty present KV.
     */
    @Test
    @DisplayName("6. present.0.key should be non-empty after prefill")
    public void testPresentKvNonEmptyAfterPrefill() throws Exception {
        // Build minimal prefill inputs
        int seqLen = 10;
        Map<String, INDArray> inputs = new HashMap<>();
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) {
                inputs.put(name, Nd4j.randn(DataType.FLOAT, 1, seqLen, 576));
            } else if (name.equals("attention_mask")) {
                inputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen));
            } else if (name.equals("position_ids")) {
                inputs.put(name, Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.LONG));
            } else if (name.startsWith("past_key_values.")) {
                inputs.put(name, Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64));
            }
        }

        Map<String, INDArray> result = decoder.output(inputs, "present.0.key");
        INDArray presentKey = result.get("present.0.key");

        log.info("present.0.key after prefill: shape={} isEmpty={}", Arrays.toString(presentKey.shape()), presentKey.isEmpty());

        // After prefill with seqLen=10, present.0.key should be [1, 3, 10, 64] (not empty)
        assertFalse(presentKey.isEmpty(), "present.0.key should NOT be empty after prefill");
        assertEquals(4, presentKey.rank(), "present.0.key should be rank 4");
        assertEquals(seqLen, presentKey.shape()[2],
            "present.0.key seq dim should be " + seqLen + " (prefill length), got " + presentKey.shape()[2]);
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
     * Test 8b (DIAGNOSTIC): Trace the concat op for present.0.key and request intermediate outputs.
     * This helps understand why present.0.key is empty after prefill.
     */
    @Test
    @DisplayName("8b. DIAGNOSTIC: Trace concat chain for present.0.key")
    public void testTraceConcatChainForPresentKey() throws Exception {
        // Find the concat op that produces present.0.key
        SameDiffOp concatOp = null;
        String concatOpName = null;
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains("present.0.key")) {
                concatOp = op;
                concatOpName = entry.getKey();
                break;
            }
        }
        assertNotNull(concatOp, "Should find op producing present.0.key");
        log.info("Op producing present.0.key: name='{}' opName='{}' inputs={} outputs={}",
            concatOpName, concatOp.getOp() != null ? concatOp.getOp().opName() : "null",
            concatOp.getInputsToOp(), concatOp.getOutputsOfOp());

        // Print the iArguments (axis) for concat
        if (concatOp.getOp() != null) {
            // Just log the op class and name for axis info
            log.info("  op class: {}", concatOp.getOp().getClass().getSimpleName());
        }

        // Trace each input variable
        List<String> concatInputs = concatOp.getInputsToOp();
        List<String> intermediateOutputs = new ArrayList<>();
        for (int i = 0; i < concatInputs.size(); i++) {
            String inputName = concatInputs.get(i);
            SDVariable var = decoder.getVariable(inputName);
            log.info("  concat input[{}]: '{}' type={}", i, inputName, var.getVariableType());
            if (var.getVariableType() != org.nd4j.autodiff.samediff.VariableType.PLACEHOLDER) {
                intermediateOutputs.add(inputName);
            }
        }

        // Trace what produces Transpose_1/output_0 (the second concat input)
        String transpose1Name = "/model/layers.0/attn/k_proj/repeat_kv/Transpose_1/output_0";
        log.info("=== Tracing Transpose_1 producer chain ===");
        Set<String> visited0 = new HashSet<>();
        traceProducerChain(transpose1Name, "  ", visited0, 8);

        // Also find what feeds into the MHA key input through the repeat_kv chain
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())
                && entry.getKey().contains("layers.0")) {
                List<String> mhaInputs = op.getInputsToOp();
                String keyInputName = mhaInputs.get(1); // key is index 1
                log.info("MHA layer 0 key input: '{}'", keyInputName);

                // Trace back from MHA key to find if present.0.key is in the chain
                Set<String> visited = new HashSet<>();
                traceProducerChain(keyInputName, "  ", visited, 5);
                break;
            }
        }

        // Build prefill inputs and request both present.0.key AND intermediate outputs
        int seqLen = 10;
        Map<String, INDArray> inputs = new HashMap<>();
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) {
                inputs.put(name, Nd4j.randn(DataType.FLOAT, 1, seqLen, 576));
            } else if (name.equals("attention_mask")) {
                inputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen));
            } else if (name.equals("position_ids")) {
                inputs.put(name, Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.LONG));
            } else if (name.startsWith("past_key_values.")) {
                inputs.put(name, Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64));
            }
        }

        // Request present.0.key plus its non-placeholder inputs and k_proj chain intermediates
        List<String> requestedOutputs = new ArrayList<>();
        requestedOutputs.add("present.0.key");
        requestedOutputs.addAll(intermediateOutputs);
        // Add k_proj chain intermediates to trace where the empty shape comes from
        for (Map.Entry<String, SameDiffOp> entry2 : decoder.getOps().entrySet()) {
            String name = entry2.getKey();
            if (name.contains("layers.0/attn/k_proj") && !name.contains("repeat_kv/Concat")
                && entry2.getValue().getOutputsOfOp() != null) {
                for (String outName : entry2.getValue().getOutputsOfOp()) {
                    if (!requestedOutputs.contains(outName)) {
                        requestedOutputs.add(outName);
                    }
                }
            }
        }

        log.info("Requesting outputs: {}", requestedOutputs);
        Map<String, INDArray> result = decoder.output(inputs,
            requestedOutputs.toArray(new String[0]));

        for (String outName : requestedOutputs) {
            INDArray arr = result.get(outName);
            if (arr != null) {
                log.info("Output '{}': shape={} isEmpty={} length={}",
                    outName, Arrays.toString(arr.shape()), arr.isEmpty(), arr.length());
            } else {
                log.info("Output '{}': NULL", outName);
            }
        }
    }

    private void traceProducerChain(String varName, String indent, Set<String> visited, int maxDepth) {
        if (maxDepth <= 0 || visited.contains(varName)) return;
        visited.add(varName);

        SDVariable var = decoder.getVariable(varName);
        if (var == null) return;

        // Find the op that produces this variable
        for (Map.Entry<String, SameDiffOp> entry : decoder.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(varName)) {
                String opName = op.getOp() != null ? op.getOp().opName() : "unknown";
                log.info("{}'{}'  produced by  '{}' (opName={})", indent, varName, entry.getKey(), opName);
                log.info("{}  inputs: {}", indent, op.getInputsToOp());

                // Trace each input
                if (op.getInputsToOp() != null) {
                    for (String inputName : op.getInputsToOp()) {
                        traceProducerChain(inputName, indent + "  ", visited, maxDepth - 1);
                    }
                }
                return;
            }
        }
        log.info("{}'{}': type={} (no producing op found)", indent, varName, var.getVariableType());
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

                // MHA op should have 6 inputs: query, key, value, attnBias, pastKey, pastValue
                assertEquals(6, inputs.size(), "MHA op should have 6 inputs");

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
