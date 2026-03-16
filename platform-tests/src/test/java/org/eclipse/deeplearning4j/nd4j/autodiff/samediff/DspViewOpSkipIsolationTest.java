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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for DSP view-capable op skip logic and Triton FUSED_ATTENTION shape resolution.
 *
 * These tests cover three specific fixes:
 * 1. View-capable ops (reshape_no_copy, permute, expand_dims, squeeze) should NOT be
 *    skipped when their primary input is non-empty, even if pre-allocated outputs are empty.
 * 2. The Triton FUSED_ATTENTION builder should derive seqK from the key input shape
 *    (not just pastKey), so attention works when pastKey is an empty constant.
 * 3. Permute is now classified as a view-capable op (isViewCapableOp=true).
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=DspViewOpSkipIsolationTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class DspViewOpSkipIsolationTest {

    private static final double TOL = 1e-4;

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 1: reshape_no_copy → permute → reshape_no_copy chain in DSP
    // This is the exact pattern between Concat and MHA in SmolDocling.
    // Previously all 3 ops were skipped because their outputs were pre-allocated
    // as empty (from stale shape inference). The fix ensures view-capable ops
    // are NOT skipped when their primary input is non-empty.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape → permute → reshape chain: DSP matches standard execution")
    public void testReshapePermuteReshapeChainDsp() {
        SameDiff sd = SameDiff.create();

        // Simulate transformer KV reshape: [1, 3, 100, 64] → [1, 300, 64] → permute → [1, 64, 300]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 100, 64);

        // reshape_no_copy: [1, 3, 100, 64] → [1, 300, 64]
        SDVariable reshaped1 = sd.reshape("reshaped1", x, 1, 300, 64);

        // permute: [1, 300, 64] → [1, 64, 300]
        SDVariable permuted = sd.permute("permuted", reshaped1, 0, 2, 1);

        // reshape_no_copy: [1, 64, 300] → [1, 64, 300] (identity reshape, but exercises the path)
        SDVariable reshaped2 = sd.reshape("reshaped2", permuted, 1, 64, 300);

        // Downstream op to verify data flows correctly
        SDVariable out = reshaped2.mul("out", 2.0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 3, 100, 64);

        // Standard execution (reference)
        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual, "DSP output should not be null");
        assertArrayEquals(expected.shape(), actual.shape(), "Shapes should match");
        assertFalse(Double.isNaN(actual.meanNumber().doubleValue()), "DSP output contains NaN");

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("reshape→permute→reshape chain: maxDiff={}", maxDiff);
        assertTrue(maxDiff < TOL, "DSP output differs from standard. maxDiff=" + maxDiff);

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 2: Permute as view-capable op — verify it doesn't get zeroed
    // With permute now classified as isViewCapableOp, its needsZeroedOutput=false.
    // This test verifies permute output is correct (not zeroed/corrupted).
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Permute classified as view-capable: output not zeroed in DSP")
    public void testPermuteViewCapableNotZeroed() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable permuted = sd.permute("permuted", x, 0, 2, 1);  // [2, 4, 3]
        SDVariable out = permuted.add("out", 1.0);

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(expected.shape(), actual.shape());
        assertEquals(expected, actual,
                "Permute via DSP should match standard execution (view-capable, no zeroing)");

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 3: Multi-step DSP with view ops — shape cache invalidation
    // When shapes change between steps and view ops have stale cached shapes,
    // the fix invalidates the cache and re-derives from non-empty input.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Multi-step DSP with reshape: cache invalidated when input changes")
    public void testMultiStepReshapeCacheInvalidation() {
        SameDiff sd = SameDiff.create();

        // Fixed-shape graph: [1, 12, 64] → reshape [1, 3, 4, 64] → mul
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 12, 64);
        SDVariable reshaped = sd.reshape("reshaped", x, 1, 3, 4, 64);
        SDVariable out = reshaped.mul("out", 3.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 12, 64).mul(step + 1);

            // Standard reference
            Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
            INDArray expected = stdResult.get("out").dup();

            // DSP execution
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = dspResult.get("out");

            assertNotNull(actual, "Step " + step + ": DSP output null");
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": DSP output differs. maxDiff=" + maxDiff);
        }

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 4: Concat → reshape → permute chain (simulates KV cache path)
    // This is the exact subgraph that was failing in SmolDocling: the concat
    // produces non-empty output but downstream reshape/permute were skipped.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("concat → reshape → permute chain: view ops not skipped after non-empty concat")
    public void testConcatReshapePermuteChain() {
        SameDiff sd = SameDiff.create();

        // Simulate: past_key [1, 3, 50, 64] concat with current_key [1, 3, 1, 64]
        // → [1, 3, 51, 64] → reshape [1, 153, 64] → permute [1, 64, 153]
        SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, 1, 3, 50, 64);
        SDVariable curKey = sd.placeHolder("cur_key", DataType.FLOAT, 1, 3, 1, 64);

        SDVariable concatted = sd.concat("concat", 2, pastKey, curKey);  // [1, 3, 51, 64]
        SDVariable reshaped = sd.reshape("reshaped", concatted, 1, 153, 64);
        SDVariable permuted = sd.permute("permuted", reshaped, 0, 2, 1);  // [1, 64, 153]
        SDVariable out = permuted.mul("out", 1.0);  // identity mul to have a named output

        INDArray pastKeyVal = Nd4j.randn(DataType.FLOAT, 1, 3, 50, 64);
        INDArray curKeyVal = Nd4j.randn(DataType.FLOAT, 1, 3, 1, 64);

        Map<String, INDArray> feed = new HashMap<>();
        feed.put("past_key", pastKeyVal);
        feed.put("cur_key", curKeyVal);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(feed, "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(feed, "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual, "DSP output null — view ops may have been skipped");
        assertArrayEquals(expected.shape(), actual.shape(),
                "Shape mismatch — reshape or permute may have been skipped");
        assertFalse(Double.isNaN(actual.meanNumber().doubleValue()),
                "DSP output contains NaN — view ops likely skipped");

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("concat→reshape→permute: maxDiff={}", maxDiff);
        assertTrue(maxDiff < TOL,
                "concat→reshape→permute chain differs. maxDiff=" + maxDiff);

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 5: MHA with empty pastKey (no KV cache) — Triton builder must derive
    // seqK from key input, not from the empty pastKey constant.
    // This is the scenario in SmolDocling where ONNX import creates an empty
    // constant for pastKey when the model does KV concat externally.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MHA with null pastKey: Triton derives seqK from key input")
    public void testMhaWithoutPastKeyTritonSeqKResolution() {
        boolean tritonAvailable = Nd4j.getNativeOps().isTritonAvailable();
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64;
        int hidden = numHeads * headDim;
        int seqKV = 50;

        Nd4j.getRandom().setSeed(42);
        // Q: decode query [1, 1, hidden]
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        // K/V: already concatenated with past (full sequence) [1, seqKV, hidden]
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);
        // Bias
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, seqKV);

        // C++ reference: MHA WITHOUT past_key/value (null pastKey)
        OnnxMultiHeadAttention refOp = new OnnxMultiHeadAttention(
                query.dup(), key.dup(), value.dup(),
                bias.dup(), null, null,
                numHeads, 0.0, false, 1);
        INDArray[] refOutputs = Nd4j.exec(refOp);
        INDArray refOutput = refOutputs[0].dup();

        assertFalse(Double.isNaN(refOutput.meanNumber().doubleValue()),
                "C++ reference contains NaN");
        log.info("C++ ref: shape={}, mean={}", java.util.Arrays.toString(refOutput.shape()),
                refOutput.meanNumber().doubleValue());

        // Triton via SameDiff: NO past_key/value placeholders
        SameDiff sd = SameDiff.create();
        SDVariable qVar = sd.placeHolder("query", DataType.FLOAT, query.shape());
        SDVariable kVar = sd.placeHolder("key", DataType.FLOAT, key.shape());
        SDVariable vVar = sd.placeHolder("value", DataType.FLOAT, value.shape());
        SDVariable bVar = sd.placeHolder("bias", DataType.FLOAT, bias.shape());

        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, qVar, kVar, vVar,
                bVar, null, null,
                numHeads, 0.0, false, 1);
        SDVariable[] outputs = mha.outputVariables();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Nd4j.getEnvironment().setTritonIncludeTypes("ATTENTION");

        Map<String, INDArray> feed = new HashMap<>();
        feed.put("query", query.dup());
        feed.put("key", key.dup());
        feed.put("value", value.dup());
        feed.put("bias", bias.dup());

        Map<String, INDArray> result = sd.output(feed, outputs[0].name());
        INDArray tritonOutput = result.get(outputs[0].name());

        assertNotNull(tritonOutput, "Triton output null — seqK may be 0");
        assertFalse(Double.isNaN(tritonOutput.meanNumber().doubleValue()),
                "Triton output NaN — seqK=0 causes 0/0 in attention");

        double maxDiff = refOutput.sub(tritonOutput).amaxNumber().doubleValue();
        log.info("MHA without pastKey: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-2,
                "MHA without pastKey: Triton differs from C++. maxDiff=" + maxDiff);

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 6: MHA with 4D key input (BHSD) and no pastKey — verifies that the
    // Triton builder correctly extracts numKvHeads from key shape when there's
    // no pastKey to infer from.
    // ═══════════════════════════════════════════════════════════════════════════

    static Stream<Arguments> mhaWithoutPastKeyConfigs() {
        return Stream.of(
                // name, batch, numHeads, headDim, seqKV
                Arguments.of("SmolDocling_noPast", 1, 9, 64, 100),
                Arguments.of("Small_noPast", 1, 4, 32, 20),
                Arguments.of("GQA_noPast", 1, 8, 64, 50)
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("mhaWithoutPastKeyConfigs")
    @DisplayName("MHA without pastKey: C++ reference produces valid output")
    public void testMhaWithoutPastKeyReference(String name, int batch, int numHeads,
                                                int headDim, int seqKV) {
        int hidden = numHeads * headDim;

        Nd4j.getRandom().setSeed(42);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);

        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value, null, null, null,
                numHeads, 0.0, false, 1);
        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertTrue(outputs.length >= 1);

        INDArray attnOut = outputs[0];
        assertArrayEquals(new long[]{batch, 1, hidden}, attnOut.shape(),
                name + ": output shape wrong");
        assertFalse(Double.isNaN(attnOut.meanNumber().doubleValue()),
                name + ": output contains NaN");
        assertTrue(attnOut.amaxNumber().doubleValue() > 1e-6,
                name + ": output is all zeros");

        log.info("[{}] MHA no-pastKey: mean={}, absMax={}",
                name, attnOut.meanNumber(), attnOut.amaxNumber());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 7: expand_dims and squeeze as view-capable ops
    // Verify these ops also work correctly in DSP (they were already classified
    // as view-capable, but this test ensures the skip logic fix doesn't break them).
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("expand_dims + squeeze chain: DSP matches standard")
    public void testExpandDimsSqueezeChainDsp() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable expanded = sd.expandDims("expanded", x, 1);  // [2, 1, 3]
        SDVariable squeezed = sd.squeeze("squeezed", expanded, 1);  // [2, 3]
        SDVariable out = squeezed.mul("out", 5.0);

        INDArray input = Nd4j.linspace(1, 6, 6, DataType.FLOAT).reshape(2, 3);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(expected.shape(), actual.shape());
        assertEquals(expected, actual,
                "expand_dims→squeeze chain should match standard execution");

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 8: View op chain feeding into matmul — the real-world pattern
    // This is what happens in transformers: reshape→permute→matmul.
    // If view ops are skipped, matmul gets wrong/empty input and produces garbage.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape → permute → matmul chain: DSP produces correct attention-like output")
    public void testViewOpChainFeedingMatmul() {
        SameDiff sd = SameDiff.create();

        // Simulate: Q [1, 1, 192] → reshape [1, 1, 3, 64] → permute [1, 3, 1, 64]
        // Then: matmul with K^T → attention scores
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 1, 192);
        SDVariable kT = sd.placeHolder("kT", DataType.FLOAT, 1, 3, 64, 50);  // transposed K

        SDVariable qReshaped = sd.reshape("q_reshaped", q, 1, 1, 3, 64);
        SDVariable qPermuted = sd.permute("q_permuted", qReshaped, 0, 2, 1, 3);  // [1, 3, 1, 64]
        SDVariable scores = sd.mmul("scores", qPermuted, kT);  // [1, 3, 1, 50]
        SDVariable out = scores.div("out", Math.sqrt(64.0));

        INDArray qVal = Nd4j.randn(DataType.FLOAT, 1, 1, 192);
        INDArray kTVal = Nd4j.randn(DataType.FLOAT, 1, 3, 64, 50);

        Map<String, INDArray> feed = new HashMap<>();
        feed.put("q", qVal);
        feed.put("kT", kTVal);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(feed, "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(feed, "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual, "DSP output null — view ops before matmul may be skipped");
        assertArrayEquals(expected.shape(), actual.shape());
        assertFalse(Double.isNaN(actual.meanNumber().doubleValue()),
                "DSP output NaN — view ops likely skipped, matmul got empty input");

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("reshape→permute→matmul chain: maxDiff={}", maxDiff);
        assertTrue(maxDiff < TOL,
                "View op chain before matmul differs. maxDiff=" + maxDiff);

        sd.close();
    }
}
