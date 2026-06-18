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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduces error 700 in native-only CUDA graph replay when intermediate
 * buffer addresses drift between capture and replay.
 *
 * <h3>Root cause under investigation</h3>
 * Native-only CUDA graph capture (triggered when gap slots are present)
 * records cuBLAS kernels with device pointers baked into the graph nodes.
 * On replay, if any intermediate buffer gets a different device address
 * (because the ND4J memory pool hands out a different allocation), the
 * baked-in pointers are stale and the GPU reads garbage → error 700.
 *
 * <h3>VLM reproduction pattern</h3>
 * The VLM vision encoder has:
 * <ul>
 *   <li>391 slots, 134 gap slots (cuBLAS matmul) → forces native-only capture</li>
 *   <li>322 external inputs (many are model weights marked as variable)</li>
 *   <li>889 CUDA graph nodes</li>
 *   <li>Error 700 on execution count 3 (first replay after capture on exec 2)</li>
 * </ul>
 *
 * This test builds a graph with similar characteristics:
 * many placeholders whose INDArray instances are re-created each step
 * (simulating address drift), interleaved with cuBLAS matmul ops.
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DspNativeOnlyReplayDriftTest {

    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    /**
     * Build a VLM-like graph with many external inputs (placeholders)
     * and cuBLAS gap ops (matmul). Each "layer" has:
     *   - weight placeholder (simulating model weight as variable external input)
     *   - bias placeholder
     *   - matmul (cuBLAS gap op)
     *   - add (Triton-eligible)
     *   - rmsNorm (Triton-eligible)
     *
     * This creates a mixed graph that forces native-only capture.
     */
    private SameDiff buildVlmLikeGraph(int embedDim, int numLayers) {
        SameDiff g = SameDiff.create();
        SDVariable input = g.placeHolder("pixel_values", DataType.FLOAT, 1, embedDim);

        SDVariable x = input;
        for (int layer = 0; layer < numLayers; layer++) {
            String p = "enc_l" + layer + "_";
            // Weight and bias as PLACEHOLDERS (not constants) — simulates
            // model weights marked as variable external inputs, which is
            // what happens in the VLM ONNX import
            SDVariable w = g.placeHolder(p + "weight", DataType.FLOAT, embedDim, embedDim);
            SDVariable b = g.placeHolder(p + "bias", DataType.FLOAT, 1, embedDim);
            SDVariable gamma = g.placeHolder(p + "gamma", DataType.FLOAT, embedDim);

            // matmul = cuBLAS gap op (not Triton-eligible)
            SDVariable mm = g.mmul(p + "qkv", x, w);
            // add = Triton-eligible
            SDVariable added = mm.add(p + "bias_add", b);
            // rmsNorm = Triton-eligible
            x = g.nn().rmsNorm(p + "norm", added, gamma, 1e-5);
        }

        // Final projection (another cuBLAS gap)
        SDVariable wFinal = g.placeHolder("head_weight", DataType.FLOAT, embedDim, 32);
        g.mmul("out", x, wFinal);
        return g;
    }

    /**
     * Build a VLM vision encoder-like graph that includes gather, concat,
     * and split ops — these produce temporary intermediate buffers and
     * H2D pointer-array copies during CUDA graph capture. The VLM's
     * seg[395-785] has 889 graph nodes and fails on replay because
     * intermediate buffer addresses drift.
     *
     * Key VLM ops that create intermediates:
     *   - gather: copies host pointer array to device (CAPTURE_H2D)
     *   - concat: copies host pointer array + shape info to device
     *   - split: similar to concat
     *   - reshape/permute: creates view ops with new NDArray wrappers
     */
    private SameDiff buildVlmVisionEncoderGraph(int embedDim, int numHeads, int numLayers) {
        SameDiff g = SameDiff.create();
        int headDim = embedDim / numHeads;

        // Patch embeddings (input)
        SDVariable patches = g.placeHolder("pixel_values", DataType.FLOAT, 1, 49, embedDim);

        SDVariable x = patches;
        for (int layer = 0; layer < numLayers; layer++) {
            String p = "enc_l" + layer + "_";
            // QKV projection weights as placeholders (VLM pattern)
            SDVariable wQ = g.placeHolder(p + "wQ", DataType.FLOAT, embedDim, embedDim);
            SDVariable wK = g.placeHolder(p + "wK", DataType.FLOAT, embedDim, embedDim);
            SDVariable wV = g.placeHolder(p + "wV", DataType.FLOAT, embedDim, embedDim);
            SDVariable wO = g.placeHolder(p + "wO", DataType.FLOAT, embedDim, embedDim);
            SDVariable bQ = g.placeHolder(p + "bQ", DataType.FLOAT, embedDim);
            SDVariable bK = g.placeHolder(p + "bK", DataType.FLOAT, embedDim);
            SDVariable bV = g.placeHolder(p + "bV", DataType.FLOAT, embedDim);
            SDVariable gamma1 = g.placeHolder(p + "gamma1", DataType.FLOAT, embedDim);
            SDVariable gamma2 = g.placeHolder(p + "gamma2", DataType.FLOAT, embedDim);
            SDVariable wFF1 = g.placeHolder(p + "wFF1", DataType.FLOAT, embedDim, embedDim * 4);
            SDVariable wFF2 = g.placeHolder(p + "wFF2", DataType.FLOAT, embedDim * 4, embedDim);

            // Layer norm (Triton-eligible)
            SDVariable normed = g.nn().rmsNorm(p + "norm1",
                    g.reshape(p + "flatN", x, -1, embedDim), gamma1, 1e-5);

            // QKV matmuls (cuBLAS gap ops)
            SDVariable q = g.mmul(p + "q", normed, wQ);     // gap
            SDVariable k = g.mmul(p + "k", normed, wK);     // gap
            SDVariable v = g.mmul(p + "v", normed, wV);     // gap

            // Add biases (Triton-eligible)
            q = q.add(p + "qb", g.reshape(p + "bQr", bQ, 1, embedDim));
            k = k.add(p + "kb", g.reshape(p + "bKr", bK, 1, embedDim));
            v = v.add(p + "vb", g.reshape(p + "bVr", bV, 1, embedDim));

            // Reshape to multi-head: [batch*seq, embed] -> [batch, heads, seq, headDim]
            // These reshapes create view ops with intermediate NDArrays
            SDVariable qH = g.reshape(p + "qH", q, 1, 49, numHeads, headDim);
            qH = g.permute(p + "qP", qH, 0, 2, 1, 3); // [1, heads, 49, headDim]
            SDVariable kH = g.reshape(p + "kH", k, 1, 49, numHeads, headDim);
            kH = g.permute(p + "kP", kH, 0, 2, 3, 1); // [1, heads, headDim, 49] for score
            SDVariable vH = g.reshape(p + "vH", v, 1, 49, numHeads, headDim);
            vH = g.permute(p + "vP", vH, 0, 2, 1, 3); // [1, heads, 49, headDim]

            // Attention scores (cuBLAS gap)
            SDVariable scores = g.mmul(p + "attn", qH, kH); // gap
            // Scale (Triton-eligible)
            scores = scores.div(p + "scale", g.constant(p + "sDim", (float) Math.sqrt(headDim)));
            // Softmax (Triton-eligible or native)
            scores = g.nn().softmax(p + "sm", scores, -1);
            // Attention output (cuBLAS gap)
            SDVariable attnOut = g.mmul(p + "attnOut", scores, vH); // gap

            // Reshape back: [1, heads, 49, headDim] -> [49, embed]
            attnOut = g.permute(p + "attnOutP", attnOut, 0, 2, 1, 3);
            attnOut = g.reshape(p + "attnOutR", attnOut, 49, embedDim);

            // Output projection (cuBLAS gap)
            SDVariable projected = g.mmul(p + "proj", attnOut, wO); // gap

            // Residual + layer norm 2
            SDVariable xFlat = g.reshape(p + "xFlat", x, -1, embedDim);
            SDVariable residual = xFlat.add(p + "res1", projected);
            SDVariable normed2 = g.nn().rmsNorm(p + "norm2", residual, gamma2, 1e-5);

            // FFN (2 cuBLAS gaps)
            SDVariable ff1 = g.mmul(p + "ff1", normed2, wFF1); // gap
            SDVariable ffAct = g.nn().gelu(p + "gelu", ff1);     // Triton-eligible
            SDVariable ff2 = g.mmul(p + "ff2", ffAct, wFF2);    // gap

            // Residual
            x = residual.add(p + "res2", ff2);
            x = g.reshape(p + "outR", x, 1, 49, embedDim);
        }

        // Final output projection
        SDVariable wHead = g.placeHolder("head_weight", DataType.FLOAT, embedDim, 32);
        SDVariable xFinal = g.reshape("xFinal", x, -1, embedDim);
        g.mmul("out", xFinal, wHead);
        return g;
    }

    private Map<String, INDArray> buildVisionPlaceholders(int embedDim, int numHeads, int numLayers) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("pixel_values", Nd4j.randn(DataType.FLOAT, 1, 49, embedDim).muli(0.1f));

        for (int layer = 0; layer < numLayers; layer++) {
            String p = "enc_l" + layer + "_";
            ph.put(p + "wQ", Nd4j.randn(DataType.FLOAT, embedDim, embedDim).muli(0.01f));
            ph.put(p + "wK", Nd4j.randn(DataType.FLOAT, embedDim, embedDim).muli(0.01f));
            ph.put(p + "wV", Nd4j.randn(DataType.FLOAT, embedDim, embedDim).muli(0.01f));
            ph.put(p + "wO", Nd4j.randn(DataType.FLOAT, embedDim, embedDim).muli(0.01f));
            ph.put(p + "bQ", Nd4j.randn(DataType.FLOAT, embedDim).muli(0.01f));
            ph.put(p + "bK", Nd4j.randn(DataType.FLOAT, embedDim).muli(0.01f));
            ph.put(p + "bV", Nd4j.randn(DataType.FLOAT, embedDim).muli(0.01f));
            ph.put(p + "gamma1", Nd4j.ones(DataType.FLOAT, embedDim));
            ph.put(p + "gamma2", Nd4j.ones(DataType.FLOAT, embedDim));
            ph.put(p + "wFF1", Nd4j.randn(DataType.FLOAT, embedDim, embedDim * 4).muli(0.01f));
            ph.put(p + "wFF2", Nd4j.randn(DataType.FLOAT, embedDim * 4, embedDim).muli(0.01f));
        }
        ph.put("head_weight", Nd4j.randn(DataType.FLOAT, embedDim, 32).muli(0.01f));
        return ph;
    }

    /**
     * Build placeholders with FRESH INDArray allocations each call.
     * This simulates the VLM pattern where Java-side arrays get new
     * device addresses each step (because they come from the memory pool).
     */
    private Map<String, INDArray> buildFreshPlaceholders(int embedDim, int numLayers) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("pixel_values", Nd4j.randn(DataType.FLOAT, 1, embedDim).muli(0.1f));

        for (int layer = 0; layer < numLayers; layer++) {
            String p = "enc_l" + layer + "_";
            ph.put(p + "weight", Nd4j.randn(DataType.FLOAT, embedDim, embedDim).muli(0.01f));
            ph.put(p + "bias", Nd4j.randn(DataType.FLOAT, 1, embedDim).muli(0.01f));
            ph.put(p + "gamma", Nd4j.ones(DataType.FLOAT, embedDim));
        }
        ph.put("head_weight", Nd4j.randn(DataType.FLOAT, embedDim, 32).muli(0.01f));
        return ph;
    }

    /**
     * Build placeholders with REUSED INDArray instances (stable addresses).
     * Only the input data changes; weight/bias arrays are created once.
     */
    private Map<String, INDArray> buildStablePlaceholders(int embedDim, int numLayers) {
        return buildFreshPlaceholders(embedDim, numLayers); // first call creates them
    }

    private void configureMode(SameDiff g, GraphExecutionMode mode) {
        g.getSessions().clear();
        g.setGraphExecutionMode(mode);
        g.setDspAutoCompileEnabled(true);
        g.setDspNativeAutoCompileEnabled(true);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 1: Native-only replay with stable placeholder addresses
    //
    // Baseline: same INDArray instances reused each step. Should succeed
    // because device addresses don't drift.
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(1)
    void test1_stableAddressesReplaySucceeds() {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.AUTO);

        // Create placeholders ONCE, reuse them
        Map<String, INDArray> ph = buildStablePlaceholders(embedDim, numLayers);

        INDArray lastOut = null;
        for (int i = 0; i < 15; i++) {
            // Only change the input data, keep same arrays
            ph.get("pixel_values").assign(Nd4j.randn(DataType.FLOAT, 1, embedDim).muli(0.1f));
            Map<String, INDArray> out = g.output(ph, "out");
            lastOut = out.get("out").dup();
            assertNotNull(lastOut, "step " + i + ": output is null");
            assertFalse(lastOut.isNaN().any(), "step " + i + ": output contains NaN");
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int planPhase = DspPlanAssertions.getPlanPhase(g);
        log.info("STABLE: totalReplays={} planPhase={}", totalReplays, planPhase);

        assertTrue(totalReplays > 0,
                "STABLE: Expected graph replays > 0 but got " + totalReplays
                        + " planPhase=" + planPhase);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 2: Native-only replay with FRESH placeholder allocations each step
    //
    // This is the VLM pattern: each step creates new INDArray instances
    // for placeholders, meaning their device buffer addresses may differ.
    // The DSP staging buffer mechanism should handle this via D2D copies
    // to plan-owned stable buffers. If native cuBLAS kernels read from
    // baked-in addresses instead of staged addresses, we get error 700.
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(2)
    void test2_freshAllocationsReplaySucceeds() {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.AUTO);

        INDArray lastOut = null;
        for (int i = 0; i < 15; i++) {
            // Create FRESH placeholders each step — different device addresses
            Map<String, INDArray> ph = buildFreshPlaceholders(embedDim, numLayers);
            Map<String, INDArray> out = g.output(ph, "out");
            lastOut = out.get("out").dup();
            assertNotNull(lastOut, "step " + i + ": output is null");
            assertFalse(lastOut.isNaN().any(), "step " + i + ": output contains NaN");
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int planPhase = DspPlanAssertions.getPlanPhase(g);
        log.info("FRESH: totalReplays={} planPhase={}", totalReplays, planPhase);

        assertTrue(totalReplays > 0,
                "FRESH: Expected graph replays > 0 but got " + totalReplays
                        + " planPhase=" + planPhase
                        + ". Native-only CUDA graph replay fails when external input "
                        + "addresses drift — staging buffers may not be wired to native "
                        + "graph kernel arguments.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 3: Output correctness: fresh-address replay matches SBS reference
    //
    // Even if replay fires, check that the output matches slot-by-slot
    // execution — a successful replay with wrong addresses could silently
    // produce garbage without error 700 if the stale address happens to
    // be readable but contains wrong data.
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(3)
    void test3_freshAllocationsOutputMatchesSbs() {
        int embedDim = 64, numLayers = 8;

        // Phase 1: compute SBS reference with fixed seed
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.SLOT_BY_SLOT);

        // Use deterministic values for reproducibility
        Nd4j.getRandom().setSeed(42);
        Map<String, INDArray> refPh = buildFreshPlaceholders(embedDim, numLayers);

        // Warm up SBS
        for (int i = 0; i < 5; i++) {
            g.output(refPh, "out");
        }
        INDArray refOut = g.output(refPh, "out").get("out").dup();

        // Phase 2: switch to DSP mode, run to replay, then check output
        configureMode(g, GraphExecutionMode.AUTO);

        // Warm up DSP with various inputs
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> warmPh = buildFreshPlaceholders(embedDim, numLayers);
            g.output(warmPh, "out");
        }

        // Now run with the same inputs as reference
        Map<String, INDArray> dspOut = g.output(refPh, "out");
        INDArray dspResult = dspOut.get("out");

        double maxDiff = Transforms.abs(refOut.sub(dspResult)).maxNumber().doubleValue();
        log.info("CORRECTNESS: maxDiff={} refNorm={} dspNorm={}",
                maxDiff, refOut.norm2Number(), dspResult.norm2Number());

        assertTrue(maxDiff < 1e-2,
                "Output mismatch between SBS and DSP: maxDiff=" + maxDiff
                        + ". Native cuBLAS replay may be using stale intermediate pointers.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 4: Large graph with many external inputs — VLM scale
    //
    // Scale up to match VLM characteristics:
    // ~300+ external inputs, 20+ layers, cuBLAS gaps throughout.
    // This is the most likely to reproduce the error 700.
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(4)
    void test4_largeGraphManyExternalsReplaySucceeds() {
        int embedDim = 128, numLayers = 24;  // ~75 external inputs, ~100+ slots
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.AUTO);

        INDArray lastOut = null;
        for (int i = 0; i < 15; i++) {
            Map<String, INDArray> ph = buildFreshPlaceholders(embedDim, numLayers);
            Map<String, INDArray> out = g.output(ph, "out");
            lastOut = out.get("out").dup();
            assertNotNull(lastOut, "step " + i + ": output is null");
            assertFalse(lastOut.isNaN().any(), "step " + i + ": output contains NaN");
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int planPhase = DspPlanAssertions.getPlanPhase(g);
        log.info("LARGE: totalReplays={} planPhase={} numLayers={} embedDim={}",
                totalReplays, planPhase, numLayers, embedDim);

        assertTrue(totalReplays > 0,
                "LARGE: Expected graph replays > 0 but got " + totalReplays
                        + " planPhase=" + planPhase);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 5: Mixed stable/fresh — some placeholders reused, some recreated
    //
    // This matches the real VLM pattern where pixel_values changes each
    // page but model weights are the same Java objects (though the DSP
    // may still see them as "variable" due to ONNX import metadata).
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(5)
    void test5_mixedStableFreshReplaySucceeds() {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.AUTO);

        // Create weight placeholders ONCE (stable addresses)
        Map<String, INDArray> stablePh = buildStablePlaceholders(embedDim, numLayers);

        INDArray lastOut = null;
        for (int i = 0; i < 15; i++) {
            // Replace only the input — weights stay at same device addresses
            stablePh.put("pixel_values", Nd4j.randn(DataType.FLOAT, 1, embedDim).muli(0.1f));

            Map<String, INDArray> out = g.output(stablePh, "out");
            lastOut = out.get("out").dup();
            assertNotNull(lastOut, "step " + i + ": output is null");
            assertFalse(lastOut.isNaN().any(), "step " + i + ": output contains NaN");
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int planPhase = DspPlanAssertions.getPlanPhase(g);
        log.info("MIXED: totalReplays={} planPhase={}", totalReplays, planPhase);

        assertTrue(totalReplays > 0,
                "MIXED: Expected graph replays > 0 but got " + totalReplays
                        + " planPhase=" + planPhase);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 6: Vision encoder graph — multi-head attention with many matmuls
    //
    // Most realistic VLM reproduction: multi-head attention with QKV
    // projections, reshape/permute for head splitting, attention score
    // matmul, softmax, output projection, FFN. This produces the same
    // op mix as the SmolDocling vision encoder seg[395-785]:
    //   - 8+ cuBLAS matmuls per layer (gap ops)
    //   - reshape/permute creating intermediate view ops
    //   - softmax producing temporary buffers
    //   - ~11 placeholders per layer (many external inputs)
    //   - Fresh allocations each step to trigger address drift
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(6)
    void test6_visionEncoderReplaySucceeds() {
        int embedDim = 64, numHeads = 4, numLayers = 6;
        // 6 layers * 11 placeholders + 2 = 68 external inputs, ~200+ slots
        SameDiff g = buildVlmVisionEncoderGraph(embedDim, numHeads, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.AUTO);

        INDArray lastOut = null;
        for (int i = 0; i < 15; i++) {
            Map<String, INDArray> ph = buildVisionPlaceholders(embedDim, numHeads, numLayers);
            Map<String, INDArray> out = g.output(ph, "out");
            lastOut = out.get("out").dup();
            assertNotNull(lastOut, "step " + i + ": output is null");
            assertFalse(lastOut.isNaN().any(), "step " + i + ": output contains NaN");
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int planPhase = DspPlanAssertions.getPlanPhase(g);
        log.info("VISION_ENC: totalReplays={} planPhase={} numLayers={} numHeads={} embedDim={}",
                totalReplays, planPhase, numLayers, numHeads, embedDim);

        assertTrue(totalReplays > 0,
                "VISION_ENC: Expected graph replays > 0 but got " + totalReplays
                        + " planPhase=" + planPhase);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 7: Vision encoder at VLM scale — 12 layers, 128-dim
    //
    // Full scale: matches the actual SmolDocling vision encoder dimensions.
    // ~132 external inputs, ~400+ slots, should produce 500+ CUDA graph nodes.
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(7)
    void test7_visionEncoderFullScaleReplaySucceeds() {
        int embedDim = 128, numHeads = 8, numLayers = 12;
        SameDiff g = buildVlmVisionEncoderGraph(embedDim, numHeads, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.AUTO);

        INDArray lastOut = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> ph = buildVisionPlaceholders(embedDim, numHeads, numLayers);
            Map<String, INDArray> out = g.output(ph, "out");
            lastOut = out.get("out").dup();
            assertNotNull(lastOut, "step " + i + ": output is null");
            assertFalse(lastOut.isNaN().any(), "step " + i + ": output contains NaN");
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int planPhase = DspPlanAssertions.getPlanPhase(g);
        log.info("VISION_FULL: totalReplays={} planPhase={} numLayers={} numHeads={} embedDim={}",
                totalReplays, planPhase, numLayers, numHeads, embedDim);

        assertTrue(totalReplays > 0,
                "VISION_FULL: Expected graph replays > 0 but got " + totalReplays
                        + " planPhase=" + planPhase);
    }
}
