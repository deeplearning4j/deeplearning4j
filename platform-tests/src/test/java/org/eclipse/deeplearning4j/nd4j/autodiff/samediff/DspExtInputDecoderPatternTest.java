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
import org.junit.jupiter.params.ParameterizedTest;
import org.nd4j.common.tests.tags.TagNames;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP decoder-pattern staleness tests extracted from DspExtInputStalenessTest.
 *
 * Covers:
 * - Category 8: VLM Decode Pattern Reproduction
 * - Category 18: Decoder-Graph Tipping Point Isolation
 * - Category 19: Decoder Bug Bisection — Progressive Build-Up
 * - Category 20: Confirming the Trigger — Constant-Derived Add to Placeholder-Derived
 * - Position Encoding Isolation (VLM decode degenerate root cause)
 * - Minimal Staleness Isolation (single placeholder matmul)
 * - SV12D: No-reference island+gap staleness test
 * - SV12E: Stream sync test — compositeReplay on DSP stream, copyBuffer on LC stream
 * - SV12F: Island-only (no gap) test
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class DspExtInputDecoderPatternTest {

    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GRAPH FIXTURES
    // ═══════════════════════════════════════════════════════════════════════════

    /** Single placeholder x → matmul(w) + b → out. Weights always positive. */
    private SameDiff buildSinglePlaceholder(int inDim, int outDim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, inDim, outDim)).addi(0.1f));
        SDVariable b = g.var("b", Nd4j.ones(DataType.FLOAT, 1, outDim));
        SDVariable mm = g.mmul("mm", x, w);
        mm.add("out", b);
        return g;
    }

    /** Build single placeholder graph with pre-generated weights (for reference comparison). */
    private SameDiff buildSinglePlaceholder(int inDim, int outDim, INDArray wArr, INDArray bArr) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        SDVariable w = g.var("w", wArr.dup());
        SDVariable b = g.var("b", bArr.dup());
        SDVariable mm = g.mmul("mm", x, w);
        mm.add("out", b);
        return g;
    }

    /** Multi-placeholder: matmul(x, w_ph) + b_ph → out */
    private SameDiff buildMultiPlaceholder(int inDim, int outDim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        SDVariable w = g.placeHolder("w", DataType.FLOAT, inDim, outDim);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 1, outDim);
        SDVariable mm = g.mmul("mm", x, w);
        g.math().add("out", mm, b);
        return g;
    }

    /** Large decoder-like graph: embed + multiple attention-like layers.
     *  Creates enough ops for multiple segments with gaps. */
    private SameDiff buildLargeDecoderGraph(int embedDim, int numLayers) {
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, embedDim);
        SDVariable posIds = g.placeHolder("position_ids", DataType.FLOAT, 1, 1);

        // Position encoding (add scalar)
        SDVariable x = embed.add("pos_add", posIds);

        for (int layer = 0; layer < numLayers; layer++) {
            String prefix = "layer_" + layer + "_";
            // KV cache placeholders (simulate 30 layers × 2 = 60 KV inputs)
            SDVariable kv = g.placeHolder(prefix + "kv", DataType.FLOAT, 1, 4, embedDim);

            // Attention: Q=x*Wq, K=kv, V=kv, out = softmax(Q*K^T)*V (simplified as matmul chain)
            SDVariable wq = g.var(prefix + "wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f));
            SDVariable wv = g.var(prefix + "wv", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f));

            // Reshape x from [1,1,embed] to [1,embed] for matmul
            SDVariable xFlat = g.reshape(prefix + "xflat", x, 1, embedDim);
            SDVariable q = g.mmul(prefix + "q", xFlat, wq);

            // KV: take mean along seq dim → [1, embedDim]
            SDVariable kvMean = g.mean(prefix + "kv_mean", kv, 1);

            // Attention score (simplified): q * kvMean^T → [1, 1]
            SDVariable kvMeanT = g.permute(prefix + "kvt", kvMean, 1, 0);
            SDVariable score = g.mmul(prefix + "score", q, kvMeanT);

            // Output projection
            SDVariable attnOut = g.mmul(prefix + "attn_out", score, g.reshape(prefix + "kvr", kvMean, 1, embedDim));

            // Residual + layer norm (simplified as add + tanh for nonlinearity)
            SDVariable residual = xFlat.add(prefix + "residual", attnOut);
            x = g.reshape(prefix + "out", residual, 1, 1, embedDim);
        }

        // Final projection to logits
        SDVariable wFinal = g.var("w_final", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, 32)).addi(0.01f));
        SDVariable xFinal = g.reshape("x_final_flat", x, 1, embedDim);
        g.mmul("out", xFinal, wFinal);
        return g;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SHARED HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    private void configureMode(SameDiff sd, GraphExecutionMode mode) {
        sd.getSessions().clear();
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    private Map<String, INDArray> singlePh(String name, INDArray arr) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put(name, arr);
        return ph;
    }

    /** Run N warmup steps to get plan to REPLAYING state */
    private void warmup(SameDiff sd, Map<String, INDArray> ph, String outName, int steps) {
        for (int i = 0; i < steps; i++) {
            sd.output(ph, outName);
        }
    }

    /** Run N warmup steps mutating placeholder value each step */
    private void warmupWithChangingInput(SameDiff sd, String phName, INDArray arr,
                                          String outName, int steps, long[] shape) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put(phName, arr);
        for (int i = 0; i < steps; i++) {
            arr.assign(Nd4j.valueArrayOf(shape, (double)(i + 1)));
            sd.output(ph, outName);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 8: VLM Decode Pattern Reproduction
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "decodePatternPrefillThenSingleToken mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("VLM pattern: prefill step, then single-token decode with same buffer")
    void testDecodePatternPrefillThenSingleToken(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        // "Prefill": large input (simulates long sequence)
        INDArray embed = Nd4j.randn(DataType.FLOAT, 1, 16);
        warmupWithChangingInput(sd, "x", embed, "out", 8, new long[]{1, 16});

        // "Decode": same buffer, but assign different "token embeddings"
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 30; step++) {
            // Simulate embedding lookup: different token → different values
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 16}, (double)(step + 1) * 0.1));
            Map<String, INDArray> result = sd.output(singlePh("x", embed), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) {
                stuckCount++;
            }
        }
        assertTrue(stuckCount < 3,
                mode + ": DEGENERATE — " + stuckCount + "/29 steps stuck! sums=" + sums.subList(0, Math.min(10, sums.size())));
        log.info("[DECODE_PATTERN] mode={} PASS — {}/29 steps unique", mode, 29 - stuckCount);
    }

    @ParameterizedTest(name = "decodePattern50Steps mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("50-step decode: embed lookup + position increment")
    void testDecodePattern50Steps(GraphExecutionMode mode) {
        // Graph with embed + position
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 16);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));
        SDVariable mm = g.mmul("mm", x, w);
        g.math().add("out", mm, pos);
        sd = g;
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 16);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", embed);
        ph.put("pos", posArr);

        // Warmup
        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 16}, (double)(i + 1)));
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        // 50 decode steps
        Set<Long> uniqueOutputs = new HashSet<>();
        for (int step = 0; step < 50; step++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 16}, (double)(step + 10) * 0.1));
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 8)));
            Map<String, INDArray> result = sd.output(ph, "out");
            long hash = Double.doubleToLongBits(result.get("out").sumNumber().doubleValue());
            uniqueOutputs.add(hash);
        }

        double uniqueRate = (double) uniqueOutputs.size() / 50.0;
        assertTrue(uniqueRate >= 0.8,
                mode + ": only " + uniqueOutputs.size() + "/50 unique outputs (" +
                        String.format("%.1f%%", uniqueRate * 100) + ") — DEGENERATE");
        log.info("[DECODE_50] mode={} PASS — {}/50 unique ({}%)", mode,
                uniqueOutputs.size(), String.format("%.1f", uniqueRate * 100));
    }

    @ParameterizedTest(name = "decodePatternTransition mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Warmup→Replay transition: step 4 output != step 3 output")
    void testDecodePatternTransitionFromWarmupToReplay(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        // Track outputs across the transition
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 12; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 1)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        // NO consecutive pair should be stuck
        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck (transition point?). sums=" + sums);
        }
        log.info("[TRANSITION] mode={} PASS — all 12 steps across warmup→replay unique", mode);
    }

    @ParameterizedTest(name = "fixedEmbedChangingPos mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Fixed embed + changing position → output still changes")
    void testDecodePatternFixedEmbedChangingPos(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable mm = g.mmul("mm", x, w);
        g.math().add("out", mm, pos);
        sd = g;
        configureMode(sd, mode);

        INDArray embed = Nd4j.valueArrayOf(new long[]{1, 8}, 5.0).castTo(DataType.FLOAT); // FIXED
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", embed);
        ph.put("pos", posArr);

        // Warmup
        for (int i = 0; i < 8; i++) {
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        // 10 steps: embed FIXED, pos changes
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck despite changing pos! sums=" + sums);
        }
        log.info("[FIXED_EMBED] mode={} PASS — changing pos reflected with fixed embed", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 18: Decoder-Graph Tipping Point Isolation
    // Finds exactly which combination of ops+placeholders triggers stuck output
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "singleLayerDecoder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Single-layer decoder: embed + pos + 1 KV + matmuls + mean + permute")
    void testSingleLayerDecoder(GraphExecutionMode mode) {
        // Exactly buildLargeDecoderGraph structure but numLayers=1
        sd = buildLargeDecoderGraph(16, 1);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        ph.put("layer_0_kv", kv0);

        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(step + 100)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 8)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": STUCK 1-layer decoder! " + stuckCount + "/19 steps. sums=" + sums.subList(0, Math.min(5, sums.size())));
        log.info("[1_LAYER_DECODER] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "decoderNoKV mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Decoder structure with embed+pos but NO KV placeholders")
    void testDecoderNoKV(GraphExecutionMode mode) {
        // Same structure but KV is a constant weight, not a placeholder
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, 16);
        SDVariable posIds = g.placeHolder("position_ids", DataType.FLOAT, 1, 1);
        SDVariable x = embed.add("pos_add", posIds);

        // One layer with CONSTANT kv (not placeholder)
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable wq = g.var("wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));
        SDVariable wv = g.var("wv", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, wq);
        SDVariable kvMean = g.mean("kv_mean", kv, 1);
        SDVariable kvMeanT = g.permute("kvt", kvMean, 1, 0);
        SDVariable score = g.mmul("score", q, kvMeanT);
        SDVariable attnOut = g.mmul("attn_out", score, g.reshape("kvr", kvMean, 1, 16));
        SDVariable residual = xFlat.add("residual", attnOut);

        SDVariable wFinal = g.var("w_final", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 32)).addi(0.01f));
        g.mmul("out", residual, wFinal);
        sd = g;
        configureMode(sd, mode);

        INDArray embedArr = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embedArr);
        ph.put("position_ids", posArr);

        for (int i = 0; i < 8; i++) {
            embedArr.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            embedArr.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(step + 100)));
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 8)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": STUCK decoder-no-KV! " + stuckCount + "/19 steps. sums=" + sums.subList(0, Math.min(5, sums.size())));
        log.info("[DECODER_NO_KV] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "decoderEmbedOnlyChanges mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Full decoder: only embed changes, pos+KV stable after warmup")
    void testDecoderEmbedOnlyChanges(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 2);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
        INDArray kv1 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        ph.put("layer_0_kv", kv0);
        ph.put("layer_1_kv", kv1);

        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        // Now only embed changes, everything else stable
        posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, 99.0));
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(step + 200)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": STUCK embed-only-changes! " + stuckCount + "/19 steps. sums=" + sums.subList(0, Math.min(5, sums.size())));
        log.info("[EMBED_ONLY] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "decoderPosOnlyChanges mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Full decoder: only position_ids changes, embed+KV stable after warmup")
    void testDecoderPosOnlyChanges(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 2);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
        INDArray kv1 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        ph.put("layer_0_kv", kv0);
        ph.put("layer_1_kv", kv1);

        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        // Now only pos changes, everything else stable
        embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, 50.0));
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 200)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": STUCK pos-only-changes! " + stuckCount + "/19 steps. sums=" + sums.subList(0, Math.min(5, sums.size())));
        log.info("[POS_ONLY] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "decoderKVOnlyChanges mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Full decoder: only KV changes, embed+pos stable after warmup")
    void testDecoderKVOnlyChanges(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 2);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
        INDArray kv1 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        ph.put("layer_0_kv", kv0);
        ph.put("layer_1_kv", kv1);

        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            kv0.assign(Nd4j.valueArrayOf(new long[]{1, 4, 16}, (double)(i + 1) * 0.1));
            sd.output(ph, "out");
        }

        // Now only KV changes, embed+pos stable
        embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, 50.0));
        posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, 99.0));
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            kv0.assign(Nd4j.valueArrayOf(new long[]{1, 4, 16}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": STUCK KV-only-changes! " + stuckCount + "/19 steps. sums=" + sums.subList(0, Math.min(5, sums.size())));
        log.info("[KV_ONLY] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "decoderAllChangeWithStableKV mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Full decoder: embed+pos change, KV stable (closest VLM pattern)")
    void testDecoderAllChangeWithStableKV(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 2);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
        INDArray kv1 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        ph.put("layer_0_kv", kv0);
        ph.put("layer_1_kv", kv1);

        // Warmup — ALL inputs change during warmup
        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            kv0.assign(Nd4j.valueArrayOf(new long[]{1, 4, 16}, (double)(i + 1) * 0.1));
            kv1.assign(Nd4j.valueArrayOf(new long[]{1, 4, 16}, (double)(i + 1) * 0.2));
            sd.output(ph, "out");
        }

        // Post-warmup: only embed+pos change (VLM pattern: KV grows but address stable)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 30; step++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(step + 100)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 8)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": STUCK embed+pos changing, KV stable! " + stuckCount + "/29 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[EMBED_POS_CHANGE_KV_STABLE] mode={} PASS — {}/29 unique", mode, 29 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 19: Decoder Bug Bisection — Progressive Build-Up
    // Each test adds ONE element to the simplest passing case.
    // Goal: find the exact op/combination that triggers stuck output in TRITON/AUTO.
    // ═══════════════════════════════════════════════════════════════════════════

    /** Helper: run standard staleness check for 20 steps with single placeholder */
    private void assertNotStuck(SameDiff g, GraphExecutionMode mode, String phName,
                                long[] phShape, String outName, String tag) {
        configureMode(g, mode);
        INDArray input = Nd4j.ones(DataType.FLOAT, phShape);
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(phShape, (double)(i + 1)));
            g.output(singlePh(phName, input), outName);
        }
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(phShape, (double)(step + 100)));
            Map<String, INDArray> result = g.output(singlePh(phName, input), outName);
            sums.add(result.get(outName).sumNumber().doubleValue());
        }
        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [" + tag + "]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[{}] mode={} PASS — {}/19 unique", tag, mode, 19 - stuckCount);
    }

    /** Helper: run staleness check with multi-placeholder map */
    private void assertNotStuckMultiPh(SameDiff g, GraphExecutionMode mode,
                                       Map<String, INDArray> ph, String changingPh,
                                       long[] changingShape, String outName, String tag) {
        configureMode(g, mode);
        INDArray changingArr = ph.get(changingPh);
        for (int i = 0; i < 8; i++) {
            changingArr.assign(Nd4j.valueArrayOf(changingShape, (double)(i + 1)));
            g.output(ph, outName);
        }
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            changingArr.assign(Nd4j.valueArrayOf(changingShape, (double)(step + 100)));
            Map<String, INDArray> result = g.output(ph, outName);
            sums.add(result.get(outName).sumNumber().doubleValue());
        }
        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [" + tag + "]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[{}] mode={} PASS — {}/19 unique", tag, mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "bisect_3Dinput_reshape_matmul mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 1: 3D input [1,1,16] → reshape [1,16] → matmul → out")
    void testBisect_3DInput_Reshape_Matmul(GraphExecutionMode mode) {
        // This is the first element from the decoder: 3D placeholder + reshape to 2D
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));
        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        g.mmul("out", xFlat, w);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "BISECT_3D_RESHAPE_MM");
    }

    @ParameterizedTest(name = "bisect_3Dinput_reshape_matmul_add mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 2: 3D input → reshape → matmul → add(pos_placeholder)")
    void testBisect_3DInput_Reshape_Matmul_Add(GraphExecutionMode mode) {
        // Add second placeholder (position_ids) via add
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));
        SDVariable xPos = x.add("xpos", pos); // [1,1,16] + [1,1] broadcast
        SDVariable xFlat = g.reshape("xflat", xPos, 1, 16);
        g.mmul("out", xFlat, w);
        sd = g;
        configureMode(g, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("pos", posArr);
        assertNotStuckMultiPh(g, mode, ph, "x", new long[]{1, 1, 16}, "out", "BISECT_3D_ADD_RESHAPE_MM");
    }

    @ParameterizedTest(name = "bisect_3Dinput_reshape_2matmuls mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 3: 3D input → reshape → matmul1 → matmul2 → out (2 matmuls chained)")
    void testBisect_3DInput_Reshape_2Matmuls(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));
        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable mm1 = g.mmul("mm1", xFlat, w1);
        g.mmul("out", mm1, w2);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "BISECT_3D_2MM");
    }

    @ParameterizedTest(name = "bisect_meanOnConstant mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 4: 3D input → reshape → matmul + mean(constant) → matmul")
    void testBisect_MeanOnConstant(GraphExecutionMode mode) {
        // Adds mean on a CONSTANT (not placeholder) — like decoder's kvMean
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16] (mean along seq dim)
        // score = q * kvMean element-wise then sum to scalar-ish
        SDVariable combined = q.add("combined", kvMean); // [1, 16]
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));
        g.mmul("out", combined, wOut);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "BISECT_MEAN_CONST");
    }

    @ParameterizedTest(name = "bisect_meanOnConst_permuteOnConst mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 5: 3D input → reshape → matmul + mean(const) + permute(const) → matmul → out")
    void testBisect_MeanOnConst_PermuteOnConst(GraphExecutionMode mode) {
        // Adds permute on constant-derived value — the exact decoder pattern
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable kvMeanT = g.permute("kvt", kvMean, 1, 0); // [16, 1]
        SDVariable score = g.mmul("score", q, kvMeanT); // [1, 1]
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 1, 8)).addi(0.1f));
        g.mmul("out", score, wOut);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "BISECT_MEAN_PERMUTE_CONST");
    }

    @ParameterizedTest(name = "bisect_fullDecoderMinimal mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 6: Full decoder-no-KV minus residual — just Q*K^T*V chain")
    void testBisect_FullDecoderMinimal(GraphExecutionMode mode) {
        // Q*K^T*V chain like decoder but NO residual add — isolates whether residual triggers it
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable wq = g.var("wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, wq); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable kvMeanT = g.permute("kvt", kvMean, 1, 0); // [16, 1]
        SDVariable score = g.mmul("score", q, kvMeanT); // [1, 1]
        // attn_out = score * kvMean (reshaped)
        SDVariable kvr = g.reshape("kvr", kvMean, 1, 16);
        SDVariable attnOut = g.mmul("attn_out", score, kvr); // [1, 16]
        // Final projection directly (NO residual)
        SDVariable wFinal = g.var("w_final", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.01f));
        g.mmul("out", attnOut, wFinal);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "BISECT_QKV_NO_RESIDUAL");
    }

    @ParameterizedTest(name = "bisect_fullDecoderWithResidual mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 7: Q*K^T*V chain + residual add — the full decoder-no-KV without final reshape back to 3D")
    void testBisect_FullDecoderWithResidual(GraphExecutionMode mode) {
        // Same as decoder-no-KV: adds the residual connection. This should match testDecoderNoKV behavior.
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable wq = g.var("wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, wq); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable kvMeanT = g.permute("kvt", kvMean, 1, 0); // [16, 1]
        SDVariable score = g.mmul("score", q, kvMeanT); // [1, 1]
        SDVariable kvr = g.reshape("kvr", kvMean, 1, 16);
        SDVariable attnOut = g.mmul("attn_out", score, kvr); // [1, 16]
        // Residual: xFlat + attnOut
        SDVariable residual = xFlat.add("residual", attnOut); // [1, 16]
        SDVariable wFinal = g.var("w_final", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.01f));
        g.mmul("out", residual, wFinal);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "BISECT_QKV_WITH_RESIDUAL");
    }

    @ParameterizedTest(name = "bisect_2phDecoder_noMeanPermute mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 8: 2 placeholders + reshape + 2 matmuls + add (no mean, no permute)")
    void testBisect_2PhDecoder_NoMeanPermute(GraphExecutionMode mode) {
        // 2 placeholders like decoder, but replace mean+permute with direct matmul
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.01f));

        SDVariable xPos = embed.add("xpos", pos);
        SDVariable xFlat = g.reshape("xflat", xPos, 1, 16);
        SDVariable mm1 = g.mmul("mm1", xFlat, w1);
        // No mean, no permute — straight second matmul
        g.mmul("out", mm1, w2);
        sd = g;
        configureMode(g, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("pos", posArr);
        assertNotStuckMultiPh(g, mode, ph, "x", new long[]{1, 1, 16}, "out", "BISECT_2PH_NO_MEAN_PERMUTE");
    }

    @ParameterizedTest(name = "bisect_2phDecoder_withMean mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 9: 2 placeholders + reshape + matmul + mean(const) + add (no permute)")
    void testBisect_2PhDecoder_WithMean(GraphExecutionMode mode) {
        // Adds mean on constant to the 2-placeholder graph
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));

        SDVariable xPos = embed.add("xpos", pos);
        SDVariable xFlat = g.reshape("xflat", xPos, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        // Combine q and kvMean via add (no permute, no score matmul)
        SDVariable combined = q.add("combined", kvMean);
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.01f));
        g.mmul("out", combined, wOut);
        sd = g;
        configureMode(g, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("pos", posArr);
        assertNotStuckMultiPh(g, mode, ph, "x", new long[]{1, 1, 16}, "out", "BISECT_2PH_WITH_MEAN");
    }

    @ParameterizedTest(name = "bisect_2phDecoder_meanAndPermute mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 10: 2 placeholders + reshape + matmul + mean(const) + permute + score matmul")
    void testBisect_2PhDecoder_MeanAndPermute(GraphExecutionMode mode) {
        // Full mean+permute path with 2 placeholders — should match decoder-no-KV
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable wq = g.var("wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));

        SDVariable xPos = embed.add("xpos", pos);
        SDVariable xFlat = g.reshape("xflat", xPos, 1, 16);
        SDVariable q = g.mmul("q", xFlat, wq); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable kvMeanT = g.permute("kvt", kvMean, 1, 0); // [16, 1]
        SDVariable score = g.mmul("score", q, kvMeanT); // [1, 1]
        // Final output from score
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 1, 8)).addi(0.01f));
        g.mmul("out", score, wOut);
        sd = g;
        configureMode(g, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("pos", posArr);
        assertNotStuckMultiPh(g, mode, ph, "x", new long[]{1, 1, 16}, "out", "BISECT_2PH_MEAN_PERMUTE");
    }

    @ParameterizedTest(name = "bisect_2phDecoder_meanPermuteResidual mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Bisect step 11: 2 placeholders + reshape + Q*K^T*V + residual + final matmul (FULL decoder-no-KV)")
    void testBisect_2PhDecoder_MeanPermuteResidual(GraphExecutionMode mode) {
        // This is exactly the decoder-no-KV with pos placeholder — should fail in TRITON/AUTO
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable wq = g.var("wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));

        SDVariable xPos = embed.add("xpos", pos);
        SDVariable xFlat = g.reshape("xflat", xPos, 1, 16);
        SDVariable q = g.mmul("q", xFlat, wq); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable kvMeanT = g.permute("kvt", kvMean, 1, 0); // [16, 1]
        SDVariable score = g.mmul("score", q, kvMeanT); // [1, 1]
        SDVariable kvr = g.reshape("kvr", kvMean, 1, 16);
        SDVariable attnOut = g.mmul("attn_out", score, kvr); // [1, 16]
        // Residual add
        SDVariable residual = xFlat.add("residual", attnOut); // [1, 16]
        SDVariable wFinal = g.var("w_final", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.01f));
        g.mmul("out", residual, wFinal);
        sd = g;
        configureMode(g, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("pos", posArr);
        assertNotStuckMultiPh(g, mode, ph, "x", new long[]{1, 1, 16}, "out", "BISECT_2PH_FULL_DECODER");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 20: Confirming the Trigger — Constant-Derived Add to Placeholder-Derived
    // Bisection found: mean(constant) + add(placeholder-derived) = STUCK
    // These tests confirm exactly which combination is required.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "confirm_multiConsumerNoMean mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Multi-consumer xFlat (2 matmuls) without mean — tests if multi-consumer alone triggers it")
    void testConfirm_MultiConsumerNoMean(GraphExecutionMode mode) {
        // xFlat feeds TWO matmuls (multi-consumer) but NO mean/reduce op
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.01f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.01f));
        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable mm1 = g.mmul("mm1", xFlat, w1); // [1, 16] — first consumer
        SDVariable residual = xFlat.add("residual", mm1); // second consumer of xFlat
        g.mmul("out", residual, w2);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "CONFIRM_MULTI_CONSUMER_NO_MEAN");
    }

    @ParameterizedTest(name = "confirm_meanConstAddedToPlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("mean(constant) added to placeholder-derived value — THE trigger")
    void testConfirm_MeanConstAddedToPlaceholder(GraphExecutionMode mode) {
        // Minimal reproduction: reshape + matmul + mean(constant) + add → matmul
        // This should FAIL in TRITON/AUTO (same as BISECT_MEAN_CONST)
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable combined = q.add("combined", kvMean); // ADD constant-derived to placeholder-derived
        g.mmul("out", combined, wOut);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "CONFIRM_MEAN_CONST_ADD_PH");
    }

    @ParameterizedTest(name = "confirm_sumConstInsteadOfMean mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("sum(constant) instead of mean — tests if it's mean-specific or any reduction")
    void testConfirm_SumConstInsteadOfMean(GraphExecutionMode mode) {
        // Replace mean with sum — same reduction semantics, different op
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable kvSum = g.sum("kv_sum", kv, 1); // [1, 16]
        SDVariable combined = q.add("combined", kvSum);
        g.mmul("out", combined, wOut);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "CONFIRM_SUM_CONST_ADD_PH");
    }

    @ParameterizedTest(name = "confirm_constDirectAddNoReduction mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Direct constant (no reduction) added to placeholder-derived — tests if reduction is needed")
    void testConfirm_ConstDirectAddNoReduction(GraphExecutionMode mode) {
        // Constant [1, 16] added directly (no mean/sum reduction) to q
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable bias = g.var("bias_const", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable combined = q.add("combined", bias); // ADD raw constant (no reduction)
        g.mmul("out", combined, wOut);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "CONFIRM_CONST_DIRECT_ADD_NO_REDUCE");
    }

    @ParameterizedTest(name = "confirm_meanConstMatmulNotAdd mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("mean(constant) used in matmul (not add) with placeholder-derived — tests if add specifically triggers it")
    void testConfirm_MeanConstMatmulNotAdd(GraphExecutionMode mode) {
        // q [1, 16] matmul with kvMean^T [16, 1] — uses mean output in matmul not add
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 1, 8)).addi(0.1f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable kvMeanT = g.permute("kvt", kvMean, 1, 0); // [16, 1]
        SDVariable score = g.mmul("score", q, kvMeanT); // [1, 1] — mean used via matmul
        g.mmul("out", score, wOut);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "CONFIRM_MEAN_CONST_MATMUL_NOT_ADD");
    }

    @ParameterizedTest(name = "confirm_meanConstMulNotAdd mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("mean(constant) MULtiplied (not added) with placeholder-derived — tests if op type matters")
    void testConfirm_MeanConstMulNotAdd(GraphExecutionMode mode) {
        // Replace add with mul: q * kvMean element-wise
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 16));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));

        SDVariable xFlat = g.reshape("xflat", x, 1, 16);
        SDVariable q = g.mmul("q", xFlat, w); // [1, 16]
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 16]
        SDVariable combined = q.mul("combined", kvMean); // MUL not ADD
        g.mmul("out", combined, wOut);
        sd = g;
        assertNotStuck(g, mode, "x", new long[]{1, 1, 16}, "out", "CONFIRM_MEAN_CONST_MUL_NOT_ADD");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Category: Position Encoding Isolation (VLM decode degenerate root cause)
    // These tests isolate whether same-content embedding at different positions
    // correctly produces different output through DSP composite replay.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "ropePositionDifferentiates mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("fusedRoPE: same embedding + different position → different output (VLM decode pattern)")
    void testRopePositionDifferentiates(GraphExecutionMode mode) {
        // This is the EXACT pattern that fails in VLM decode:
        // Same token embedding at different decode positions should produce
        // different output because RoPE rotates based on position.
        // Graph structure: matmul (capturable) → fusedRoPE (gap) → matmul (capturable)
        // This ensures fusedRoPE is a gap op between capturable islands.
        int dim = 64; // must be even for RoPE
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, 1, dim);
        SDVariable posOffset = g.placeHolder("pos", DataType.FLOAT); // scalar
        SDVariable wPre = g.var("w_pre", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
        SDVariable wPost = g.var("w_post", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, 8)).addi(0.1f));

        // Pre-RoPE matmul (capturable island), then RoPE (gap), then post-RoPE matmul (capturable)
        SDVariable flat = g.reshape("flat_in", embed, 1, dim);
        SDVariable preRope = g.mmul("pre_rope", flat, wPre); // [1, dim] — capturable
        SDVariable preRope3d = g.reshape("pre3d", preRope, 1, 1, dim);
        SDVariable rotated = g.nn().fusedRoPE("rope", preRope3d, posOffset, 0, 10000.0, 1.0, dim);
        SDVariable flatOut = g.reshape("flat_out", rotated, 1, dim);
        g.mmul("out", flatOut, wPost); // capturable
        sd = g;

        configureMode(g, mode);

        // Use FIXED embedding content (simulates same token predicted repeatedly)
        INDArray fixedEmbed = Nd4j.randn(DataType.FLOAT, 1, 1, dim);
        INDArray posArr = Nd4j.scalar(DataType.FLOAT, 0.0f);

        // Warmup with varying positions to reach frozen/replay state
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("embed", fixedEmbed);
        ph.put("pos", posArr);
        for (int i = 0; i < 8; i++) {
            posArr.assign(i);
            g.output(ph, "out");
        }

        // Test: SAME embedding, DIFFERENT positions — output must differ
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            posArr.assign(step + 100); // position changes, embedding stays same
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [ROPE_POSITION]: STUCK! " + stuckCount + "/19 steps. " +
                        "Same embedding + different positions should produce different RoPE output. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[ROPE_POSITION] mode={} PASS — {}/19 unique (same embed, different pos)", mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "ropeFixedPositionFixedEmbed mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("fusedRoPE: same embedding + SAME position → SAME output (control)")
    void testRopeFixedPositionFixedEmbed(GraphExecutionMode mode) {
        // Control test: if BOTH embedding AND position are fixed, output SHOULD be identical.
        // This confirms the test infrastructure works and RoPE doesn't randomly vary.
        int dim = 64;
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, 1, dim);
        SDVariable posOffset = g.placeHolder("pos", DataType.FLOAT);
        SDVariable wPre = g.var("w_pre", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
        SDVariable wPost = g.var("w_post", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, 8)).addi(0.1f));

        SDVariable flat = g.reshape("flat_in", embed, 1, dim);
        SDVariable preRope = g.mmul("pre_rope", flat, wPre);
        SDVariable preRope3d = g.reshape("pre3d", preRope, 1, 1, dim);
        SDVariable rotated = g.nn().fusedRoPE("rope", preRope3d, posOffset, 0, 10000.0, 1.0, dim);
        SDVariable flatOut = g.reshape("flat_out", rotated, 1, dim);
        g.mmul("out", flatOut, wPost);
        sd = g;

        configureMode(g, mode);

        INDArray fixedEmbed = Nd4j.randn(DataType.FLOAT, 1, 1, dim);
        INDArray posArr = Nd4j.scalar(DataType.FLOAT, 42.0f); // fixed position

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("embed", fixedEmbed);
        ph.put("pos", posArr);
        for (int i = 0; i < 8; i++) {
            g.output(ph, "out");
        }

        // All steps should produce SAME output (nothing changes)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int diffCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) > 1e-3) diffCount++;
        }
        assertTrue(diffCount == 0,
                mode + " [ROPE_FIXED_CONTROL]: outputs should be IDENTICAL but " + diffCount +
                        "/9 differed. sums=" + sums.subList(0, Math.min(5, sums.size())));
        log.info("[ROPE_FIXED_CONTROL] mode={} PASS — all 10 steps identical (expected)", mode);
    }

    @ParameterizedTest(name = "twoPlaceholderOneFixed mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Two placeholders: one fixed (embed), one changing (position scalar) — output must change")
    void testTwoPlaceholderOneFixed(GraphExecutionMode mode) {
        // Simpler version without RoPE — just verifies that a FIXED placeholder
        // combined with a CHANGING scalar placeholder produces different outputs.
        // This isolates whether the frozen fast-path incorrectly freezes when
        // only ONE of multiple placeholders changes.
        int dim = 16;
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, dim);
        SDVariable scale = g.placeHolder("scale", DataType.FLOAT); // scalar
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, 8)).addi(0.1f));

        // embed * scale → matmul → out
        SDVariable scaled = embed.mul("scaled", scale);
        g.mmul("out", scaled, w);
        sd = g;

        configureMode(g, mode);

        INDArray fixedEmbed = Nd4j.ones(DataType.FLOAT, 1, dim);
        INDArray scaleArr = Nd4j.scalar(DataType.FLOAT, 1.0f);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("embed", fixedEmbed);
        ph.put("scale", scaleArr);
        for (int i = 0; i < 8; i++) {
            scaleArr.assign(i + 1);
            g.output(ph, "out");
        }

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            scaleArr.assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [TWO_PH_ONE_FIXED]: STUCK! " + stuckCount + "/19 steps. " +
                        "Fixed embed + changing scale should produce different outputs. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[TWO_PH_ONE_FIXED] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    @ParameterizedTest(name = "scalarPositionInGraph mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Scalar placeholder used as position index in graph — output must track position changes")
    void testScalarPositionInGraph(GraphExecutionMode mode) {
        // Tests the specific pattern where a scalar placeholder (like position_ids)
        // is used in computation. This is critical because scalars may be misclassified
        // as constants by the frozen fast-path or gap slot cache.
        int dim = 16;
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, dim);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT); // scalar position
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
        SDVariable wOut = g.var("w_out", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, 8)).addi(0.1f));

        // Simulate position-dependent transformation:
        // embed + pos*0.01 → matmul → out (pos shifts the embedding)
        SDVariable posScale = pos.mul("pos_scaled", 0.01);
        SDVariable shifted = embed.add("shifted", posScale); // broadcast scalar to [1, dim]
        SDVariable hidden = g.mmul("hidden", shifted, w);
        g.mmul("out", hidden, wOut);
        sd = g;

        configureMode(g, mode);

        INDArray fixedEmbed = Nd4j.ones(DataType.FLOAT, 1, dim);
        INDArray posArr = Nd4j.scalar(DataType.FLOAT, 0.0f);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("embed", fixedEmbed);
        ph.put("pos", posArr);
        for (int i = 0; i < 8; i++) {
            posArr.assign(i);
            g.output(ph, "out");
        }

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            posArr.assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [SCALAR_POS_IN_GRAPH]: STUCK! " + stuckCount + "/19 steps. " +
                        "Scalar position placeholder must differentiate output. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[SCALAR_POS_IN_GRAPH] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MINIMAL STALENESS ISOLATION: single placeholder matmul
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Minimal isolation test: single placeholder x → mmul(w) + b → out.
     * Uses buildSinglePlaceholder(16,16).
     * Passes new random input each step, checks output changes.
     * Compares against SLOT_BY_SLOT reference for numerical correctness.
     */
    @ParameterizedTest(name = "minimalSinglePlaceholder_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "TRITON", "AUTO", "CUDA_GRAPHS"})
    void testMinimalSinglePlaceholderStuck(GraphExecutionMode mode) {
        int dim = 16;
        // Pre-generate weights so both graphs get independent copies of identical data
        INDArray wArr = Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f);
        INDArray bArr = Nd4j.ones(DataType.FLOAT, 1, dim);

        sd = buildSinglePlaceholder(dim, dim, wArr, bArr);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Reference graph in SLOT_BY_SLOT — built from same weight copies BEFORE either runs
        SameDiff sdRef = buildSinglePlaceholder(dim, dim, wArr, bArr);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int totalSteps = 20;
        int stuckCount = 0;
        int mismatchCount = 0;
        INDArray prevResult = null;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Map.of("x", input);

            INDArray result = sd.output(ph, "out").get("out").dup();
            INDArray ref = sdRef.output(ph, "out").get("out").dup();

            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN in result");
            assertFalse(ref.isNaN().any(), mode + " step " + step + ": NaN in ref");

            // Check vs reference
            double diffVsRef = ref.sub(result).amaxNumber().doubleValue();
            if (diffVsRef > 0.01) mismatchCount++;

            // Check stuck
            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) {
                    stuckCount++;
                    log.warn("[MINIMAL] {} step {}: STUCK change={} result[0..3]=[{},{},{},{}]",
                            mode, step, change,
                            result.getFloat(0), result.getFloat(1),
                            result.getFloat(2), result.getFloat(3));
                }
            }
            prevResult = result;

            if (step < 5 || step == totalSteps - 1) {
                log.info("[MINIMAL] {} step {}: diffVsRef={} result[0]={} ref[0]={}",
                        mode, step, String.format("%.6f", diffVsRef),
                        String.format("%.6f", result.getFloat(0)),
                        String.format("%.6f", ref.getFloat(0)));
            }
        }

        sdRef.close();

        assertTrue(stuckCount <= 1,
                mode + " MINIMAL: output stuck for " + stuckCount + "/" + totalSteps
                        + " steps. Single-placeholder matmul produces frozen output.");
        assertTrue(mismatchCount <= 2,
                mode + " MINIMAL: mismatch vs SLOT_BY_SLOT ref for " + mismatchCount + "/" + totalSteps
                        + " steps (need <=2).");
        log.info("[MINIMAL] {} PASS — stuck={} mismatch={}", mode, stuckCount, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SV12D: No-reference island+gap staleness test — proves Java output freshness
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * SV12-D: Same graph as SV12-A but WITHOUT a SLOT_BY_SLOT reference.
     * Eliminates confounding diagnostic output from the reference graph.
     * Explicitly logs Java-side float values each step to prove whether
     * sd.output() returns fresh data or stale warmup data.
     *
     * If this test fails with STUCK, it proves the bug is in how Java
     * extracts output from the native plan (not in compositeReplay itself,
     * which POST_COMPOSITE_REPLAY_ARGMAX already showed produces fresh GPU data).
     */
    @ParameterizedTest(name = "noRefIslandGapStuck mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV12D_NoRefIslandGapStuck(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // Island: constant gather + stridedSlice (same as SV12A)
        INDArray embTable = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        sd.constant("emb_table", embTable.dup());
        INDArray indices = Nd4j.arange(dim).castTo(DataType.INT64);
        sd.constant("indices", indices.dup());

        SDVariable gathered = sd.gather("gather", sd.getVariable("emb_table"), sd.getVariable("indices"), 0);
        SDVariable reshaped = sd.reshape("reshape", gathered,
                sd.constant("rshape", Nd4j.createFromArray(1L, (long) dim * dim)));
        SDVariable sliced = sd.stridedSlice("slice", reshaped,
                new long[]{0, 0}, new long[]{1, dim}, new long[]{1, 1});

        // Gap: add placeholder + island output, then matmul
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w", w.dup());
        SDVariable added = x.add("add_input", sliced);
        sd.mmul("out", added, sd.getVariable("w"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int totalSteps = 20;
        int stuckCount = 0;
        INDArray prevResult = null;

        // Store all results for final analysis
        float[][] allFirstFour = new float[totalSteps][4];

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Map.of("x", input);

            INDArray result = sd.output(ph, "out").get("out").dup();

            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN");

            // Store first 4 values
            for (int j = 0; j < Math.min(4, (int) result.length()); j++) {
                allFirstFour[step][j] = result.getFloat(j);
            }

            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) {
                    stuckCount++;
                }
            }
            prevResult = result;

            // Log every step for Java-side visibility
            log.info("[SV12D] {} step {}: first4=[{},{},{},{}]",
                    mode, step,
                    String.format("%.6f", result.getFloat(0)),
                    String.format("%.6f", result.getFloat(1)),
                    String.format("%.6f", result.getFloat(2)),
                    String.format("%.6f", result.getFloat(3)));
        }

        // Final analysis: are all first4 arrays identical?
        int identicalPairs = 0;
        for (int s = 1; s < totalSteps; s++) {
            boolean same = true;
            for (int j = 0; j < 4; j++) {
                if (Math.abs(allFirstFour[s][j] - allFirstFour[s - 1][j]) > 1e-6) {
                    same = false;
                    break;
                }
            }
            if (same) identicalPairs++;
        }
        log.info("[SV12D] {} SUMMARY: stuckCount={} identicalPairs={}/{}", mode, stuckCount, identicalPairs, totalSteps - 1);

        assertTrue(stuckCount <= 1,
                mode + " SV12D: Java-visible output stuck for " + stuckCount + "/" + totalSteps
                        + " steps. CompositeReplay GPU data is fresh but Java extraction is stale.");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SV12E: Stream sync test — compositeReplay on DSP stream, copyBuffer on LC stream
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * SV12-E: Tests whether the output extraction D2D copy happens AFTER
     * compositeReplay finishes. If there's a missing stream sync between
     * compositeReplay (DSP stream) and the Java copyBuffer (LC default stream),
     * the copy might read pre-replay data.
     *
     * This test forces a large computation in the gap to make timing-dependent
     * races more likely to manifest.
     */
    @ParameterizedTest(name = "streamSyncGapOutput mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    void testSV12E_StreamSyncGapOutput(GraphExecutionMode mode) {
        // Use a larger dimension to make gap computation take longer
        // and increase chance of catching stream ordering bugs
        int dim = 64;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // Island: constant ops
        INDArray embTable = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        sd.constant("emb_table", embTable.dup());
        INDArray indices = Nd4j.arange(dim).castTo(DataType.INT64);
        sd.constant("indices", indices.dup());

        SDVariable gathered = sd.gather("gather", sd.getVariable("emb_table"), sd.getVariable("indices"), 0);
        SDVariable reshaped = sd.reshape("reshape", gathered,
                sd.constant("rshape", Nd4j.createFromArray(1L, (long) dim * dim)));
        SDVariable sliced = sd.stridedSlice("slice", reshaped,
                new long[]{0, 0}, new long[]{1, dim}, new long[]{1, 1});

        // Gap: add + TWO matmuls (chain) to increase computation time
        INDArray w1 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w1", w1.dup());
        sd.constant("w2", w2.dup());

        SDVariable added = x.add("add_input", sliced);
        SDVariable mm1 = sd.mmul("mm1", added, sd.getVariable("w1"));
        sd.mmul("out", mm1, sd.getVariable("w2"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int totalSteps = 20;
        int stuckCount = 0;
        INDArray prevResult = null;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Map.of("x", input);

            INDArray result = sd.output(ph, "out").get("out").dup();

            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN");

            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) stuckCount++;
            }
            prevResult = result;

            if (step < 5 || step == totalSteps - 1) {
                log.info("[SV12E] {} step {}: result[0]={}", mode, step,
                        String.format("%.6f", result.getFloat(0)));
            }
        }

        assertTrue(stuckCount <= 1,
                mode + " SV12E: stuck for " + stuckCount + "/" + totalSteps
                        + " steps. Stream sync missing between compositeReplay and Java output copy.");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SV12F: Island-only (no gap) test — isolates whether islands alone cause staleness
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * SV12-F: Graph where ALL ops can be Triton-compiled (no gap).
     * If this PASSES but SV12D FAILS, the bug is specifically in gap execution
     * during composite replay, not in the Triton island infrastructure itself.
     */
    @ParameterizedTest(name = "islandOnlyNoGap mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    void testSV12F_IslandOnlyNoGap(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // Only element-wise ops that Triton will compile — no matmul, no gather
        INDArray b1 = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
        INDArray b2 = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
        sd.constant("b1", b1.dup());
        sd.constant("b2", b2.dup());

        SDVariable added1 = x.add("add1", sd.getVariable("b1"));
        SDVariable added2 = added1.add("out", sd.getVariable("b2"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int totalSteps = 20;
        int stuckCount = 0;
        INDArray prevResult = null;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Map.of("x", input);

            INDArray result = sd.output(ph, "out").get("out").dup();

            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN");

            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) stuckCount++;
            }
            prevResult = result;

            if (step < 3 || step == totalSteps - 1) {
                log.info("[SV12F] {} step {}: result[0]={}", mode, step,
                        String.format("%.6f", result.getFloat(0)));
            }
        }

        assertTrue(stuckCount <= 1,
                mode + " SV12F: island-only stuck for " + stuckCount + "/" + totalSteps
                        + " steps. Triton island infrastructure broken even without gap ops.");
    }
}
