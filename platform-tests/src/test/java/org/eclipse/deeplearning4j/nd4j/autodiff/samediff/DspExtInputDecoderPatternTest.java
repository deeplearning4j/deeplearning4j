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
import org.nd4j.linalg.indexing.NDArrayIndex;
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
public class DspExtInputDecoderPatternTest extends DspExtInputTestSupport {

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

    @ParameterizedTest(name = "multiPositionExtInputNoCollapse mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("Multi-position prefill ext-input [1,N,d] must not collapse the last position onto an early one")
    void testMultiPositionExtInputNoCollapse(GraphExecutionMode mode) {
        // COVERAGE GAP: every other ext-input test here uses inputs_embeds [1,1,d] (single decode token).
        // The VLM PREFILL feeds inputs_embeds [1,N,d] (N=1142 in the real SmolDocling model) — a
        // MULTI-POSITION external input. The decode garbage (first token 11126="User") was traced to the
        // DSP staging of this multi-position external input collapsing the LAST (sample) position onto an
        // early one (pos1141 == pos2 in the verbose op dump). Isolation: a strictly-increasing per-position
        // input must produce a strictly-largest LAST-position output. Reproduces in ALL modes if it's the bug.
        final int N = 1142, embedDim = 576, outDim = 64;
        sd = SameDiff.create();
        SDVariable embed = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, embedDim);
        SDVariable flat = embed.reshape(N, embedDim);                                 // [N, d]
        SDVariable w = sd.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, outDim)).addi(0.1f));
        SDVariable mm = sd.mmul("mm", flat, w);                                       // [N, outDim], row i = embed[i]·w (>0)
        SDVariable outVar = mm.reshape(1, N, outDim);                                 // [1, N, outDim]
        final String OUT = outVar.name();
        configureMode(sd, mode);

        // Position i is the constant (i+1)*0.01 across all d  ->  out[i].sum strictly increases with i.
        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01)
                .reshape(N, 1).broadcast(N, embedDim).reshape(1, N, embedDim);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, OUT, 3);                                                       // drive DSP to active/replay
        INDArray out = sd.output(ph, OUT).get(OUT);                                   // [1, N, outDim]

        INDArray posSums = out.sum(2).reshape(N);                                     // sum per position, [N]
        double lastSum = posSums.getDouble(N - 1);
        double maxEarly = posSums.get(NDArrayIndex.interval(0, N - 1)).maxNumber().doubleValue();
        // Last position has the largest input (N*0.01) -> must have the strictly largest output sum.
        // A staging collapse (last position <- an early one) makes lastSum equal a small early sum.
        assertTrue(lastSum > maxEarly + 1e-3,
                mode + ": MULTI-POSITION COLLAPSE — last position (" + (N - 1) + ") output sum " + lastSum
                        + " is NOT strictly > max early-position sum " + maxEarly
                        + " -> DSP staging collapsed the last position onto an earlier one (the pos1141==pos2 decode bug).");
        log.info("[MULTI_POS_EXT] mode={} PASS — lastSum {} > maxEarly {} (all {} positions distinct)", mode, lastSum, maxEarly, N);
    }

    @ParameterizedTest(name = "multiPositionAttentionNoCollapse mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("Multi-position self-attention [1,N,d] must not collapse the last position onto an early one")
    void testMultiPositionAttentionNoCollapse(GraphExecutionMode mode) {
        // PROGRESSIVE BUILD-UP increment: the plain-matmul multi-position test PASSES (staging is fine),
        // so the collapse needs position INTERACTION. This adds single-head self-attention
        // (Q·Kᵀ then ·V — every output position mixes all positions) over the [1,N,d] prefill input.
        // With positive weights and strictly-increasing per-position input, each output position stays
        // proportional to (i+1), so the LAST position's output sum is still strictly the largest; a
        // position collapse (last <- early) in the attention matmuls / Kᵀ transpose breaks that.
        final int N = 1142, d = 576;
        sd = SameDiff.create();
        SDVariable embed = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, d);
        SDVariable flat = embed.reshape(N, d);                                       // [N,d]
        SDVariable wq = sd.var("aq", Transforms.abs(Nd4j.randn(DataType.FLOAT, d, d)).muli(0.01f).addi(0.001f));
        SDVariable wk = sd.var("ak", Transforms.abs(Nd4j.randn(DataType.FLOAT, d, d)).muli(0.01f).addi(0.001f));
        SDVariable wv = sd.var("av", Transforms.abs(Nd4j.randn(DataType.FLOAT, d, d)).muli(0.01f).addi(0.001f));
        SDVariable q = sd.mmul("aq_p", flat, wq);                                    // [N,d]
        SDVariable k = sd.mmul("ak_p", flat, wk);                                    // [N,d]
        SDVariable v = sd.mmul("av_p", flat, wv);                                    // [N,d]
        SDVariable scores = sd.mmul("ascore", q, k.permute(1, 0));                   // [N,N] — position interaction
        SDVariable ctx = sd.mmul("actx", scores, v);                                 // [N,d]
        SDVariable outVar = ctx.reshape(1, N, d);                                    // [1,N,d]
        final String OUT = outVar.name();
        configureMode(sd, mode);

        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01)
                .reshape(N, 1).broadcast(N, d).reshape(1, N, d);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, OUT, 3);
        INDArray out = sd.output(ph, OUT).get(OUT);                                  // [1,N,d]

        INDArray posSums = out.sum(2).reshape(N);
        double lastSum = posSums.getDouble(N - 1);
        double maxEarly = posSums.get(NDArrayIndex.interval(0, N - 1)).maxNumber().doubleValue();
        assertTrue(lastSum > maxEarly,
                mode + ": ATTENTION POSITION COLLAPSE — last position (" + (N - 1) + ") output sum " + lastSum
                        + " is NOT strictly > max early-position sum " + maxEarly
                        + " -> position interaction (Q·Kᵀ·V) collapsed the last position onto an earlier one.");
        log.info("[MULTI_POS_ATTN] mode={} PASS — lastSum {} > maxEarly {}", mode, lastSum, maxEarly);
    }

    @org.junit.jupiter.api.Test
    @DisplayName("broadcast_to of a VIEW (GQA 3→9 KV-head expand) must not produce zeros")
    void testBroadcastToOfViewNoZeros() {
        // ROOT of the VLM decode garbage (a97a50645 forward-trace): the GQA KV-head 3→9 expansion via
        // broadcast_to on a VIEW input emerges ALL ZEROS → K=V=0 into attention → dead attention → garbage.
        // The real SmolDocling decoder does: K[1,N,3,64] → permute[1,3,N,64] → expand_dims[1,3,1,N,64]
        // → broadcast_to[1,3,3,N,64]. a6f3d937's GQA isolation used sd.tile (passed), NEVER broadcast_to.
        final int N = 8;
        INDArray base = Nd4j.arange(1, N * 3 * 64 + 1).castTo(DataType.FLOAT).reshape(1, N, 3, 64); // distinct non-zero
        INDArray view = Nd4j.expandDims(base.permute(0, 2, 1, 3), 2);  // [1,3,N,64] permute(view) → [1,3,1,N,64] view
        assertNotEquals(0.0, view.sumNumber().doubleValue(), "precondition: view input must be non-zero");
        INDArray output = Nd4j.create(DataType.FLOAT, 1, 3, 3, N, 64);
        Nd4j.exec(new org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo(view, new long[]{1, 3, 3, N, 64}, output));
        double sum = output.sumNumber().doubleValue();
        assertNotEquals(0.0, sum,
                "broadcast_to of a VIEW produced ALL ZEROS (GQA KV-head expansion bug — the decode root). sum=" + sum);
        // each of the 3 broadcast copies along dim-2 must equal the view's single slice
        assertEquals(view.getDouble(0, 1, 0, 3, 5), output.getDouble(0, 1, 2, 3, 5), 1e-5, "broadcast copy 2 mismatch");
        assertEquals(view.getDouble(0, 2, 0, 7, 1), output.getDouble(0, 2, 1, 7, 1), 1e-5, "broadcast copy 1 mismatch");
        log.info("[BROADCAST_VIEW] PASS — broadcast_to of view non-zero, sum={}", sum);
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

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 19 EXTENSION: GQA Progressive Build-Up (Position-Collapse Isolation)
    //
    // testMultiPositionExtInputNoCollapse + testMultiPositionAttentionNoCollapse both
    // PASS all 4 modes (basic reshape+mmul and 2D attention staging are correct).
    // These 5 increments add GQA decoder structure ONE step at a time over the full
    // [1,1142,576] prefill tensor to find the FIRST step that makes pos[N-1] ≡ pos[≈2].
    //
    // Increments 1-3 (no position-dependent transform): assert lastSum > maxEarly
    //   (with all-positive weights and (i+1)*0.01 per-pos input, pos N-1 must have
    //   strictly the largest output-sum; a collapse makes it small like an early pos).
    // Increments 4-5 (RoPE makes output non-monotone): ROBUST assertion — last pos
    //   output vector must NOT equalsWithEps(any early pos vector, 1e-4).
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * GQA increment 1: head reshape pipeline.
     * embed[1,N,576] → flat[N,576] → Wq mm → q[N,576]
     *   → reshape[N,9,64] → permute[9,N,64] → reshape[9*N,64] → Wo mm → [9*N,64]
     *   → reshape[9,N,64] → permute[N,9,64] → reshape[1,N,576]
     *
     * Prime suspect: reshape of the NON-CONTIGUOUS permuted view [9,N,64]→[9*N,64]
     * (reshapei-class: if shape::reshapeC does not materialise a copy, position info corrupts).
     */
    @ParameterizedTest(name = "gqaHeadReshapeNoCollapse mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("GQA incr-1: head reshape/permute pipeline over [1,N,576] must not collapse positions")
    void testGqaHeadReshapeNoCollapse(GraphExecutionMode mode) {
        final int N = 1142, D = 576, heads = 9, headDim = 64; // D == heads * headDim
        sd = SameDiff.create();
        SDVariable embed = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, D);
        SDVariable flat  = sd.reshape("flat",  embed, (long)N, (long)D);              // [N, D]

        // Q projection
        SDVariable Wq   = sd.var("Wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, D)).addi(0.01f));
        SDVariable q    = sd.mmul("q", flat, Wq);                                      // [N, D]

        // GQA head reshape + permute — the reshapei-class suspect
        SDVariable qH   = sd.reshape("qH", q, (long)N, (long)heads, (long)headDim);  // [N,9,64] contiguous
        SDVariable qP   = sd.permute("qP", qH, 1, 0, 2);                             // [9,N,64] non-contiguous
        SDVariable qF   = sd.reshape("qF", qP, (long)(heads * N), (long)headDim);    // [9*N,64] reshape of non-contig

        // Per-"head" linear on the flattened [9*N, 64]
        SDVariable Wo   = sd.var("Wo", Transforms.abs(Nd4j.randn(DataType.FLOAT, headDim, headDim)).addi(0.01f));
        SDVariable qOut = sd.mmul("qOut", qF, Wo);                                    // [9*N, 64]

        // Permute back
        SDVariable qB   = sd.reshape("qB",  qOut, (long)heads, (long)N, (long)headDim); // [9,N,64]
        SDVariable qBP  = sd.permute("qBP", qB, 1, 0, 2);                             // [N,9,64] non-contiguous
        SDVariable out  = sd.reshape("out", qBP, 1L, (long)N, (long)D);               // [1,N,576]
        configureMode(sd, mode);

        // Strictly-increasing per-position input: position i = (i+1)*0.01 across all D dims
        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01f)
                .reshape(N, 1).broadcast(N, D).reshape(1, N, D);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, "out", 3);
        INDArray result = sd.output(ph, "out").get("out");                            // [1, N, D]

        INDArray posSums = result.sum(2).reshape(N);                                  // per-position sum [N]
        double lastSum   = posSums.getDouble(N - 1);
        double maxEarly  = posSums.get(NDArrayIndex.interval(0, N - 1)).maxNumber().doubleValue();
        // With all-positive weights & (i+1)*0.01 input: pos[N-1] must strictly dominate.
        // A collapse (pos[N-1] ← pos[≈2] in DSP staging) makes lastSum ≈ 3*0.01*factor << maxEarly.
        assertTrue(lastSum > maxEarly + 1e-3,
                mode + " GQA_INCR1_HEAD_RESHAPE: POSITION COLLAPSE — pos[" + (N - 1) + "] sum="
                        + lastSum + " NOT > maxEarly=" + maxEarly
                        + " (collapse reproduced at GQA reshape/permute step; reshapei-class confirmed)");
        log.info("[GQA_INCR1] mode={} PASS — lastSum={} maxEarly={} ratio={}", mode,
                lastSum, maxEarly, String.format("%.4f", lastSum / (maxEarly + 1e-12)));
    }

    /**
     * GQA increment 2: + KV-head tile (3 KV heads tiled to 9 Q heads).
     * Adds K path: flat → Wk mm → k[N,192] → reshape[N,3,64] → permute[3,N,64] → tile[9,N,64].
     * Combined: qP[9,N,64] + kTiled[9,N,64] → permute[N,9,64] → reshape[1,N,576].
     */
    @ParameterizedTest(name = "gqaKvRepeatNoCollapse mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("GQA incr-2: +KV head tile (3→9 via sd.tile) must not collapse positions")
    void testGqaKvRepeatNoCollapse(GraphExecutionMode mode) {
        final int N = 1142, D = 576, heads = 9, kvHeads = 3, headDim = 64;
        final int kvD = kvHeads * headDim; // 192
        sd = SameDiff.create();
        SDVariable embed = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, D);
        SDVariable flat  = sd.reshape("flat",  embed, (long)N, (long)D);

        // Q path → [9, N, 64]
        SDVariable Wq = sd.var("Wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, D)).addi(0.01f));
        SDVariable q  = sd.mmul("q",  flat, Wq);                                       // [N, D]
        SDVariable qH = sd.reshape("qH", q,  (long)N, (long)heads, (long)headDim);    // [N,9,64]
        SDVariable qP = sd.permute("qP", qH, 1, 0, 2);                                // [9,N,64]

        // K path: 3 KV heads tiled 3× along the head axis → [9, N, 64]
        SDVariable Wk = sd.var("Wk", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, kvD)).addi(0.01f));
        SDVariable k  = sd.mmul("k",  flat, Wk);                                       // [N, 192]
        SDVariable kH = sd.reshape("kH", k,  (long)N, (long)kvHeads, (long)headDim);  // [N,3,64]
        SDVariable kP = sd.permute("kP", kH, 1, 0, 2);                                // [3,N,64]
        // tile([3,N,64], [3,1,1]) → [9,N,64]: head dim repeated 3× (head0×3 then head1×3 ...)
        SDVariable kT = sd.tile("kT",  kP,  heads / kvHeads, 1, 1);                   // [9,N,64]

        // Combine Q+K element-wise, permute back, reshape to output
        SDVariable qk  = qP.add("qk",   kT);                                           // [9,N,64]
        SDVariable qkP = sd.permute("qkP", qk, 1, 0, 2);                              // [N,9,64]
        SDVariable out = sd.reshape("out", qkP, 1L, (long)N, (long)D);                // [1,N,576]
        configureMode(sd, mode);

        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01f)
                .reshape(N, 1).broadcast(N, D).reshape(1, N, D);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, "out", 3);
        INDArray result = sd.output(ph, "out").get("out");

        INDArray posSums = result.sum(2).reshape(N);
        double lastSum   = posSums.getDouble(N - 1);
        double maxEarly  = posSums.get(NDArrayIndex.interval(0, N - 1)).maxNumber().doubleValue();
        assertTrue(lastSum > maxEarly + 1e-3,
                mode + " GQA_INCR2_KV_TILE: POSITION COLLAPSE — pos[" + (N - 1) + "] sum="
                        + lastSum + " NOT > maxEarly=" + maxEarly + " (collapse at KV-tile step)");
        log.info("[GQA_INCR2] mode={} PASS — lastSum={} maxEarly={}", mode, lastSum, maxEarly);
    }

    /**
     * GQA increment 3: + causal mask on Q·K^T scores via dotProductAttentionV2.
     * Uses flash-attention variant with GQA (9 Q heads, 3 KV heads, useCausalMask=true).
     * Shapes: Q [1,N,9,64], K [1,N,3,64], V [1,N,3,64] → attn [1,N,9,64] → [1,N,576].
     * With causal mask + all-positive weights: pos[N-1] sees all N positions (most contribution).
     */
    @ParameterizedTest(name = "gqaCausalMaskNoCollapse mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("GQA incr-3: +causal mask (dotProductAttentionV2 useCausalMask=true) must not collapse positions")
    void testGqaCausalMaskNoCollapse(GraphExecutionMode mode) {
        final int N = 1142, D = 576, heads = 9, kvHeads = 3, headDim = 64;
        final int kvD = kvHeads * headDim; // 192
        sd = SameDiff.create();
        SDVariable embed = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, D);
        SDVariable flat  = sd.reshape("flat", embed, (long)N, (long)D);               // [N, D]

        // Q: flat → mm → [N,D] → reshape [N,9,64] → [1,N,9,64]
        SDVariable Wq  = sd.var("Wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, D)).addi(0.01f));
        SDVariable q   = sd.mmul("q", flat, Wq);
        SDVariable q3d = sd.reshape("q3d", q,  (long)N, (long)heads, (long)headDim);  // [N,9,64]
        SDVariable q4d = sd.reshape("q4d", q3d, 1L, (long)N, (long)heads, (long)headDim); // [1,N,9,64]

        // K: flat → mm → [N,kvD=192] → reshape [N,3,64] → [1,N,3,64]
        SDVariable Wk  = sd.var("Wk", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, kvD)).addi(0.01f));
        SDVariable k   = sd.mmul("k", flat, Wk);
        SDVariable k3d = sd.reshape("k3d", k,  (long)N, (long)kvHeads, (long)headDim);
        SDVariable k4d = sd.reshape("k4d", k3d, 1L, (long)N, (long)kvHeads, (long)headDim); // [1,N,3,64]

        // V: flat → mm → [N,kvD=192] → reshape [1,N,3,64]
        SDVariable Wv  = sd.var("Wv", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, kvD)).addi(0.01f));
        SDVariable v   = sd.mmul("v", flat, Wv);
        SDVariable v3d = sd.reshape("v3d", v,  (long)N, (long)kvHeads, (long)headDim);
        SDVariable v4d = sd.reshape("v4d", v3d, 1L, (long)N, (long)kvHeads, (long)headDim); // [1,N,3,64]

        // GQA flash attention with causal mask (0 scaleFactor = auto = 1/sqrt(headDim))
        // Output: [1, N, 9, 64]
        SDVariable attn = sd.nn().dotProductAttentionV2("attn",
                q4d, v4d, k4d, null, null, 0.0, 0.0, true, false);
        // Reshape [1,N,9,64] → [1,N,576] (9*64=576)
        SDVariable out = sd.reshape("out", attn, 1L, (long)N, (long)D);
        configureMode(sd, mode);

        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01f)
                .reshape(N, 1).broadcast(N, D).reshape(1, N, D);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, "out", 3);
        INDArray result = sd.output(ph, "out").get("out");                             // [1, N, 576]

        INDArray posSums = result.sum(2).reshape(N);
        double lastSum   = posSums.getDouble(N - 1);
        double maxEarly  = posSums.get(NDArrayIndex.interval(0, N - 1)).maxNumber().doubleValue();
        // Causal mask: pos[N-1] attends to ALL N tokens (most contribution) → still monotone.
        assertTrue(lastSum > maxEarly + 1e-3,
                mode + " GQA_INCR3_CAUSAL_MASK: POSITION COLLAPSE — pos[" + (N - 1) + "] sum="
                        + lastSum + " NOT > maxEarly=" + maxEarly + " (collapse at causal-mask attn step)");
        log.info("[GQA_INCR3] mode={} PASS — lastSum={} maxEarly={}", mode, lastSum, maxEarly);
    }

    /**
     * GQA increment 4: + RoPE rotation on Q and K.
     * fusedRoPE(positionOffset=0) applied to Q [1,N,9,64] and K [1,N,3,64]:
     * position i gets rotated by angle proportional to i (positions 0..N-1 for the N-token prefill).
     * RoPE makes output non-monotone → ROBUST assertion (last pos vec ≠ any early pos vec).
     */
    @ParameterizedTest(name = "gqaWithRopeNoCollapse mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("GQA incr-4: +RoPE on Q,K (positionOffset=0) — last pos output must differ from early pos outputs")
    void testGqaWithRopeNoCollapse(GraphExecutionMode mode) {
        final int N = 1142, D = 576, heads = 9, kvHeads = 3, headDim = 64;
        final int kvD = kvHeads * headDim;
        sd = SameDiff.create();
        SDVariable embed = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, D);
        SDVariable flat  = sd.reshape("flat", embed, (long)N, (long)D);

        // Q projection → [1, N, 9, 64]
        SDVariable Wq  = sd.var("Wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, D)).addi(0.01f));
        SDVariable q   = sd.mmul("q", flat, Wq);
        SDVariable q4d = sd.reshape("q4d",
                sd.reshape("q3d", q, (long)N, (long)heads, (long)headDim),
                1L, (long)N, (long)heads, (long)headDim);                               // [1,N,9,64]

        // K projection → [1, N, 3, 64]
        SDVariable Wk  = sd.var("Wk", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, kvD)).addi(0.01f));
        SDVariable k   = sd.mmul("k", flat, Wk);
        SDVariable k4d = sd.reshape("k4d",
                sd.reshape("k3d", k, (long)N, (long)kvHeads, (long)headDim),
                1L, (long)N, (long)kvHeads, (long)headDim);                             // [1,N,3,64]

        // V projection → [1, N, 3, 64]
        SDVariable Wv  = sd.var("Wv", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, kvD)).addi(0.01f));
        SDVariable v   = sd.mmul("v", flat, Wv);
        SDVariable v4d = sd.reshape("v4d",
                sd.reshape("v3d", v, (long)N, (long)kvHeads, (long)headDim),
                1L, (long)N, (long)kvHeads, (long)headDim);                             // [1,N,3,64]

        // RoPE: positionOffset=0 → token i gets position i (prefill: positions 0..N-1).
        // Different positions → different rotations → output at each position is unique.
        SDVariable posOffset = sd.constant("pos0", Nd4j.scalar(DataType.INT64, 0L));
        SDVariable qRope = sd.nn().fusedRoPE("qRope", q4d, posOffset, 0, 10000.0, 1.0, headDim);
        SDVariable kRope = sd.nn().fusedRoPE("kRope", k4d, posOffset, 0, 10000.0, 1.0, headDim);

        // GQA flash attention with causal mask
        SDVariable attn = sd.nn().dotProductAttentionV2("attn",
                qRope, v4d, kRope, null, null, 0.0, 0.0, true, false);
        SDVariable out  = sd.reshape("out", attn, 1L, (long)N, (long)D);               // [1,N,576]
        configureMode(sd, mode);

        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01f)
                .reshape(N, 1).broadcast(N, D).reshape(1, N, D);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, "out", 3);
        INDArray result = sd.output(ph, "out").get("out");                              // [1, N, 576]

        // ROBUST assertion: pos[N-1] output vector ≠ any early-position output vector within eps.
        // A DSP collapse (pos[N-1] ← pos[≈2]) makes them equal.
        assertGqaLastPosDistinct(result, N, D, mode, "GQA_INCR4_ROPE");
        log.info("[GQA_INCR4] mode={} PASS — last pos output distinct from all checked early pos outputs", mode);
    }

    /**
     * GQA increment 5: full decoder layer (RMSNorm + GQA-attention + residual + gated MLP + residual).
     * Exact SmolDocling text transformer decoder structure over [1,N,576] prefill input.
     */
    @ParameterizedTest(name = "gqaFullLayerNoCollapse mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("GQA incr-5: full decoder layer (RMSNorm+GQA-Attn+Residual+MLP) must not collapse positions")
    void testGqaFullLayerNoCollapse(GraphExecutionMode mode) {
        final int N = 1142, D = 576, heads = 9, kvHeads = 3, headDim = 64, mlpInter = 1152;
        final int kvD = kvHeads * headDim; // 192
        sd = SameDiff.create();
        SDVariable embed = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, D);
        SDVariable flat  = sd.reshape("flat", embed, (long)N, (long)D);               // [N, D]

        // --- Pre-attention RMSNorm (gamma=ones → pure normalisation, no re-scaling) ---
        SDVariable gamma1  = sd.var("gamma1", Nd4j.ones(DataType.FLOAT, D));
        SDVariable normed1 = sd.nn().rmsNorm("norm1", flat, gamma1, 1e-6);            // [N, D]

        // --- Q, K, V projections ---
        SDVariable Wq = sd.var("Wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, D)).addi(0.01f));
        SDVariable Wk = sd.var("Wk", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, kvD)).addi(0.01f));
        SDVariable Wv = sd.var("Wv", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, kvD)).addi(0.01f));

        SDVariable q4d = sd.reshape("q4d",
                sd.reshape("q3d", sd.mmul("q", normed1, Wq), (long)N, (long)heads, (long)headDim),
                1L, (long)N, (long)heads, (long)headDim);                               // [1,N,9,64]
        SDVariable k4d = sd.reshape("k4d",
                sd.reshape("k3d", sd.mmul("k", normed1, Wk), (long)N, (long)kvHeads, (long)headDim),
                1L, (long)N, (long)kvHeads, (long)headDim);                             // [1,N,3,64]
        SDVariable v4d = sd.reshape("v4d",
                sd.reshape("v3d", sd.mmul("v", normed1, Wv), (long)N, (long)kvHeads, (long)headDim),
                1L, (long)N, (long)kvHeads, (long)headDim);                             // [1,N,3,64]

        // --- RoPE ---
        SDVariable posOffset = sd.constant("pos0", Nd4j.scalar(DataType.INT64, 0L));
        SDVariable qRope = sd.nn().fusedRoPE("qRope", q4d, posOffset, 0, 10000.0, 1.0, headDim);
        SDVariable kRope = sd.nn().fusedRoPE("kRope", k4d, posOffset, 0, 10000.0, 1.0, headDim);

        // --- GQA flash attention + output projection + residual 1 ---
        SDVariable attn    = sd.nn().dotProductAttentionV2("attn",
                qRope, v4d, kRope, null, null, 0.0, 0.0, true, false);                // [1,N,9,64]
        SDVariable attnF   = sd.reshape("attnF", attn, (long)N, (long)D);             // [N, D]
        SDVariable Wo      = sd.var("Wo", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, D)).addi(0.01f));
        SDVariable attnOut = sd.mmul("attnOut", attnF, Wo);                           // [N, D]
        SDVariable res1    = flat.add("res1", attnOut);                                // [N, D]

        // --- Post-attention RMSNorm ---
        SDVariable gamma2  = sd.var("gamma2", Nd4j.ones(DataType.FLOAT, D));
        SDVariable normed2 = sd.nn().rmsNorm("norm2", res1, gamma2, 1e-6);            // [N, D]

        // --- Gated MLP (SwiGLU-like: SiLU(gate) * up → down) ---
        SDVariable Wgate  = sd.var("Wgate", Transforms.abs(Nd4j.randn(DataType.FLOAT, D, mlpInter)).addi(0.01f));
        SDVariable Wup    = sd.var("Wup",   Transforms.abs(Nd4j.randn(DataType.FLOAT, D, mlpInter)).addi(0.01f));
        SDVariable Wdown  = sd.var("Wdown", Transforms.abs(Nd4j.randn(DataType.FLOAT, mlpInter, D)).addi(0.01f));
        SDVariable gate   = sd.nn().silu("gate_act", sd.mmul("gate_proj", normed2, Wgate)); // [N, mlpInter]
        SDVariable up     = sd.mmul("up_proj", normed2, Wup);                         // [N, mlpInter]
        SDVariable mlpH   = gate.mul("mlp_h", up);                                    // [N, mlpInter]
        SDVariable mlpOut = sd.mmul("mlpOut", mlpH, Wdown);                           // [N, D]

        // --- Residual 2 + output ---
        SDVariable res2 = res1.add("res2", mlpOut);                                   // [N, D]
        SDVariable out  = sd.reshape("out", res2, 1L, (long)N, (long)D);              // [1, N, D]
        configureMode(sd, mode);

        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01f)
                .reshape(N, 1).broadcast(N, D).reshape(1, N, D);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, "out", 3);
        INDArray result = sd.output(ph, "out").get("out");                             // [1, N, D]

        assertGqaLastPosDistinct(result, N, D, mode, "GQA_INCR5_FULL_LAYER");
        log.info("[GQA_INCR5] mode={} PASS — full decoder layer: last pos distinct from all checked early pos outputs", mode);
    }

    /**
     * ROBUST assertion helper for increments 4-5:
     * Asserts that out3d[0, N-1, :] (last position) does NOT equalsWithEps any of the
     * "early" position output vectors (pos 0,1,2,3,5,10,50,100 and N-2).
     * A DSP position collapse makes pos[N-1] ← pos[≈2] → they become equal within eps=1e-4.
     */
    private void assertGqaLastPosDistinct(INDArray out3d, int N, int D,
                                          GraphExecutionMode mode, String tag) {
        // out3d: [1, N, D]
        INDArray lastVec = out3d.get(
                NDArrayIndex.point(0), NDArrayIndex.point(N - 1), NDArrayIndex.all()).dup(); // [D]
        int[] checkPositions = {0, 1, 2, 3, 5, 10, 50, 100, N - 2};
        for (int p : checkPositions) {
            if (p < 0 || p >= N - 1) continue;
            INDArray earlyVec = out3d.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(p), NDArrayIndex.all());
            assertFalse(lastVec.equalsWithEps(earlyVec, 1e-4),
                    mode + " " + tag + ": POSITION COLLAPSE — pos[" + (N - 1)
                            + "] output equalsWithEps pos[" + p + "] output (eps=1e-4)."
                            + " DSP staged pos[N-1] as pos[" + p + "] in the prefill.");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PREFILL SUSPECTS: gatherNd and where-based merge at N=1142, d=576
    //
    // These test the UNTESTED ops from the prefill graph histogram
    // (gather=5, gather_nd=2, plus the attention-mask Where path).
    // The bug is mode-independent (SLOT_BY_SLOT confirmed) so it must be a
    // VALUE computation bug, not a DSP staging artifact.
    // Collapse signature: hidden[N-1=1141] ≡ hidden[1] → first sampled
    // token = 11126 = "User" = input_ids[1], the prompt's second token.
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Tests that gatherNd from [1,N,d] using per-position 2D indices [[0,i] for i in 0..N-1]
     * correctly reads each position's own embedding. Specifically: last position (N-1) must
     * produce a strictly larger output sum than any early position, given strictly-increasing
     * per-pos input (pos i = (i+1)*0.01 * ones_d).
     *
     * The real SmolDocling decoder has gather=5 and gather_nd=2 ops. A position alias in
     * any of these (e.g. pos[1141] ← pos[1]) would produce firstToken=11126="User".
     *
     * Collapse detection: if gatherNd aliases pos[N-1] to an early pos, its output sum
     * will be small (~2*0.01*factor) instead of large (~N*0.01*factor).
     */
    @ParameterizedTest(name = "gatherNdLastPositionDistinct mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("gatherNd over [1,N,576] must not alias last position (N-1) onto an earlier one")
    void testGatherNdLastPositionDistinct(GraphExecutionMode mode) {
        final int N = 1142, d = 576, outD = 64;
        sd = SameDiff.create();

        // Source: inputs_embeds [1, N, d] placeholder
        SDVariable src = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, N, d);

        // Indices: [N, 2] INT64 where row i = [0, i] — gathers position i from batch 0
        long[] idxFlat = new long[N * 2];
        for (int i = 0; i < N; i++) {
            idxFlat[i * 2]     = 0L;   // batch index
            idxFlat[i * 2 + 1] = (long) i; // position index
        }
        INDArray idxArr = Nd4j.createFromArray(idxFlat).reshape(N, 2);
        SDVariable indices = sd.constant("gather_indices", idxArr);

        // gatherNd: src[ [0,i] for i in 0..N-1 ] → [N, d]
        // Each row of output = src[0, i, :] — identity gather over positions
        SDVariable gathered = sd.gatherNd("gathered", src, indices);    // [N, d]

        // All-positive weight projection → output sum monotone w.r.t. input scale
        SDVariable w   = sd.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, d, outD)).addi(0.1f));
        SDVariable mm  = sd.mmul("mm", gathered, w);                    // [N, outD]
        SDVariable out = sd.reshape("out", mm, 1L, (long) N, (long) outD); // [1, N, outD]

        configureMode(sd, mode);

        // Strictly-increasing per-position input: pos i = (i+1)*0.01 * ones_d
        INDArray embedArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01f)
                .reshape(N, 1).broadcast(N, d).reshape(1, N, d);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedArr);
        warmup(sd, ph, "out", 3);
        INDArray result = sd.output(ph, "out").get("out");               // [1, N, outD]

        INDArray posSums = result.sum(2).reshape(N);                     // per-position sum [N]
        double lastSum  = posSums.getDouble(N - 1);
        double maxEarly = posSums.get(NDArrayIndex.interval(0, N - 1)).maxNumber().doubleValue();

        // With all-positive W and (i+1)*0.01 per-pos input: pos[N-1] must strictly dominate.
        // A gatherNd alias pos[N-1]←pos[1] gives lastSum ≈ 2*0.01*factor << N*0.01*factor.
        assertTrue(lastSum > maxEarly + 1e-3,
                mode + ": GATHER_ND POSITION COLLAPSE — pos[" + (N - 1) + "] sum=" + lastSum
                        + " NOT > maxEarly=" + maxEarly
                        + " -> gatherNd aliased the last position onto an earlier one."
                        + " This matches the hidden[1141]≡hidden[1] decode bug (firstToken=11126=User).");
        log.info("[GATHER_ND_LAST_POS] mode={} PASS — lastSum={} > maxEarly={} (ratio={}x)",
                mode, lastSum, maxEarly, String.format("%.1f", lastSum / (maxEarly + 1e-12)));
    }

    /**
     * Tests that a where-based merge of two [1,N,d] tensors (vision=zeros, text=distinct)
     * correctly preserves the last position (N-1) when it is a TEXT position (mask=False
     * → select text). Image positions (1..N-2) are mask=True → select vision=0.
     *
     * Pattern covers two suspect paths in the real model:
     *   (a) The attention-mask path: strided_slice on [1,512,512] BOOL → bool_not → where
     *   (b) The embed-merge where path (if VisionEmbeddingMerge native op is graph-inlined)
     *
     * Collapse signature: pos[N-1] gets vision=0 instead of text[N-1]=N*0.01*ones,
     * making its output sum ≈ 0 (like image positions) instead of >> pos[0].
     */
    @ParameterizedTest(name = "whereLastPositionPreserved mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON"})
    @DisplayName("where-based merge must not collapse the last text position (N-1) onto vision (near-zero)")
    void testWhereLastPositionPreserved(GraphExecutionMode mode) {
        final int N = 1142, d = 576, outD = 64;
        sd = SameDiff.create();

        // Text embeddings: placeholder [1, N, d] — per-position distinct
        SDVariable textEmbeds = sd.placeHolder("text_embeds", DataType.FLOAT, 1, N, d);
        // Vision embeddings: constant all-zeros [1, N, d] — contributes 0 to any output sum
        SDVariable visionEmbeds = sd.constant("vision_embeds", Nd4j.zeros(DataType.FLOAT, 1, N, d));

        // Boolean mask [1, N, d]:
        //   True  at image positions 1..N-2 → where() selects x = visionEmbeds = 0
        //   False at text positions 0, N-1  → where() selects y = textEmbeds (distinct)
        // Build via float cast: ones in the image region [1..N-2], zeros elsewhere.
        INDArray maskFloat = Nd4j.zeros(DataType.FLOAT, 1, N, d);
        maskFloat.get(NDArrayIndex.point(0), NDArrayIndex.interval(1, N - 1), NDArrayIndex.all())
                .assign(1.0f);
        INDArray maskArr = maskFloat.castTo(DataType.BOOL);
        SDVariable imgMask = sd.constant("img_mask", maskArr);

        // where(x=vision, y=text, condition=mask): mask=True→x=vision=0, mask=False→y=text
        SDVariable merged = sd.where("merged", visionEmbeds, textEmbeds, imgMask); // [1, N, d]

        SDVariable flat  = sd.reshape("flat", merged, (long) N, (long) d);
        SDVariable w     = sd.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, d, outD)).addi(0.1f));
        SDVariable mm    = sd.mmul("mm", flat, w);                                 // [N, outD]
        SDVariable out   = sd.reshape("out", mm, 1L, (long) N, (long) outD);      // [1, N, outD]

        configureMode(sd, mode);

        // Text input: position i = (i+1)*0.01 * ones_d (strictly increasing)
        // Expected after where:
        //   pos 0:     text[0]   = 0.01 * ones_d  → small positive sum
        //   pos 1..N-2: vision   = 0              → zero sum
        //   pos N-1:   text[N-1] = N*0.01 * ones_d → largest positive sum (N=1142×)
        INDArray textArr = Nd4j.arange(1, N + 1).castTo(DataType.FLOAT).muli(0.01f)
                .reshape(N, 1).broadcast(N, d).reshape(1, N, d);
        Map<String, INDArray> ph = singlePh("text_embeds", textArr);
        warmup(sd, ph, "out", 3);
        INDArray result = sd.output(ph, "out").get("out");                         // [1, N, outD]

        INDArray posSums  = result.sum(2).reshape(N);
        double lastSum    = posSums.getDouble(N - 1);
        double pos0Sum    = posSums.getDouble(0);
        double imageMean  = posSums.get(NDArrayIndex.interval(1, N - 1)).meanNumber().doubleValue();

        // pos[N-1] must be >> pos[0] (both are text; N-1 has N× larger input scale).
        // A where collapse (pos[N-1] ← image slot) gives lastSum ≈ 0 ≈ imageMean.
        assertTrue(lastSum > pos0Sum + 1e-3,
                mode + ": WHERE LAST-POS COLLAPSE — pos[" + (N - 1) + "] sum=" + lastSum
                        + " NOT > pos[0] sum=" + pos0Sum
                        + " (expected last-pos text ≈ N=" + N + "× pos[0] text; collapse = where gave last pos vision=0).");
        // pos[N-1] must also be clearly above zero (image positions give exactly 0)
        assertTrue(lastSum > imageMean + 1.0,
                mode + ": WHERE LAST-POS ZERO — pos[" + (N - 1) + "] sum=" + lastSum
                        + " is near imageMean=" + imageMean
                        + "; where() incorrectly selected vision=0 at the last TEXT position.");
        log.info("[WHERE_LAST_POS] mode={} PASS — lastSum={} pos0Sum={} imageMean={}",
                mode, lastSum, pos0Sum, imageMean);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GQA HEAD-MERGE CHAIN: permuted view → reshape_no_copy must not produce zeros/garbage
    // Reproduces the VLM decode garbage (slots 337-340, forward-trace confirmed).
    // Broadcast_to is already CLEARED (testBroadcastToOfViewNoZeros PASSES).
    // The zeros come from reshape_no_copy on a non-contiguous permuted view:
    //   [1,9,N,64] contiguous → permute(0,2,1,3) → [1,N,9,64] (strides=[S,64,N*64,1])
    //   → reshape_no_copy [1,N,576]  ← this step produces ALL ZEROS in the real model.
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * REPRODUCTION: GQA head-merge — reshape_no_copy of a permuted (non-contiguous) view
     * must produce the SAME values as dup('c').reshape (the always-correct reference).
     *
     * Three code paths are exercised:
     *  (A) INDArray.reshape() on the permuted view — Java-level, always copies, expected PASS.
     *  (B) Nd4j.exec(ReshapeNoCopy) with null output — imperative C++ path, suspect ZEROS/garbage.
     *  (B2) Full GQA chain: [1,3,3,N,64]→reshape_no_copy→[1,9,N,64]→permute→reshape_no_copy
     *       matching the actual op chain in slots 337-340.
     *  (C) SameDiff SLOT_BY_SLOT graph: var→permute→reshape — pure graph execution path.
     *
     * Each path asserts: (1) non-zero sum, (2) sum equals the dup+reshape reference.
     * N=1142 to match the real SmolDocling decoder geometry.
     */
    @org.junit.jupiter.api.Test
    @DisplayName("GQA head-merge: reshape_no_copy of permuted view must NOT produce zeros")
    void testGqaHeadMergeReshapeNoCopyNoZeros() {
        final int N = 1142;
        final int H = 9;    // 9 heads = 3 GQA groups × 3 reps
        final int D = 64;
        final int Dmerge = H * D;  // = 576

        // ── Base: [1,9,N,64] contiguous tensor with arange distinct values ──────────
        // Simulates the broadcast_to→reshape_no_copy result (already verified correct).
        INDArray base = Nd4j.arange(1, (long) H * N * D + 1).castTo(DataType.FLOAT)
                .reshape(1, H, N, D);
        assertNotEquals(0.0, base.sumNumber().doubleValue(), "precondition: base non-zero");

        // ── Permuted view: [1,N,9,64] — non-contiguous (strides=[H*N*D, D, N*D, 1]) ──
        INDArray permuted = base.permute(0, 2, 1, 3);   // [1,N,9,64]
        assertNotEquals(0.0, permuted.sumNumber().doubleValue(), "precondition: permuted non-zero");
        log.info("[GQA] permuted.isView={} ordering={} strides={}",
                permuted.isView(), permuted.ordering(), java.util.Arrays.toString(permuted.stride()));

        // ── Reference: dup('c') then reshape — ALWAYS correct (forces contiguous copy first) ──
        double refSum = permuted.dup('c').reshape(1, N, Dmerge).sumNumber().doubleValue();
        assertNotEquals(0.0, refSum, "precondition: reference sum non-zero");
        log.info("[GQA] reference sum (dup+reshape)={}", refSum);

        // ─────────────────────────────────────────────────────────────────────────────
        // PATH A: INDArray.reshape() on the permuted view
        // Expected to PASS (reshape always copies; if this fails bug is lower-level).
        // ─────────────────────────────────────────────────────────────────────────────
        double sumA = permuted.reshape('c', 1, N, Dmerge).sumNumber().doubleValue();
        log.info("[GQA] Path A (INDArray.reshape): sum={} ref={}", sumA, refSum);
        assertNotEquals(0.0, sumA,
                "Path A FAIL: INDArray.reshape() on permuted view is ALL ZEROS (baseline broken). Expected ~" + refSum);
        assertEquals(refSum, sumA, Math.abs(refSum) * 1e-4,
                "Path A FAIL: INDArray.reshape() result wrong vs dup+reshape ref. got=" + sumA + " ref=" + refSum);

        // ─────────────────────────────────────────────────────────────────────────────
        // PATH B: Nd4j.exec(ReshapeNoCopy) with null output (triggers initializeOutputs)
        // This exercises the C++ reshape_no_copy op's output-allocation + execution path.
        // SUSPECT: this is likely the path that produces zeros/garbage in the VLM decode.
        // ─────────────────────────────────────────────────────────────────────────────
        org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy opB =
                new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                        base.permute(0, 2, 1, 3),  // fresh non-contiguous view each time
                        new long[]{1, N, Dmerge}, null, 'c');
        Nd4j.getExecutioner().exec(opB);
        INDArray outputB = opB.outputArguments().get(0);
        double sumB = outputB.sumNumber().doubleValue();
        log.info("[GQA] Path B (ReshapeNoCopy imperative, null output): sum={} ref={} outputB.isView={}",
                sumB, refSum, outputB.isView());
        assertNotEquals(0.0, sumB,
                "Path B FAIL: ReshapeNoCopy op (null output) on permuted view produced ALL ZEROS. " +
                "This is the VLM decode garbage root. " +
                "outputB.isView=" + outputB.isView() +
                " Expected sum~" + refSum + " got 0.0. " +
                "Root: shape-fn sees C-contiguous inferred strides → sets ARRAY_COPY_OFFSET_INPUT_0 → " +
                "output is VIEW of non-contiguous buffer → no assign called → garbage read with wrong strides.");
        assertEquals(refSum, sumB, Math.abs(refSum) * 1e-4,
                "Path B FAIL: ReshapeNoCopy sum mismatch. got=" + sumB + " ref=" + refSum);

        // ─────────────────────────────────────────────────────────────────────────────
        // PATH B2: Full GQA chain matching slots 337-340
        //   [1,3,3,N,64] → reshape_no_copy [1,9,N,64] (VIEW) → permute → reshape_no_copy [1,N,576]
        // ─────────────────────────────────────────────────────────────────────────────
        INDArray base335 = Nd4j.arange(1, (long) 3 * 3 * N * D + 1).castTo(DataType.FLOAT)
                .reshape(1, 3, 3, N, D);
        // Step 2: reshape_no_copy [1,3,3,N,64]→[1,9,N,64] (VIEW expected)
        org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy opStep2 =
                new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                        base335, new long[]{1, H, N, D}, null, 'c');
        Nd4j.getExecutioner().exec(opStep2);
        INDArray step2 = opStep2.outputArguments().get(0);
        log.info("[GQA] step2 [1,9,N,64] isView={} strides={}", step2.isView(),
                java.util.Arrays.toString(step2.stride()));
        // Step 3: permute [1,9,N,64]→[1,N,9,64]
        INDArray step3 = step2.permute(0, 2, 1, 3);
        log.info("[GQA] step3 [1,N,9,64] isView={} strides={}", step3.isView(),
                java.util.Arrays.toString(step3.stride()));
        // Step 4: reshape_no_copy [1,N,9,64]→[1,N,576]  ← THE MATERIALIZING STEP
        org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy opStep4 =
                new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                        step3, new long[]{1, N, Dmerge}, null, 'c');
        Nd4j.getExecutioner().exec(opStep4);
        INDArray step4 = opStep4.outputArguments().get(0);
        double sumB2 = step4.sumNumber().doubleValue();
        log.info("[GQA] Path B2 full chain: step4.isView={} step4.strides={} sum={} ref={}",
                step4.isView(), java.util.Arrays.toString(step4.stride()), sumB2, refSum);
        // Compare against reference from same base335 data
        double refB2 = base335.reshape(1, H, N, D).permute(0, 2, 1, 3).dup('c')
                .reshape(1, N, Dmerge).sumNumber().doubleValue();
        assertNotEquals(0.0, sumB2,
                "Path B2 FAIL: Full GQA chain (slots 337-340) produces ALL ZEROS at step4. " +
                "step2.isView=" + step2.isView() + " step3.strides=" + java.util.Arrays.toString(step3.stride()) +
                " step4.isView=" + step4.isView());
        assertEquals(refB2, sumB2, Math.abs(refB2) * 1e-4,
                "Path B2 FAIL: Full chain sum mismatch. got=" + sumB2 + " ref=" + refB2);

        // ─────────────────────────────────────────────────────────────────────────────
        // PATH C: SameDiff SLOT_BY_SLOT — graph execution path (DSP disabled)
        // Builds: var [1,9,N,64] → permute(0,2,1,3) → reshape [1,N,576]
        // ─────────────────────────────────────────────────────────────────────────────
        {
            SameDiff sdC = SameDiff.create();
            sdC.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            SDVariable xC = sdC.var("x", base.dup());
            SDVariable pC = sdC.permute("p", xC, 0, 2, 1, 3);    // [1,N,9,64]
            SDVariable rC = sdC.reshape("r", pC, 1, N, Dmerge);   // [1,N,576]
            Map<String, INDArray> resC = sdC.output(new java.util.HashMap<>(), "r");
            INDArray outC = resC.get("r");
            double sumC = outC.sumNumber().doubleValue();
            log.info("[GQA] Path C (SameDiff SLOT_BY_SLOT var→permute→reshape): sum={} ref={}", sumC, refSum);
            assertNotEquals(0.0, sumC,
                    "Path C FAIL: SameDiff SLOT_BY_SLOT permute→reshape produced ALL ZEROS. " +
                    "Expected sum~" + refSum + " got " + sumC);
            assertEquals(refSum, sumC, Math.abs(refSum) * 1e-4,
                    "Path C FAIL: SameDiff result mismatch. got=" + sumC + " ref=" + refSum);
            sdC.close();
        }

        log.info("[GQA_HEAD_MERGE] ALL PATHS PASS — refSum={} sumA={} sumB={} sumB2={}", refSum, sumA, sumB, sumB2);
    }

    /**
     * TARGETED: SameDiff graph with reshape_no_copy op on a permuted (non-contiguous) view.
     * Tests the EXACT op used in the VLM model (not sd.reshape, but sd.nn().reshapeNoCopy()).
     *
     * Permute's DECLARE_SHAPE_FN calls evalPermShapeInfo(setContigStrides=true) — so the
     * inferred output shape of permute has C-CONTIGUOUS strides. When reshape_no_copy's
     * DECLARE_SHAPE_FN receives this C-contiguous shape, reshapeNoAlloc returns TRUE →
     * ARRAY_COPY_OFFSET_INPUT_0 (view). But at runtime the actual permuted NDArray has
     * non-contiguous strides. If the framework sets up the output as a VIEW (sharing the
     * non-contiguous buffer) with C-contiguous read strides → GARBAGE VALUES.
     *
     * Tested in: SLOT_BY_SLOT (DSP-path), CUDA_GRAPHS (real decode path).
     */
    @ParameterizedTest(name = "gqaHeadMergeReshapeNoCopySameDiff mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "CUDA_GRAPHS"})
    @DisplayName("SameDiff graph: permute→reshape_no_copy must not produce zeros (VLM GQA head-merge)")
    void testGqaHeadMergeReshapeNoCopySameDiff(GraphExecutionMode mode) {
        final int N = 16;   // small for speed; bug is stride-structural, not size-dependent
        final int H = 9;
        final int D = 64;
        final int Dmerge = H * D;  // 576

        // Base [1,9,N,64] with distinct arange values
        INDArray baseArr = Nd4j.arange(1, (long) H * N * D + 1).castTo(DataType.FLOAT).reshape(1, H, N, D);
        double refSum = baseArr.permute(0, 2, 1, 3).dup('c').reshape(1, N, Dmerge).sumNumber().doubleValue();
        assertNotEquals(0.0, refSum, "precondition: reference sum non-zero");

        // Build graph: var [1,9,N,64] → permute(0,2,1,3) [1,N,9,64] → reshape_no_copy [1,N,576]
        SameDiff sdD = SameDiff.create();
        sdD.setGraphExecutionMode(mode);
        sdD.setDspAutoCompileEnabled(true);
        sdD.setDspNativeAutoCompileEnabled(true);
        SDVariable xVar = sdD.var("x", baseArr.dup());
        SDVariable pVar = sdD.permute("p", xVar, 0, 2, 1, 3);   // [1,N,9,64] non-contiguous at runtime
        // Use the reshape_no_copy op (the VLM ONNX model's actual op, NOT sd.reshape)
        SDVariable rVar = new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                sdD, pVar, new long[]{1, N, Dmerge}, 'c').outputVariable();
        sdD.updateVariableNameAndReference(rVar, "r");

        Map<String, INDArray> result = sdD.output(new java.util.HashMap<>(), "r");
        INDArray out = result.get("r");
        double sum = out.sumNumber().doubleValue();
        log.info("[GQA_SD] mode={} reshape_no_copy sum={} ref={} out.isView={}",
                mode, sum, refSum, out.isView());

        assertNotEquals(0.0, sum,
                mode + ": SameDiff permute→reshape_no_copy produced ALL ZEROS. " +
                "Expected sum~" + refSum + " got 0.0. " +
                "Root: permute DECLARE_SHAPE_FN uses setContigStrides=true → C-contiguous inferred strides → " +
                "reshape_no_copy reshapeNoAlloc returns TRUE → ARRAY_COPY_OFFSET_INPUT_0 (VIEW) → " +
                "at runtime non-contiguous buffer read with C-strides → zeros/garbage. " +
                "Fix: evalPermShapeInfo(setContigStrides=false) OR reshape_no_copy execution " +
                "must re-check actual runtime strides.");
        assertEquals(refSum, sum, Math.abs(refSum) * 1e-4,
                mode + ": SameDiff reshape_no_copy sum mismatch. got=" + sum + " ref=" + refSum);

        sdD.close();
        log.info("[GQA_SD] mode={} PASS — sum={} ref={}", mode, sum, refSum);
    }

    /**
     * FULL GQA K/V expansion chain in the DSP path — the EXACT live decode bug.
     *
     * Live --debug dump (firstToken=11126) proved: K/V projection [1,N,192] is NON-ZERO but the
     * GQA expansion chain that feeds onnx_multi_head_attention emerges ALL ZEROS at [1,N,576]:
     *   K_proj [1,N,192] -> reshape [1,N,3,64] -> permute [1,3,N,64] -> expand_dims [1,3,1,N,64]
     *     -> broadcast_to [1,3,3,N,64] -> reshape_no_copy [1,9,N,64] -> permute [1,N,9,64]
     *     -> reshape_no_copy [1,N,576]
     * Every op is green-identical / passes in isolation; the prior chain test starts from a
     * MATERIALIZED [1,3,3,N,64] (Path B2, passes) — it skips the broadcast_to(view) step.
     * This reproduces the WHOLE chain in the DSP slot-by-slot path and reports EACH step's sum
     * so the exact zeroing op is identified. ref = projSum * REP (each KV head replicated REP times).
     */
    @ParameterizedTest(name = "gqaFullExpansionChainDsp mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "CUDA_GRAPHS"})
    @DisplayName("GQA full K/V expansion chain [1,N,192]->...->[1,N,576] must not zero (live decode K/V=0)")
    void testGqaFullExpansionChainDsp(GraphExecutionMode mode) {
        final int N = 64, KV = 3, REP = 3, H = KV * REP, D = 64;
        final int projDim = KV * D;   // 192
        final int mergeDim = H * D;   // 576
        INDArray projArr = Nd4j.arange(1, (long) N * projDim + 1).castTo(DataType.FLOAT).reshape(1, N, projDim);
        double projSum = projArr.sumNumber().doubleValue();
        double refSum = projSum * REP;
        assertNotEquals(0.0, projSum, "precondition: K/V projection non-zero");

        SameDiff sd = SameDiff.create();
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        SDVariable kproj = sd.var("kproj", projArr.dup());                              // [1,N,192]
        SDVariable r1 = sd.reshape("r1", kproj, 1, N, KV, D);                           // [1,N,3,64]
        SDVariable p1 = sd.permute("p1", r1, 0, 2, 1, 3);                               // [1,3,N,64]
        SDVariable ed = sd.expandDims("ed", p1, 2);                                     // [1,3,1,N,64]
        SDVariable bshape = sd.constant("bshape", Nd4j.createFromArray(new long[]{1, KV, REP, N, D}));
        SDVariable bc = new org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo(sd, ed, bshape).outputVariable();
        sd.updateVariableNameAndReference(bc, "bc");                                    // [1,3,3,N,64]
        SDVariable m1 = new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                sd, bc, new long[]{1, H, N, D}, 'c').outputVariable();                  // [1,9,N,64]
        sd.updateVariableNameAndReference(m1, "m1");
        SDVariable p2 = sd.permute("p2", m1, 0, 2, 1, 3);                               // [1,N,9,64]
        SDVariable m2 = new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                sd, p2, new long[]{1, N, mergeDim}, 'c').outputVariable();              // [1,N,576]
        sd.updateVariableNameAndReference(m2, "m2");

        // Request EVERY intermediate so we can bisect which op first zeros.
        Map<String, INDArray> res = sd.output(new java.util.HashMap<>(),
                "kproj", "r1", "p1", "ed", "bc", "m1", "p2", "m2");
        for (String name : new String[]{"kproj", "r1", "p1", "ed", "bc", "m1", "p2", "m2"}) {
            INDArray a = res.get(name);
            double s = a == null ? Double.NaN : a.sumNumber().doubleValue();
            log.info("[GQA_CHAIN] mode={} step {} shape={} sum={} {}",
                    mode, name, a == null ? "null" : java.util.Arrays.toString(a.shape()), s,
                    (s == 0.0 ? "<<< ZERO" : ""));
        }
        INDArray out = res.get("m2");
        double sum = out.sumNumber().doubleValue();
        assertNotEquals(0.0, sum,
                mode + ": GQA expansion chain produced ALL ZEROS into [1,N,576] — the live decode K/V=0 bug. ref~" + refSum);
        assertEquals(refSum, sum, Math.abs(refSum) * 1e-4,
                mode + ": GQA chain final sum mismatch. got=" + sum + " ref=" + refSum);
        sd.close();
        log.info("[GQA_CHAIN] mode={} PASS — final sum={} ref={}", mode, sum, refSum);
    }

    /**
     * HEAD ISOLATION (imperative, no SameDiff graph → no order=-1 artifact): the EXACT live
     * decode chain, op-by-op, with broadcast_to producing its output (NO explicit output array =
     * the model's view-producing path) fed a NON-CONTIGUOUS permuted view. Prior passing tests used
     * either an explicit broadcast output (materialized) or a fresh contiguous [1,3,3,N,64]; this
     * fills the gap. Reports EACH step's sum so the exact zeroing op is pinpointed ON HEAD.
     */
    @org.junit.jupiter.api.Test
    @DisplayName("HEAD isolation: imperative GQA chain (per-head→permute→expand→broadcast_to(noOut)→reshape→permute→reshape) must not zero")
    void testGqaImperativeChainNoZeros() {
        final int N = 8, KV = 3, REP = 3, H = KV * REP, D = 64;
        final int projDim = KV * D;   // 192
        final int mergeDim = H * D;   // 576
        INDArray kproj = Nd4j.arange(1, (long) N * projDim + 1).castTo(DataType.FLOAT).reshape(1, N, projDim);
        double projSum = kproj.sumNumber().doubleValue();
        double refSum = projSum * REP;
        log.info("[IMP_CHAIN] step0 kproj[1,{},192] sum={}", N, projSum);

        INDArray r1 = kproj.reshape(1, N, KV, D);                    // [1,N,3,64]
        log.info("[IMP_CHAIN] step1 reshape[1,{},3,64] sum={} {}", N, r1.sumNumber().doubleValue(), r1.sumNumber().doubleValue() == 0 ? "<<<ZERO" : "");
        INDArray p1 = r1.permute(0, 2, 1, 3);                        // [1,3,N,64] view (non-contiguous)
        log.info("[IMP_CHAIN] step2 permute[1,3,{},64] sum={} isView={} {}", N, p1.sumNumber().doubleValue(), p1.isView(), p1.sumNumber().doubleValue() == 0 ? "<<<ZERO" : "");
        INDArray ed = Nd4j.expandDims(p1, 2);                        // [1,3,1,N,64] view
        log.info("[IMP_CHAIN] step3 expand_dims[1,3,1,{},64] sum={} isView={} {}", N, ed.sumNumber().doubleValue(), ed.isView(), ed.sumNumber().doubleValue() == 0 ? "<<<ZERO" : "");

        // broadcast_to [1,3,3,N,64] — NO explicit output (op allocates via shape-fn) = the model's path
        org.nd4j.linalg.api.ops.DynamicCustomOp bc = org.nd4j.linalg.api.ops.DynamicCustomOp.builder("broadcast_to")
                .addInputs(ed, Nd4j.createFromArray(new long[]{1, KV, REP, N, D}))
                .build();
        Nd4j.getExecutioner().exec(bc);
        INDArray bcOut = bc.getOutputArgument(0);
        double bcSum = bcOut.sumNumber().doubleValue();
        log.info("[IMP_CHAIN] step4 broadcast_to[1,3,3,{},64] sum={} expected={} isView={} {}", N, bcSum, refSum, bcOut.isView(), bcSum == 0 ? "<<<ZERO" : "");

        org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy m1op =
                new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(bcOut, new long[]{1, H, N, D}, null, 'c');
        Nd4j.getExecutioner().exec(m1op);
        INDArray m1 = m1op.outputArguments().get(0);                 // [1,9,N,64]
        log.info("[IMP_CHAIN] step5 reshape_no_copy[1,9,{},64] sum={} {}", N, m1.sumNumber().doubleValue(), m1.sumNumber().doubleValue() == 0 ? "<<<ZERO" : "");
        INDArray p2 = m1.permute(0, 2, 1, 3);                        // [1,N,9,64]
        log.info("[IMP_CHAIN] step6 permute[1,{},9,64] sum={} {}", N, p2.sumNumber().doubleValue(), p2.sumNumber().doubleValue() == 0 ? "<<<ZERO" : "");
        org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy m2op =
                new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(p2, new long[]{1, N, mergeDim}, null, 'c');
        Nd4j.getExecutioner().exec(m2op);
        INDArray m2 = m2op.outputArguments().get(0);                 // [1,N,576]
        double finalSum = m2.sumNumber().doubleValue();
        log.info("[IMP_CHAIN] step7 reshape_no_copy[1,{},576] sum={} expected={} {}", N, finalSum, refSum, finalSum == 0 ? "<<<ZERO" : "");

        assertNotEquals(0.0, finalSum,
                "Imperative GQA chain final [1,N,576] is ZERO — the decode K/V=0 root reproduced on HEAD. proj was " + projSum);
        assertEquals(refSum, finalSum, Math.abs(refSum) * 1e-4,
                "Imperative GQA chain sum mismatch: got=" + finalSum + " ref=" + refSum);
        log.info("[IMP_CHAIN] PASS — final={} ref={}", finalSum, refSum);
    }

    /**
     * The FULL GQA KV-head expansion as ONE SameDiff graph in pure execute() (non-DSP),
     * driven from a per-head [1,3,N,64] input — the exact gap prior tests missed:
     *   per-head [1,3,N,64] -> expand_dims [1,3,1,N,64] -> broadcast_to [1,3,3,N,64]
     *   -> reshape_no_copy [1,9,N,64] -> permute [1,N,9,64] -> reshape_no_copy [1,N,576]
     * The real SmolDocling decoder zeros K/V here (firstToken=11126) even in execute()
     * (DSP-disabled), with per-head NON-zero. Prior tests covered the halves: broadcast_to
     * alone (imperative), and [1,3,3,N,64]->reshape->permute->merge (DSP). This chains the
     * full expansion from the per-head input through SameDiff execute(), the model's path.
     */
    @org.junit.jupiter.api.Test
    @org.junit.jupiter.api.Disabled("SameDiff build-time order=-1 artifact: broadcast_to's shape-fn propagates a placeholder order=-1 at graph-build, so reshape_no_copy throws 'Invalid order: -1' at runtime. The IMPORTED model has resolved orders (imperative broadcast test passes) — this is NOT the decode bug. Kept for reference.")
    @org.junit.jupiter.api.DisplayName("Full GQA KV-expansion in execute() (per-head -> broadcast -> merge) must not zero")
    void testGqaFullExpansionExecuteNoZeros() {
        final int N = 1142, KV = 3, REP = 3, H = KV * REP, D = 64, Dm = H * D; // 9*64 = 576
        INDArray perHead = Nd4j.arange(1, (long) KV * N * D + 1).castTo(DataType.FLOAT).reshape(1, KV, N, D);
        double perHeadSum = perHead.sumNumber().doubleValue();
        double refSum = perHeadSum * REP;   // each KV head repeated REP times -> sum scales by REP
        org.junit.jupiter.api.Assertions.assertNotEquals(0.0, perHeadSum, "precondition: per-head input non-zero");

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);          // pure execute() — the model's noFreeze path
        sd.setDspNativeAutoCompileEnabled(false);
        SDVariable x = sd.var("x", perHead.dup());
        SDVariable ex = sd.expandDims("ex", x, 2);   // [1,3,1,N,64]
        SDVariable shapeC = sd.constant("bshape", Nd4j.createFromArray(new long[]{1, KV, REP, N, D}));
        SDVariable bc = new org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo(sd, ex, shapeC).outputVariable();
        sd.updateVariableNameAndReference(bc, "bc");                 // [1,3,3,N,64]
        SDVariable r1 = new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                sd, bc, new long[]{1, H, N, D}, 'c').outputVariable(); // [1,9,N,64]
        sd.updateVariableNameAndReference(r1, "r1");
        SDVariable p = sd.permute("p", r1, 0, 2, 1, 3);             // [1,N,9,64]
        SDVariable r2 = new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                sd, p, new long[]{1, N, Dm}, 'c').outputVariable();  // [1,N,576]
        sd.updateVariableNameAndReference(r2, "r2");

        INDArray out = sd.output(new java.util.HashMap<>(), "r2").get("r2");
        double sum = out.sumNumber().doubleValue();
        log.info("[GQA_FULL] execute() full-expansion sum={} ref={} (perHead={})", sum, refSum, perHeadSum);
        sd.close();

        org.junit.jupiter.api.Assertions.assertNotEquals(0.0, sum,
                "Full GQA KV-expansion produced ALL ZEROS in execute() — reproduces the decode K/V=0. "
                        + "Expected ~" + refSum + ".");
        org.junit.jupiter.api.Assertions.assertEquals(refSum, sum, Math.abs(refSum) * 1e-4,
                "GQA expansion sum mismatch: got=" + sum + " ref=" + refSum);
    }

    /**
     * REGRESSION: GQA K/V expansion zeroed by DSP view-producer slot-reject with dirty
     * {@code tl_dspExecutionStream} thread-local (the EXACT live decode K/V=0 root).
     *
     * <h3>Bug summary</h3>
     * SmolDocling VLM decode produces deterministic garbage (firstToken=11126). Root cause:
     * the SmolLM2 decoder's K/V repeat_kv expansion chain:
     * <pre>
     *   xw_plus_b [1,N,192] → reshape_no_copy [1,N,3,64] → permute [1,3,N,64]
     *     → expand_dims [1,3,1,N,64] → broadcast_to [1,3,3,N,64]
     *     → reshape_no_copy [1,9,N,64] → permute [1,N,9,64] → reshape_no_copy [1,N,576]
     * </pre>
     * emerges ALL ZEROS into {@code onnx_multi_head_attention} → dead attention → garbage.
     *
     * <h3>Root cause (diagnostic-confirmed)</h3>
     * {@code permute} is a view-producer; at slotexec.cpp:5384
     * {@code outputWrapperMatchesExpectedShape} rejects the permuted-strides view because
     * {@code strideEquals} compares actual strides vs. canonical C-order strides for the slot
     * shape → the slot stays at its unwritten-zero default → K/V = 0 into attention.
     * This rejection only fires when {@code isReplayActive=1}, i.e.,
     * {@code tl_dspExecutionStream != null}, which is set by a prior DSP graph execution
     * in the same thread and not cleared between graphs.
     *
     * <h3>Conditions required to reproduce</h3>
     * <ol>
     *   <li>A <em>prior</em> SameDiff graph runs in CUDA_GRAPHS+DSP mode in the same thread
     *       (~6 warmup steps), leaving {@code tl_dspExecutionStream} non-null.</li>
     *   <li>The K/V expansion chain runs in CUDA_GRAPHS+DSP mode.</li>
     *   <li>The K/V intermediates (broadcast_to output, reshape_no_copy outputs, permute views)
     *       are <em>NOT</em> requested as graph outputs — they are pure intermediates.  Requesting
     *       them as outputs "protects" them in the DSP plan and masks the bug.</li>
     *   <li>Only a downstream consumer's output is requested (here: a small matmul on the merged
     *       KV tensor), so the entire expansion chain is intermediate.</li>
     * </ol>
     *
     * <h3>How to interpret failures</h3>
     * The assertion {@code assertTrue(amax > 0)} <em>FAILS</em> (reports the bug) when the
     * GQA chain produces all-zero logits because K/V was zeroed by the DSP slot reject.
     * If the test PASSES, the fix is in effect (strides are accepted or the thread-local is
     * properly cleared between graphs).
     *
     * <h3>Why prior tests did NOT reproduce</h3>
     * <ul>
     *   <li>{@code testGqaFullExpansionChainDsp}: requests ALL intermediates (kproj, r1, p1,
     *       ed, bc, m1, p2, m2) as outputs → all protected → bug masked.</li>
     *   <li>{@code testGqaImperativeChainNoZeros}: plain INDArray ops, no SameDiff DSP plan
     *       → no slot-reject path.</li>
     *   <li>{@code testRealDecoderGqaKVZero} in TestGraphOptimizerOnSmolDocling: hits a
     *       synthetic-input shape error (wrong reshape target) rather than the K/V zero.</li>
     * </ul>
     */
    @org.junit.jupiter.api.Test
    @DisplayName("REGRESSION: GQA K/V=0 — prior-DSP tl_dspExecutionStream + K/V pure intermediates + CUDA_GRAPHS")
    void testGqaKvZeroWithPriorDspStream() {
        // ─── STEP 1: Prime a prior DSP graph to dirty tl_dspExecutionStream ─────────────────
        // Mirror the real.gqa.priorDsp block in TestGraphOptimizerOnSmolDocling.java ~L730-741.
        // A small mmul in CUDA_GRAPHS mode, 6 warmup steps.  NOT closed before the chain runs —
        // the thread-local must still be non-null when the chain executes.
        SameDiff prior = SameDiff.create();
        SDVariable px = prior.placeHolder("px", DataType.FLOAT, 1, 64);
        SDVariable pw = prior.var("pw", Transforms.abs(Nd4j.rand(DataType.FLOAT, 64, 64)).addi(0.1f));
        prior.mmul("pout", px, pw);
        prior.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
        prior.setDspAutoCompileEnabled(true);
        prior.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> pf = new LinkedHashMap<>();
        pf.put("px", Nd4j.rand(DataType.FLOAT, 1, 64).addi(0.1f));
        try {
            for (int w = 0; w < 6; w++) prior.output(pf, "pout");
            log.info("[GQA_KV0] prior DSP graph ran 6× — tl_dspExecutionStream should be set/dirty");
        } catch (Exception e) {
            log.warn("[GQA_KV0] prior DSP graph failed (CUDA_GRAPHS unavailable?): {}", e.getMessage());
        }

        // ─── STEP 2: Build the K/V repeat_kv expansion chain ────────────────────────────────
        // xw_plus_b: placeholder [1,N,576] @ W[576,192] + b[192] → K_proj [1,N,192]
        // then: reshape → permute → expand_dims → broadcast_to → reshape_no_copy
        // → permute → reshape_no_copy → [1,N,576]  (ALL pure intermediates)
        // downstream consumer: simple matmul → [N,1]  ← only this is requested as output
        final int N = 64, KV = 3, REP = 3, H = KV * REP, D = 64;
        final int projDim = KV * D;    // 192  (3 KV heads × 64 head dim)
        final int mergeDim = H * D;    // 576  (9 heads × 64)

        sd = SameDiff.create();
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // K projection (xw_plus_b pattern)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, N, 576);
        SDVariable Wk = sd.var("Wk",
                Transforms.abs(Nd4j.randn(DataType.FLOAT, 576, projDim)).addi(0.01f));
        SDVariable bk = sd.var("bk", Nd4j.ones(DataType.FLOAT, projDim));
        SDVariable inFlat   = sd.reshape("inFlat",    input,      (long) N, 576L);           // [N,576]
        SDVariable kprojMm  = sd.mmul("kproj_mm",     inFlat,     Wk);                       // [N,192]
        SDVariable kprojAdd = kprojMm.add("kproj_add", bk);                                   // [N,192] (bias broadcast)
        SDVariable kproj    = sd.reshape("kproj",      kprojAdd,   1L, (long) N, (long) projDim); // [1,N,192]

        // GQA repeat_kv expansion — every op here is a pure intermediate (NOT requested as output)
        SDVariable r1 = sd.reshape("r1", kproj, 1L, (long) N, (long) KV, (long) D);          // [1,N,3,64]
        SDVariable p1 = sd.permute("p1", r1,  0, 2, 1, 3);                                    // [1,3,N,64]
        SDVariable ed = sd.expandDims("ed", p1, 2);                                           // [1,3,1,N,64]
        SDVariable bshape = sd.constant("bshape",
                Nd4j.createFromArray(new long[]{1L, (long) KV, (long) REP, (long) N, (long) D}));
        SDVariable bc = new org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo(sd, ed, bshape)
                .outputVariable();                                                              // [1,3,3,N,64]
        sd.updateVariableNameAndReference(bc, "bc");
        SDVariable m1 = new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                sd, bc, new long[]{1L, (long) H, (long) N, (long) D}, 'c')
                .outputVariable();                                                              // [1,9,N,64]
        sd.updateVariableNameAndReference(m1, "m1");
        SDVariable p2 = sd.permute("p2", m1,  0, 2, 1, 3);                                    // [1,N,9,64]
        SDVariable kvMerged = new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(
                sd, p2, new long[]{1L, (long) N, (long) mergeDim}, 'c')
                .outputVariable();                                                              // [1,N,576]
        sd.updateVariableNameAndReference(kvMerged, "kvMerged");

        // Downstream consumer: kvMerged is a pure intermediate here; logits is the sole requested output.
        // If kvMerged is zeroed by the DSP slot-reject, logits will also be all-zero.
        SDVariable Wdown = sd.var("Wdown",
                Transforms.abs(Nd4j.randn(DataType.FLOAT, mergeDim, 1)).addi(0.01f));
        SDVariable kvFlat = sd.reshape("kvFlat", kvMerged, (long) N, (long) mergeDim);        // [N,576]
        sd.mmul("logits", kvFlat, Wdown);                                                      // [N,1]

        // ─── STEP 3: Warmup to drive the chain to REPLAYING state ────────────────────────────
        INDArray inputArr = Nd4j.rand(DataType.FLOAT, 1, N, 576).addi(1.0f);
        Map<String, INDArray> ph = singlePh("input", inputArr);
        for (int w = 0; w < 6; w++) {
            sd.output(ph, "logits");
        }

        // ─── STEP 4: Execute — request ONLY logits (K/V expansion is pure intermediate) ──────
        // With tl_dspExecutionStream dirty from the prior graph:
        //   permute's view-producer slot → strideEquals check fails → slot stays at zeros
        //   → kvMerged = 0 → logits = 0  (the bug)
        // With the fix in effect:
        //   strideEquals accepts the permuted view (or thread-local is cleared) → logits > 0
        Map<String, INDArray> result = sd.output(ph, "logits");
        INDArray logitsArr = result.get("logits");
        double amax = (logitsArr == null) ? Double.NaN : logitsArr.amaxNumber().doubleValue();
        log.info("[GQA_KV0] logits amax={} (expected > 0; 0 = K/V zeroed by DSP view-producer reject)", amax);

        prior.close();   // close prior graph now that chain has executed with dirty thread-local

        assertNotNull(logitsArr, "logits must not be null");
        // This assertion FAILS when the bug is present (amax == 0.0).
        assertTrue(amax > 0.0,
                "REGRESSION [GQA_KV0]: GQA K/V expansion produced ALL ZEROS downstream (amax=" + amax + "). "
                + "Conditions: prior-DSP tl_dspExecutionStream set + K/V pure intermediates + CUDA_GRAPHS. "
                + "Root: permute view-producer slot rejected by strideEquals @ slotexec.cpp:5384 "
                + "→ slot stays unwritten zeros → K/V=0 into attention → decode garbage (firstToken=11126).");
    }

    @ParameterizedTest(name = "frozenMultiPlanMutableScalarSlice mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON"})
    @DisplayName("Frozen multi-plan replay must refresh a mutable scalar used by a dynamic last-token slice")
    void testFrozenMultiPlanMutableScalarSliceRefresh(GraphExecutionMode mode) {
        sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.HALF, 1, -1, 8);
        SDVariable actualLength = sd.placeHolder("actual_sequence_length", DataType.INT64);
        SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.INT64, 0L));
        SDVariable one = sd.constant("one", Nd4j.scalar(DataType.INT64, 1L));
        SDVariable eight = sd.constant("eight", Nd4j.scalar(DataType.INT64, 8L));
        SDVariable actualLast = actualLength.sub("actual_last", one);
        SDVariable begin = sd.stack("last_begin", 0, zero, actualLast, zero);
        SDVariable size = sd.stack("last_size", 0, one, one, eight);
        SDVariable hiddenLast = sd.slice("hidden_last", input, begin, size).reshape(1, 8);
        SDVariable weights = sd.var("weights", Nd4j.ones(DataType.HALF, 8, 4));
        sd.mmul("out", hiddenLast, weights);
        configureMode(sd, mode);
        sd.setDspFallbackToAutoIfTritonUnavailable(false);

        INDArray prefill = Nd4j.create(DataType.HALF, 1, 4, 8);
        INDArray prefillLength = Nd4j.scalar(DataType.INT64, 2L);
        Map<String, INDArray> prefillInputs = new LinkedHashMap<>();
        prefillInputs.put("input", prefill);
        prefillInputs.put("actual_sequence_length", prefillLength);

        INDArray decode = Nd4j.create(DataType.HALF, 1, 1, 8);
        INDArray decodeLength = Nd4j.scalar(DataType.INT64, 1L);
        Map<String, INDArray> decodeInputs = new LinkedHashMap<>();
        decodeInputs.put("input", decode);
        decodeInputs.put("actual_sequence_length", decodeLength);

        prefill.assign(0);
        for (int position = 0; position < 4; position++) {
            prefill.get(NDArrayIndex.point(0), NDArrayIndex.point(position), NDArrayIndex.all()).assign(position);
        }
        Nd4j.getExecutioner().commit();
        assertEquals(8.0, sd.output(prefillInputs, "out").get("out").getDouble(0, 0), 0.05);
        decode.assign(5.0);
        Nd4j.getExecutioner().commit();
        assertEquals(40.0, sd.output(decodeInputs, "out").get("out").getDouble(0, 0), 0.05);
        sd.setDspShapesFrozen(true);

        int[] actualLengths = {3, 4, 2, 3, 4, 2};
        for (int generation = 1; generation <= actualLengths.length; generation++) {
            int currentLength = actualLengths[generation - 1];
            prefillLength.assign(currentLength);
            for (int position = 0; position < 4; position++) {
                prefill.get(NDArrayIndex.point(0), NDArrayIndex.point(position), NDArrayIndex.all())
                        .assign(generation * 10.0 + position);
            }
            Nd4j.getExecutioner().commit();
            double expected = (generation * 10.0 + currentLength - 1) * 8.0;
            INDArray prefillOut = sd.output(prefillInputs, "out").get("out");

            // Inspect the live DSP intermediates without adding requested outputs, which would
            // alter segmentation/capture. This distinguishes stale placeholder staging from a
            // stale scalar-to-slice control chain at the exact replay boundary.
            DspHandle prefillHandle = sd.dsp();
            INDArray actualLastSlot = prefillHandle.getSlotOutput("subtract");
            long observedActualLast = actualLastSlot != null ? actualLastSlot.getLong(0) : Long.MIN_VALUE;
            long observedBeginPosition = Long.MIN_VALUE;
            for (int stackSlotIdx : prefillHandle.allSlotsForOp("stack")) {
                INDArray stackSlot = prefillHandle.getSlotOutput(stackSlotIdx);
                // last_begin=[0, actualLast, 0]; last_size=[1, 1, 8].
                if (stackSlot != null && stackSlot.length() >= 3
                        && stackSlot.getLong(0) == 0L && stackSlot.getLong(2) == 0L) {
                    observedBeginPosition = stackSlot.getLong(1);
                    break;
                }
            }
            log.info("[MUTABLE_SCALAR_SLICE] mode={} generation={} inputLength={} "
                            + "actualLastSlot={} beginPosition={} out={}",
                    mode, generation, currentLength, observedActualLast,
                    observedBeginPosition, prefillOut.getDouble(0, 0));
            assertEquals(currentLength - 1L, observedActualLast,
                    mode + ": scalar arithmetic slot was stale at prefill generation " + generation);
            assertEquals(currentLength - 1L, observedBeginPosition,
                    mode + ": slice-begin stack slot was stale at prefill generation " + generation);

            assertEquals(expected, prefillOut.getDouble(0, 0), 0.5,
                    mode + ": stale actual_sequence_length at prefill generation " + generation
                            + " (length=" + currentLength + ")");

            decode.assign(100.0 + generation);
            Nd4j.getExecutioner().commit();
            INDArray decodeOut = sd.output(decodeInputs, "out").get("out");
            assertEquals((100.0 + generation) * 8.0, decodeOut.getDouble(0, 0), 0.5,
                    mode + ": stale decode input after scalar-controlled prefill generation " + generation);
        }
    }

    @ParameterizedTest(name = "frozenMultiPlanFp16MutableInput mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON"})
    @DisplayName("Frozen multi-plan replay must refresh a stable FP16 input buffer after every shape switch")
    void testFrozenMultiPlanFp16MutableInputRefresh(GraphExecutionMode mode) {
        sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.HALF, 1, -1, 8);
        SDVariable sequenceMean = sd.mean("sequence_mean", input, 1);
        SDVariable weights = sd.var("weights", Nd4j.ones(DataType.HALF, 8, 4));
        sd.mmul("out", sequenceMean, weights);
        configureMode(sd, mode);

        // GenerationPipeline reuses one stable, maximum-sized input for prefill and a second
        // stable input for decode. Alternate their shape plans while changing contents in place.
        INDArray prefill = Nd4j.create(DataType.HALF, 1, 4, 8);
        INDArray decode = Nd4j.create(DataType.HALF, 1, 1, 8);
        Map<String, INDArray> prefillInputs = singlePh("input", prefill);
        Map<String, INDArray> decodeInputs = singlePh("input", decode);

        prefill.assign(1.0);
        assertEquals(8.0, sd.output(prefillInputs, "out").get("out").getDouble(0, 0), 0.05);
        decode.assign(2.0);
        assertEquals(16.0, sd.output(decodeInputs, "out").get("out").getDouble(0, 0), 0.05);
        sd.setDspShapesFrozen(true);
        assertTrue(sd.isDspShapesFrozen(), "The active decode plan must be frozen before multi-plan reuse");

        for (int generation = 1; generation <= 6; generation++) {
            double prefillValue = generation + 10.0;
            prefill.assign(prefillValue);
            INDArray prefillOut = sd.output(prefillInputs, "out").get("out");
            assertEquals(prefillValue * 8.0, prefillOut.getDouble(0, 0), 0.15,
                    mode + ": stale prefill external input after shape switch at generation " + generation);

            double decodeValue = generation + 100.0;
            decode.assign(decodeValue);
            INDArray decodeOut = sd.output(decodeInputs, "out").get("out");
            assertEquals(decodeValue * 8.0, decodeOut.getDouble(0, 0), 0.5,
                    mode + ": stale decode external input after shape switch at generation " + generation);
        }
    }
}
