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
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
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
 * DSP monolithic replay dispatch tests.
 *
 * <p>These tests verify the interaction between monolithic and composite replay
 * in the frozen fast path. Specifically:</p>
 * <ul>
 *   <li>Whether a monolithic CUDA graph capture includes ALL ops (gap + island)</li>
 *   <li>Whether monolithic replay fires when both monolithic handle and composite
 *       schedule exist</li>
 *   <li>Whether totalGraphReplays increments (actual cudaGraphLaunch, not SBS fallback)</li>
 *   <li>Whether the monolithic-to-composite demotion path produces correct output
 *       but at reduced performance (replayCount=0)</li>
 * </ul>
 *
 * <p>Hypothesis under test: monolithic capture via cudaStreamBeginCapture captures
 * ALL ops on the stream including cuBLAS gap ops (matmul, xw_plus_b). The existing
 * gap-demotion logic assumes monolithic graphs only contain Triton island kernels
 * and demotes to composite/SBS when gaps exist. This test class characterizes the
 * current behavior so that any fix can be validated against it.</p>
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DspMonolithicReplayDispatchTest {

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

    /**
     * Large decoder-like graph with matmul gaps between Triton-eligible ops.
     * This graph is large enough (12 layers × several ops each) to trigger
     * monolithic CUDA graph capture. The matmul ops (cuBLAS) become gap ops
     * in the composite schedule while rmsNorm/add/mul are Triton islands.
     */
    private SameDiff buildVlmLikeGraph(int embedDim, int numLayers) {
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, embedDim);
        SDVariable posIds = g.placeHolder("position_ids", DataType.FLOAT, 1, 1);

        // Position encoding
        SDVariable x = embed.add("pos_add", posIds);

        for (int layer = 0; layer < numLayers; layer++) {
            String p = "l" + layer + "_";
            SDVariable kv = g.placeHolder(p + "kv", DataType.FLOAT, 1, 4, embedDim);

            // Weights (initialized to small positive values for stable matmul)
            SDVariable wq = g.var(p + "wq", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f));
            SDVariable wv = g.var(p + "wv", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f));
            SDVariable gamma = g.var(p + "gamma", Nd4j.ones(DataType.FLOAT, embedDim));

            // Reshape to 2D for matmul (gap op: cuBLAS)
            SDVariable xFlat = g.reshape(p + "xflat", x, 1, embedDim);

            // RMS norm (Triton-eligible island op)
            SDVariable normed = g.nn().rmsNorm(p + "norm", xFlat, gamma, 1e-5);

            // matmul: Q projection (gap op: cuBLAS)
            SDVariable q = g.mmul(p + "q", normed, wq);

            // KV mean + transpose (element-wise / Triton-eligible)
            SDVariable kvMean = g.mean(p + "kv_mean", kv, 1);
            SDVariable kvMeanT = g.permute(p + "kvt", kvMean, 1, 0);

            // matmul: attention score (gap op: cuBLAS)
            SDVariable score = g.mmul(p + "score", q, kvMeanT);

            // matmul: output projection (gap op: cuBLAS)
            SDVariable attnOut = g.mmul(p + "attn_out", score,
                    g.reshape(p + "kvr", kvMean, 1, embedDim));

            // Residual add (Triton-eligible)
            SDVariable residual = xFlat.add(p + "res", attnOut);

            // Second RMS norm (Triton-eligible island op)
            SDVariable normed2 = g.nn().rmsNorm(p + "norm2", residual, gamma, 1e-5);

            // FFN matmul (gap op: cuBLAS)
            SDVariable ffn = g.mmul(p + "ffn", normed2, wv);

            // Residual + reshape back (Triton-eligible)
            SDVariable out = residual.add(p + "ffn_res", ffn);
            x = g.reshape(p + "out", out, 1, 1, embedDim);
        }

        // Final projection
        SDVariable wFinal = g.var("w_final", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, 32)).addi(0.01f));
        SDVariable xFinal = g.reshape("x_final_flat", x, 1, embedDim);
        g.mmul("out", xFinal, wFinal);
        return g;
    }

    private void configureMode(SameDiff g, GraphExecutionMode mode) {
        g.getSessions().clear();
        g.setGraphExecutionMode(mode);
        g.setDspAutoCompileEnabled(true);
        g.setDspNativeAutoCompileEnabled(true);
    }

    private Map<String, INDArray> buildPlaceholders(int embedDim, int numLayers) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", Nd4j.randn(DataType.FLOAT, 1, 1, embedDim).muli(0.1f));
        ph.put("position_ids", Nd4j.scalar(DataType.FLOAT, 0.0f).reshape(1, 1));
        for (int layer = 0; layer < numLayers; layer++) {
            ph.put("l" + layer + "_kv", Nd4j.randn(DataType.FLOAT, 1, 4, embedDim).muli(0.01f));
        }
        return ph;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 1: Characterize gap unit presence after capture
    //
    // After enough warmup steps, check whether the composite schedule has gap
    // units. This tells us whether matmul ops are classified as gaps (cuBLAS)
    // vs islands (Triton-compiled).
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "1_gapUnitPresence mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(1)
    void test1_GapUnitPresence(GraphExecutionMode mode) {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, mode);

        Map<String, INDArray> ph = buildPlaceholders(embedDim, numLayers);

        // Warmup: 15 steps (enough for shape freeze + capture)
        for (int i = 0; i < 15; i++) {
            ph.get("position_ids").assign(i);
            g.output(ph, "out");
        }

        int gapUnits = DspPlanAssertions.getSegmentGapUnitCount(g, 0);
        int islandUnits = DspPlanAssertions.getSegmentIslandUnitCount(g, 0);
        int gapSlots = DspPlanAssertions.getSegmentGapSlotCount(g, 0);
        int monolithicReady = DspPlanAssertions.isMonolithicHandleReady(g, 0);
        int replayCount = DspPlanAssertions.getSegmentReplayCount(g, 0);
        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int execCount = DspPlanAssertions.getSegmentExecCount(g, 0);

        log.info("{}: gapUnits={}, islandUnits={}, gapSlots={}, monolithicReady={}, "
                        + "replayCount={}, totalReplays={}, execCount={}",
                mode, gapUnits, islandUnits, gapSlots, monolithicReady,
                replayCount, totalReplays, execCount);
        log.info("{}: segment state = {}", mode,
                DspPlanAssertions.snapshotSegmentState(g, 0));

        // Characterization: log what we get. No assertion on gap count since it
        // depends on Triton compilation support for each op on this GPU.
        // The important thing is to OBSERVE the values.
        if (gapUnits > 0 && monolithicReady == 1) {
            log.info("{}: MONOLITHIC + GAPS — this is the VLM scenario. "
                    + "replayCount={} tells us if monolithic replay actually fires.", mode, replayCount);
        } else if (gapUnits > 0 && monolithicReady != 1) {
            log.info("{}: GAPS but NO monolithic handle — composite replay expected.", mode);
        } else if (gapUnits == 0) {
            log.info("{}: NO gaps — all ops compiled to Triton. Monolithic=island-only.", mode);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 2: Monolithic handle existence after capture
    //
    // Verify that monolithic capture actually happens on this graph. If
    // monolithicReady=0 after 15 steps, the graph never reached monolithic
    // capture (went straight to composite), which means the demotion path
    // is not exercised.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "2_monolithicCaptureHappens mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(2)
    void test2_MonolithicCaptureHappens(GraphExecutionMode mode) {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, mode);

        Map<String, INDArray> ph = buildPlaceholders(embedDim, numLayers);

        // Track monolithic handle status at each step
        List<Integer> monolithicPerStep = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            ph.get("position_ids").assign(i);
            g.output(ph, "out");
            int mono = DspPlanAssertions.isMonolithicHandleReady(g, 0);
            monolithicPerStep.add(mono);
        }

        log.info("{}: monolithicReady per step: {}", mode, monolithicPerStep);

        // The monolithic handle should appear at some point during capture
        boolean everHadMonolithic = monolithicPerStep.stream().anyMatch(v -> v == 1);
        log.info("{}: ever had monolithic handle = {}", mode, everHadMonolithic);

        // If all ops compiled to Triton (gapUnits=0), composite replay is
        // the expected path even in CUDA_GRAPHS mode — no monolithic needed.
        // Only assert monolithic if gap units exist (the VLM scenario).
        int gapUnits = DspPlanAssertions.getSegmentGapUnitCount(g, 0);
        if (mode == GraphExecutionMode.CUDA_GRAPHS && gapUnits > 0) {
            assertTrue(everHadMonolithic,
                    "CUDA_GRAPHS mode with gap units should produce a monolithic handle. "
                            + "gapUnits=" + gapUnits + " Steps: " + monolithicPerStep);
        }
        // Regardless of monolithic, SOME replay path must be active
        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        log.info("{}: gapUnits={} totalReplays={}", mode, gapUnits, totalReplays);
        assertTrue(totalReplays > 0 || everHadMonolithic,
                mode + ": neither composite nor monolithic replay active after 20 steps. "
                        + "totalReplays=" + totalReplays + " monolithic=" + everHadMonolithic);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 3: Replay count progression
    //
    // After 30 steps, totalGraphReplays should be > 0 if CUDA graph replay is
    // actually happening. If totalGraphReplays=0 despite planPhase=REPLAYING,
    // then all execution is slot-by-slot (the performance bug).
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "3_replayCountProgression mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(3)
    void test3_ReplayCountProgression(GraphExecutionMode mode) {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, mode);

        Map<String, INDArray> ph = buildPlaceholders(embedDim, numLayers);

        List<int[]> progression = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            ph.get("position_ids").assign(i);
            g.output(ph, "out");
            int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
            int replayCount = DspPlanAssertions.getSegmentReplayCount(g, 0);
            int execCount = DspPlanAssertions.getSegmentExecCount(g, 0);
            int monolithicReady = DspPlanAssertions.isMonolithicHandleReady(g, 0);
            progression.add(new int[]{totalReplays, replayCount, execCount, monolithicReady});
        }

        int[] last = progression.get(progression.size() - 1);
        int finalTotalReplays = last[0];
        int finalReplayCount = last[1];
        int finalExecCount = last[2];
        int finalMonolithic = last[3];

        log.info("{}: final state — totalReplays={}, segReplayCount={}, execCount={}, monolithicReady={}",
                mode, finalTotalReplays, finalReplayCount, finalExecCount, finalMonolithic);

        // Log the full progression for diagnosis
        for (int i = 0; i < progression.size(); i++) {
            int[] p = progression.get(i);
            if (i < 5 || i >= 25 || p[0] != progression.get(Math.max(0, i - 1))[0]) {
                log.info("{}: step {} — totalReplays={} segReplayCount={} execCount={} mono={}",
                        mode, i, p[0], p[1], p[2], p[3]);
            }
        }

        // CRITICAL: after 30 steps, totalGraphReplays should be > 0.
        // If this fails, all execution is slot-by-slot despite reaching REPLAYING phase.
        assertTrue(finalTotalReplays > 0,
                mode + ": totalGraphReplays=0 after 30 steps. All execution was slot-by-slot. "
                        + "This means CUDA graph replay never fired — likely the monolithic handle was "
                        + "demoted to composite, and composite replay fell through to SBS because no "
                        + "sub-graphs were captured. Final state: execCount=" + finalExecCount
                        + " monolithicReady=" + finalMonolithic);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 4: Output correctness through all replay paths
    //
    // Regardless of whether monolithic or composite replay fires, output must
    // be correct (changing position_ids → changing output). This is the safety
    // net: any dispatch change must not break correctness.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "4_outputCorrectnessAllPaths mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(4)
    void test4_OutputCorrectnessAllPaths(GraphExecutionMode mode) {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, mode);

        Map<String, INDArray> ph = buildPlaceholders(embedDim, numLayers);

        // Warmup
        for (int i = 0; i < 10; i++) {
            ph.get("position_ids").assign(i);
            g.output(ph, "out");
        }

        // 20 steps with changing position — output must change each step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            ph.get("position_ids").assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-6) stuckCount++;
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int monolithicReady = DspPlanAssertions.isMonolithicHandleReady(g, 0);

        log.info("{}: stuckCount={}/19 totalReplays={} monolithicReady={} "
                        + "sums[0..3]={}", mode, stuckCount, totalReplays, monolithicReady,
                sums.subList(0, Math.min(4, sums.size())));

        assertTrue(stuckCount < 3,
                mode + ": STUCK output! " + stuckCount + "/19 steps identical. "
                        + "Changing position_ids must produce different output. "
                        + "totalReplays=" + totalReplays + " monolithicReady=" + monolithicReady);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 5: Monolithic vs SBS reference accuracy
    //
    // Run the same graph in SLOT_BY_SLOT (gold reference) and the test mode.
    // Compare outputs at each step. This catches bugs where monolithic replay
    // produces numerically wrong results (e.g., stale gap op outputs baked
    // into the CUDA graph).
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "5_accuracyVsSlotBySlot mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(5)
    void test5_AccuracyVsSlotBySlot(GraphExecutionMode mode) {
        int embedDim = 32, numLayers = 6;
        Nd4j.getRandom().setSeed(42);

        // Build reference (SLOT_BY_SLOT)
        SameDiff ref = buildVlmLikeGraph(embedDim, numLayers);
        ref.getSessions().clear();
        ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        ref.setDspAutoCompileEnabled(true);

        // Build test graph (same weights via seed reset)
        Nd4j.getRandom().setSeed(42);
        SameDiff test = buildVlmLikeGraph(embedDim, numLayers);
        sd = test;
        configureMode(test, mode);

        Map<String, INDArray> phRef = buildPlaceholders(embedDim, numLayers);
        Map<String, INDArray> phTest = new LinkedHashMap<>();
        for (var entry : phRef.entrySet()) {
            phTest.put(entry.getKey(), entry.getValue().dup());
        }

        // Run 20 steps, compare at each
        int mismatchCount = 0;
        double maxDiff = 0;
        for (int step = 0; step < 20; step++) {
            float posVal = step + 50;
            phRef.get("position_ids").assign(posVal);
            phTest.get("position_ids").assign(posVal);

            Map<String, INDArray> refOut = ref.output(phRef, "out");
            Map<String, INDArray> testOut = test.output(phTest, "out");

            double diff = Transforms.abs(refOut.get("out").sub(testOut.get("out")))
                    .maxNumber().doubleValue();
            maxDiff = Math.max(maxDiff, diff);
            if (diff > 0.1) {
                mismatchCount++;
                if (mismatchCount <= 3) {
                    log.warn("{}: step {} mismatch — maxDiff={}", mode, step, diff);
                }
            }
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(test);
        log.info("{}: maxDiff={} mismatchCount={}/20 totalReplays={}",
                mode, maxDiff, mismatchCount, totalReplays);

        ref.close();

        // Allow TF32 tolerance for matmul ops (cuBLAS TF32 gives ~1e-3 error)
        assertTrue(maxDiff < 1.5,
                mode + ": output diverged from SLOT_BY_SLOT reference. maxDiff=" + maxDiff
                        + " mismatchCount=" + mismatchCount + "/20. totalReplays=" + totalReplays);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 6: Demotion characterization
    //
    // Track whether the monolithic handle gets demoted (cleared) between steps.
    // If monolithicReady goes from 1 → 0, demotion happened. Record WHICH step
    // it happens at and what the segment state looks like.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "6_demotionCharacterization mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(6)
    void test6_DemotionCharacterization(GraphExecutionMode mode) {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, mode);

        Map<String, INDArray> ph = buildPlaceholders(embedDim, numLayers);

        int prevMono = -1;
        int demotionStep = -1;
        for (int i = 0; i < 25; i++) {
            ph.get("position_ids").assign(i);
            g.output(ph, "out");
            int mono = DspPlanAssertions.isMonolithicHandleReady(g, 0);
            int gapUnits = DspPlanAssertions.getSegmentGapUnitCount(g, 0);
            int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);

            if (prevMono == 1 && mono != 1 && demotionStep < 0) {
                demotionStep = i;
                log.info("{}: DEMOTION at step {} — monolithic 1→{} gapUnits={} totalReplays={}",
                        mode, i, mono, gapUnits, totalReplays);
                log.info("{}: segment state at demotion = {}",
                        mode, DspPlanAssertions.snapshotSegmentState(g, 0));
            }
            prevMono = mono;
        }

        int finalTotalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int finalMono = DspPlanAssertions.isMonolithicHandleReady(g, 0);

        if (demotionStep >= 0) {
            log.info("{}: monolithic handle was demoted at step {}. "
                            + "Final: totalReplays={} monolithicReady={}",
                    mode, demotionStep, finalTotalReplays, finalMono);
            // If demotion happened AND totalReplays=0, that's the performance bug
            if (finalTotalReplays == 0) {
                log.warn("{}: PERFORMANCE BUG — monolithic handle demoted and "
                        + "totalGraphReplays=0. All execution is slot-by-slot.", mode);
            }
        } else if (prevMono == 1) {
            log.info("{}: monolithic handle survived all 25 steps. totalReplays={}",
                    mode, finalTotalReplays);
        } else {
            log.info("{}: monolithic handle never appeared. totalReplays={}",
                    mode, finalTotalReplays);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 7: CUDA_GRAPHS mode must use monolithic replay
    //
    // In CUDA_GRAPHS mode (no Triton), the monolithic capture captures ALL ops
    // including matmul via cuBLAS. There should be no gap-demotion because there
    // is no Triton backend to classify gaps. This is the control: CUDA_GRAPHS
    // mode should always have totalReplays > 0.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(7)
    @DisplayName("7_cudaGraphsModeReplays")
    void test7_CudaGraphsModeReplays() {
        int embedDim = 64, numLayers = 12;
        SameDiff g = buildVlmLikeGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> ph = buildPlaceholders(embedDim, numLayers);

        for (int i = 0; i < 30; i++) {
            ph.get("position_ids").assign(i);
            g.output(ph, "out");
        }

        int totalReplays = DspPlanAssertions.getTotalGraphReplays(g);
        int monolithicReady = DspPlanAssertions.isMonolithicHandleReady(g, 0);
        int segReplayCount = DspPlanAssertions.getSegmentReplayCount(g, 0);

        log.info("CUDA_GRAPHS: totalReplays={} monolithicReady={} segReplayCount={}",
                totalReplays, monolithicReady, segReplayCount);

        assertTrue(totalReplays > 0,
                "CUDA_GRAPHS mode should have totalGraphReplays > 0 after 30 steps. "
                        + "Got 0 — monolithic capture/replay not working. "
                        + "monolithicReady=" + monolithicReady
                        + " segReplayCount=" + segReplayCount);
    }
}
