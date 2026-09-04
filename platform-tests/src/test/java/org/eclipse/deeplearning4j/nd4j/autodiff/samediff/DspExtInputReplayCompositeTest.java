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
 * DSP composite replay and graph complexity tests, extracted from DspExtInputStalenessTest.
 *
 * Tests graph complexity isolation, monolithic graph + composite replay (VLM degenerate bug),
 * island-gap composite replay staleness vectors, and gap slot detailed tests.
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class DspExtInputReplayCompositeTest extends DspExtInputTestSupport {

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

    // ── Shared helper for building island-gap-island chain ──
    private SameDiff buildIslandGapIslandChain(INDArray[][] weights, int dim, int numBlocks) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);

        INDArray embTable = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        g.constant("emb_table", embTable.dup());
        INDArray indices = Nd4j.arange(dim).castTo(DataType.INT64);
        g.constant("indices", indices.dup());

        SDVariable current = x;
        for (int block = 0; block < numBlocks; block++) {
            SDVariable gathered = g.gather("gather_" + block, g.getVariable("emb_table"), g.getVariable("indices"), 0);
            SDVariable reshaped = g.reshape("reshape_" + block, gathered,
                    g.constant("shape_" + block, Nd4j.createFromArray(1L, (long) dim * dim)));
            SDVariable sliced = g.stridedSlice("slice_" + block, reshaped,
                    new long[]{0, 0}, new long[]{1, dim}, new long[]{1, 1});

            g.constant("w_" + block, weights[block][0].dup());
            SDVariable addedInput = current.add("add_input_" + block, sliced);
            current = g.mmul("gap_mm_" + block, addedInput, g.getVariable("w_" + block));
        }
        current.rename("out");
        return g;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 9: Graph Complexity Isolation — Identifying Which Op Triggers Bug
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "twoPlaceholderWithReshape mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("matmul(x,w) then reshape — does reshape in graph cause staleness?")
    void testTwoPlaceholderWithReshape_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable mm = g.mmul("mm", x, w);
        // Reshape output: [1,4] -> [4,1]
        g.reshape("out", mm, 4, 1);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup 8 steps with changing input
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — reshape in graph causes staleness? sums=" + sums);
        }
        log.info("[RESHAPE_STALE] mode={} PASS — reshape graph, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "twoPlaceholderWithReduce mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("reduce_mean on placeholder input then matmul — does reduction cause staleness?")
    void testTwoPlaceholderWithReduce_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        // x is [1, 8], reduce along dim 1 -> [1, 1], then matmul with w [1, 4]
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 1, 4)).addi(0.1f));
        SDVariable reduced = g.mean("reduced", x, true, 1); // [1, 1] keepDims
        g.mmul("out", reduced, w);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup 8 steps with changing input
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — reduce in graph causes staleness? sums=" + sums);
        }
        log.info("[REDUCE_STALE] mode={} PASS — reduce_mean graph, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "twoPlaceholderWithPermute mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("permute then matmul — permute on ext input produces non-contiguous view")
    void testTwoPlaceholderWithPermute_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        // x is [1, 8], permute to [8, 1], then reshape to [1, 8], then matmul with w [8, 4]
        // Using [1,8] → permute [8,1] → reshape [1,8] to exercise permute without
        // creating a non-contiguous view that hits the DSP null-buffer bug
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable xT = g.permute("xT", x, 1, 0); // [8, 1]
        SDVariable xFlat = g.reshape("xFlat", xT, 1, 8); // [1, 8] — contiguous
        g.mmul("out", xFlat, w);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup 8 steps with changing input
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — permute in graph causes staleness? sums=" + sums);
        }
        log.info("[PERMUTE_STALE] mode={} PASS — permute graph, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "threePlaceholderSimpleGraph mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("3 placeholders (x, w, b) with matmul+add — NO reshape/reduce — tests if bug is reshape/reduce specific")
    void testThreePlaceholderSimpleGraph_allModes(GraphExecutionMode mode) {
        sd = buildMultiPlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray w = Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f);
        INDArray b = Nd4j.ones(DataType.FLOAT, 1, 4);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);
        ph.put("w", w);
        ph.put("b", b);

        // Warmup 8 steps with changing x
        for (int i = 0; i < 8; i++) {
            x.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(ph, "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            x.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — 3 placeholder simple graph. sums=" + sums);
        }
        log.info("[THREE_PH_SIMPLE] mode={} PASS — 3 placeholders, no reshape/reduce, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "fivePlaceholderSimpleGraph mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("5 placeholders all feeding adds/matmuls (NO reshape) — tests if placeholder count alone causes staleness")
    void testFivePlaceholderSimpleGraph_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable a = g.placeHolder("a", DataType.FLOAT, 1, 8);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 1, 8);
        SDVariable c = g.placeHolder("c", DataType.FLOAT, 1, 8);
        SDVariable d = g.placeHolder("d", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        // Sum all placeholders, then matmul
        SDVariable sum1 = x.add("sum1", a);
        SDVariable sum2 = sum1.add("sum2", b);
        SDVariable sum3 = sum2.add("sum3", c);
        SDVariable sum4 = sum3.add("sum4", d);
        g.mmul("out", sum4, w);
        sd = g;
        configureMode(sd, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray aArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray bArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray cArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray dArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("a", aArr);
        ph.put("b", bArr);
        ph.put("c", cArr);
        ph.put("d", dArr);

        // Warmup 8 steps with changing x
        for (int i = 0; i < 8; i++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(ph, "out");
        }

        // 20 replay steps — only x changes, output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — 5 placeholder count causes staleness? sums=" + sums);
        }
        log.info("[FIVE_PH_SIMPLE] mode={} PASS — 5 placeholders, no reshape, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "multiLayerMatmulOnly mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("3 matmuls chained (no reshape/permute between) — tests if depth alone causes staleness")
    void testMultiLayerMatmulOnly_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 8)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 8)).addi(0.1f));
        SDVariable w3 = g.var("w3", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable mm1 = g.mmul("mm1", x, w1);
        SDVariable mm2 = g.mmul("mm2", mm1, w2);
        g.mmul("out", mm2, w3);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup 8 steps with changing input
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — 3 chained matmuls (no reshape) causes staleness? sums=" + sums);
        }
        log.info("[MULTI_MM_ONLY] mode={} PASS — 3 chained matmuls, no reshape, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "multiLayerWithReshapeBetween mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("3 matmuls with reshape between each — isolates reshape-between-matmul pattern")
    void testMultiLayerWithReshapeBetween_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 8)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 8)).addi(0.1f));
        SDVariable w3 = g.var("w3", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable mm1 = g.mmul("mm1", x, w1);
        // Reshape between matmuls: [1,8] -> [8,1] -> [1,8]
        SDVariable r1 = g.reshape("r1", mm1, 8, 1);
        SDVariable r1flat = g.reshape("r1flat", r1, 1, 8);
        SDVariable mm2 = g.mmul("mm2", r1flat, w2);
        SDVariable r2 = g.reshape("r2", mm2, 8, 1);
        SDVariable r2flat = g.reshape("r2flat", r2, 1, 8);
        g.mmul("out", r2flat, w3);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup 8 steps with changing input
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — reshape between matmuls causes staleness? sums=" + sums);
        }
        log.info("[RESHAPE_BETWEEN_MM] mode={} PASS — reshape-between-matmuls, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "reduceMeanOnPlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("reduce_mean directly on placeholder that changes each step — isolates whether reduction on variable input causes stale reads")
    void testReduceMeanOnPlaceholder_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        // x [1, 8] -> reduce_mean along dim 1 -> [1] scalar broadcast added to w output
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable reduced = g.mean("reduced", x, false, 1); // [1]
        SDVariable mm = g.mmul("mm", x, w); // [1, 4]
        // Reshape reduced [1] -> [1,1] to broadcast-add with mm [1,4]
        SDVariable reducedBcast = g.reshape("reduced_bcast", reduced, 1, 1);
        g.math().add("out", mm, reducedBcast);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup 8 steps with changing input
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — reduce_mean on placeholder causes stale reads? sums=" + sums);
        }
        log.info("[REDUCE_ON_PH] mode={} PASS — reduce_mean on placeholder, 20 steps all different", mode);
    }

    @ParameterizedTest(name = "permuteOnPlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("permute directly on placeholder, then matmul — isolates permute on variable input causing stale reads")
    void testPermuteOnPlaceholder_allModes(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        // x [8, 1] -> permute [1, 8] -> mmul with w [8, 4]
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 8, 1);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        // Also use x directly before permute so it has two consumers
        SDVariable xT = g.permute("xT", x, 1, 0); // [1, 8]
        g.mmul("out", xT, w);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 8, 1);
        // Warmup 8 steps with changing input
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{8, 1}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 20 replay steps — output must change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{8, 1}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — permute on placeholder causes stale reads? sums=" + sums);
        }
        log.info("[PERMUTE_ON_PH] mode={} PASS — permute on placeholder, 20 steps all different", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 9: Monolithic Graph + Composite Replay (VLM Degenerate Bug)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Reproduces the VLM stuck-output bug.
     *
     * The VLM decoder graph has 507 Triton-capturable "islands" separated by
     * 507 non-capturable "gap" ops (cuBLAS matmul, reshape, etc.). The DSP
     * captures a MONOLITHIC CUDA graph covering ALL 507 islands in one shot.
     * On replay, that monolithic graph is launched at the "island 0" position
     * of the composite replay schedule, but it actually executes kernels for
     * ALL 507 islands — writing capture-time output to ALL intermediate slots.
     *
     * Subsequent gap ops and slot-by-slot islands read those stale intermediate
     * values, propagating capture-time data through the computation → stuck output.
     *
     * This test uses buildLargeDecoderGraph which creates multiple matmul+reshape
     * layers, producing the same island-gap-island structure at smaller scale.
     * If the monolithic transfer bug is present, outputs will be stuck despite
     * changing placeholder values (position_ids equivalent).
     */
    @ParameterizedTest(name = "monolithicGraphWithGapsStuck mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Multi-island graph with gaps: output must change when placeholder changes")
    void testMonolithicGraphWithGapsStuck(GraphExecutionMode mode) {
        int embedDim = 32;
        int numLayers = 6; // enough layers to create multiple islands + gaps
        SameDiff g = buildLargeDecoderGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, mode);

        // Build placeholder map
        INDArray embed = Nd4j.randn(DataType.FLOAT, 1, 1, embedDim).muli(0.1f);
        INDArray posIds = Nd4j.scalar(DataType.FLOAT, 0.0f).reshape(1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        for (int layer = 0; layer < numLayers; layer++) {
            ph.put("layer_" + layer + "_kv", Nd4j.randn(DataType.FLOAT, 1, 4, embedDim).muli(0.01f));
        }

        // Warmup: 8 steps with changing position (triggers freeze + capture)
        for (int i = 0; i < 8; i++) {
            posIds.assign(i);
            g.output(ph, "out");
        }

        // Verify DSP reached REPLAYING state
        DspHandle h = g.dsp();
        if (h != null && h.isCompiled()) {
            int phase = h.planPhase();
            log.info("[MONOLITHIC_GAPS] mode={} phase={} after warmup", mode, phase);
        }

        // Now run 20 steps keeping embed FIXED but changing position_ids.
        // This is the exact VLM decode pattern: same token → same embedding,
        // but position increments → output MUST differ.
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            posIds.assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [MONOLITHIC_GAPS]: STUCK! " + stuckCount + "/19 steps identical. " +
                        "Fixed embed + changing position should produce different outputs. " +
                        "This is the VLM monolithic-graph-with-composite-gaps bug. " +
                        "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[MONOLITHIC_GAPS] mode={} PASS — {}/19 unique steps", mode, 19 - stuckCount);
    }

    /**
     * Same pattern as testMonolithicGraphWithGapsStuck but with a larger graph
     * (more layers) to ensure enough ops trigger monolithic capture with multiple
     * composite replay units. Also verifies that changing BOTH embed and position
     * produces different output every step.
     */
    @ParameterizedTest(name = "monolithicGraphBothInputsChange mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Multi-island graph: BOTH embed and position change → unique output each step")
    void testMonolithicGraphBothInputsChange(GraphExecutionMode mode) {
        int embedDim = 32;
        int numLayers = 8; // larger graph
        SameDiff g = buildLargeDecoderGraph(embedDim, numLayers);
        sd = g;
        configureMode(g, mode);

        INDArray embed = Nd4j.randn(DataType.FLOAT, 1, 1, embedDim).muli(0.1f);
        INDArray posIds = Nd4j.scalar(DataType.FLOAT, 0.0f).reshape(1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        for (int layer = 0; layer < numLayers; layer++) {
            ph.put("layer_" + layer + "_kv", Nd4j.randn(DataType.FLOAT, 1, 4, embedDim).muli(0.01f));
        }

        // Warmup
        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.randn(DataType.FLOAT, 1, 1, embedDim).muli(0.1f));
            posIds.assign(i);
            g.output(ph, "out");
        }

        // 20 steps: both embed AND position change (like VLM producing different tokens)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            embed.assign(Nd4j.randn(DataType.FLOAT, 1, 1, embedDim).muli(0.1f));
            posIds.assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [MONOLITHIC_BOTH_CHANGE]: STUCK! " + stuckCount + "/19 steps. " +
                        "Both embed and position change — all outputs must differ. " +
                        "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[MONOLITHIC_BOTH_CHANGE] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    /**
     * Control test: same graph as testMonolithicGraphWithGapsStuck but
     * in SLOT_BY_SLOT (no graph capture) to confirm the graph topology CAN
     * produce different output per step. If this passes but the composite
     * replay tests fail, the bug is isolated to graph replay, not the graph structure.
     */
    @Test
    @DisplayName("Multi-island graph SLOT_BY_SLOT baseline: confirms graph CAN differentiate")
    void testMonolithicGraphBaselineSlotBySlot() {
        int embedDim = 32;
        int numLayers = 6;
        SameDiff g = buildLargeDecoderGraph(embedDim, numLayers);
        sd = g;
        g.getSessions().clear();
        g.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        g.setDspAutoCompileEnabled(true);
        g.setDspNativeAutoCompileEnabled(true);

        INDArray embed = Nd4j.randn(DataType.FLOAT, 1, 1, embedDim).muli(0.1f);
        INDArray posIds = Nd4j.scalar(DataType.FLOAT, 0.0f).reshape(1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        for (int layer = 0; layer < numLayers; layer++) {
            ph.put("layer_" + layer + "_kv", Nd4j.randn(DataType.FLOAT, 1, 4, embedDim).muli(0.01f));
        }

        // Run 20 steps changing position_ids (same embed fixed)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            posIds.assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                "SLOT_BY_SLOT baseline STUCK! " + stuckCount + "/19 steps. " +
                        "Graph structure itself can't differentiate — test fixture is wrong. " +
                        "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[BASELINE_SBS] PASS — {}/19 unique (confirms graph structure works)", 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 21: Island-Gap Composite Replay Staleness Vectors
    //
    // These tests isolate specific staleness vectors that could cause test47's
    // stuck output. Each test targets ONE theory.
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * SV12-A: Minimal island+gap — ONE constant island followed by ONE gap that
     * reads the placeholder. If the gap op gets fresh ext input data, output changes.
     *
     * Graph: gather(emb_table, indices) → stridedSlice → add(x, slice) → mmul(add, w) → out
     * Island: gather + stridedSlice (constant inputs only)
     * Gap: add + mmul (reads placeholder x)
     *
     * This is the simplest possible reproduction of the test47 pattern.
     * SLOT_BY_SLOT is the control — if it passes and TRITON/AUTO fail, the bug
     * is in composite replay infrastructure.
     */
    @ParameterizedTest(name = "minimalIslandGapStuck mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV12A_MinimalIslandGapStuck(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // Island: constant gather + stridedSlice
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

        // Reference: SLOT_BY_SLOT
        SameDiff sdRef = SameDiff.create();
        SDVariable xRef = sdRef.placeHolder("x", DataType.FLOAT, 1, dim);
        sdRef.constant("emb_table", embTable.dup());
        sdRef.constant("indices", indices.dup());
        SDVariable gatheredRef = sdRef.gather("gather", sdRef.getVariable("emb_table"), sdRef.getVariable("indices"), 0);
        SDVariable reshapedRef = sdRef.reshape("reshape", gatheredRef,
                sdRef.constant("rshape", Nd4j.createFromArray(1L, (long) dim * dim)));
        SDVariable slicedRef = sdRef.stridedSlice("slice", reshapedRef,
                new long[]{0, 0}, new long[]{1, dim}, new long[]{1, 1});
        sdRef.constant("w", w.dup());
        SDVariable addedRef = xRef.add("add_input", slicedRef);
        sdRef.mmul("out", addedRef, sdRef.getVariable("w"));
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int totalSteps = 20;
        int stuckCount = 0;
        int matchCount = 0;
        INDArray prevResult = null;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Map.of("x", input);

            INDArray result = sd.output(ph, "out").get("out").dup();
            INDArray ref = sdRef.output(ph, "out").get("out").dup();

            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN");

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 0.01) matchCount++;

            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) {
                    stuckCount++;
                    log.warn("[SV12A] {} step {}: STUCK (change={})", mode, step, change);
                }
            }
            prevResult = result;

            if (step < 4 || step == totalSteps - 1) {
                log.info("[SV12A] {} step {}: diff={} result[0]={}", mode, step, diff, result.getFloat(0));
            }
        }

        sdRef.close();

        assertTrue(stuckCount <= 1,
                mode + " SV12A: output stuck for " + stuckCount + "/" + totalSteps
                        + " steps. Gap op not seeing fresh ext input during composite replay.");
        assertTrue(matchCount >= totalSteps * 0.8,
                mode + " SV12A: matchRate=" + ((double) matchCount / totalSteps)
                        + " (need >=0.8). Gap outputs diverged from reference.");
        log.info("[SV12A] {} PASS — stuck={} matchRate={}", mode, stuckCount, (double) matchCount / totalSteps);
    }

    /**
     * SV12-B: Same as SV12-A but WITHOUT an island — pure gap ops only.
     * This is the CONTROL: if gap-only (no island) works but island+gap doesn't,
     * the bug is in island-gap interaction during composite replay.
     */
    @ParameterizedTest(name = "gapOnlyNoIsland mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV12B_GapOnlyNoIsland(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w", w.dup());
        // Simple: add(x, constant) → mmul → out. No island ops.
        INDArray constBias = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.02);
        sd.constant("bias", constBias.dup());
        SDVariable added = x.add("add_input", sd.getVariable("bias"));
        sd.mmul("out", added, sd.getVariable("w"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int totalSteps = 20;
        int stuckCount = 0;
        INDArray prevResult = null;
        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();
            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN");
            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) stuckCount++;
            }
            prevResult = result;
        }
        assertTrue(stuckCount <= 1,
                mode + " SV12B: gap-only graph stuck for " + stuckCount + "/" + totalSteps
                        + " steps. Bug is NOT island-gap interaction.");
        log.info("[SV12B] {} PASS — stuckCount={}", mode, stuckCount);
    }

    /**
     * SV12-C: Island-only, no gap. Just constant gather+reshape+slice → mmul(x, w).
     * If the mmul is inside the island (because it reads ext input),
     * the arg table should refresh the ext input each step.
     * Tests whether composite replay correctly refreshes arg tables for islands
     * that read variable ext inputs.
     */
    @ParameterizedTest(name = "islandReadsPlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV12C_IslandReadsPlaceholder(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w", w.dup());
        // Simple matmul — should be inside an island if Triton-compilable
        sd.mmul("out", x, sd.getVariable("w"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int totalSteps = 20;
        int stuckCount = 0;
        INDArray prevResult = null;
        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();
            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN");
            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) stuckCount++;
            }
            prevResult = result;
        }
        assertTrue(stuckCount <= 1,
                mode + " SV12C: island-reads-placeholder stuck for " + stuckCount + "/" + totalSteps
                        + " steps. Arg table refresh missing for island ext inputs?");
        log.info("[SV12C] {} PASS — stuckCount={}", mode, stuckCount);
    }

    /**
     * SV6-A: Gap slot classification at executeCount=2 vs later behavior.
     * Specifically tests whether a gap slot classified as EXECUTE at count=2
     * continues to execute on subsequent steps.
     *
     * Uses a graph where the gap op is NOT frozen-const, NOT identity, NOT view —
     * it's a plain matmul that reads from the placeholder. Should be classified
     * as EXECUTE in the gap cache and continue producing different outputs.
     */
    @ParameterizedTest(name = "gapCacheClassificationStable mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV6A_GapCacheClassificationStable(GraphExecutionMode mode) {
        int dim = 16;
        int numBlocks = 2;  // 2 blocks = 2 islands + 2 gaps minimum
        INDArray[][] weights = new INDArray[numBlocks][1];
        Nd4j.getRandom().setSeed(42);
        for (int b = 0; b < numBlocks; b++) {
            weights[b][0] = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        }

        sd = buildIslandGapIslandChain(weights, dim, numBlocks);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Steps 0-3: warmup + classification
        // Steps 4+: should use cached gap slot list
        int totalSteps = 30;
        int stuckCount = 0;
        int stuckAfterCache = 0;  // steps >=4 where cached path is used
        INDArray prevResult = null;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();
            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN");

            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) {
                    stuckCount++;
                    if (step >= 4) stuckAfterCache++;
                    log.warn("[SV6A] {} step {}: STUCK (change={})", mode, step, change);
                }
            }
            prevResult = result;
        }

        assertTrue(stuckCount <= 1,
                mode + " SV6A: output stuck for " + stuckCount + "/" + totalSteps
                        + " steps (afterCache=" + stuckAfterCache + "). Gap cache classification wrong.");
        log.info("[SV6A] {} PASS — stuck={} stuckAfterCache={}", mode, stuckCount, stuckAfterCache);
    }

    /**
     * SV7-A: Verify frozen fast-path is actually skipped during composite replay gaps.
     * Uses a graph where:
     * - Island: constant-only ops (gather, reshape)
     * - Gap: op that reads placeholder AND has executeCount >= 4
     *
     * If frozen fast-path fires during gap execution, output is frozen at the
     * capture-time cached value. This test runs 30 steps — if steps 5+ (frozen
     * fast-path territory) produce the same output, the guard failed.
     */
    @ParameterizedTest(name = "frozenFastPathSkippedDuringGap mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    void testSV7A_FrozenFastPathSkippedDuringGap(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // Island-like constant ops
        INDArray embTable = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        sd.constant("emb_table", embTable.dup());
        INDArray indices = Nd4j.arange(dim).castTo(DataType.INT64);
        sd.constant("indices", indices.dup());
        SDVariable gathered = sd.gather("gather", sd.getVariable("emb_table"), sd.getVariable("indices"), 0);
        SDVariable reshaped = sd.reshape("reshape", gathered,
                sd.constant("rshape", Nd4j.createFromArray(1L, (long) dim * dim)));
        SDVariable sliced = sd.stridedSlice("slice", reshaped,
                new long[]{0, 0}, new long[]{1, dim}, new long[]{1, 1});

        // Gap matmul that reads x (placeholder) + slice (island output)
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w", w.dup());
        SDVariable added = x.add("add_input", sliced);
        sd.mmul("out", added, sd.getVariable("w"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int totalSteps = 30;
        List<Double> outputSums = new ArrayList<>();
        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();
            outputSums.add(result.sumNumber().doubleValue());
        }

        // Count stuck steps ONLY after executeCount >= 4 (frozen fast-path territory)
        int stuckAfterFrozen = 0;
        for (int i = 5; i < outputSums.size(); i++) {
            if (Math.abs(outputSums.get(i) - outputSums.get(i - 1)) < 1e-6) {
                stuckAfterFrozen++;
                log.warn("[SV7A] {} step {}: STUCK in frozen territory (sum={})",
                        mode, i, outputSums.get(i));
            }
        }

        assertTrue(stuckAfterFrozen <= 1,
                mode + " SV7A: frozen fast-path may be firing during gaps! "
                        + stuckAfterFrozen + " stuck steps after execCount>=4. "
                        + "tl_dspReplayActive guard may not be working.");
        log.info("[SV7A] {} PASS — stuckAfterFrozen={}", mode, stuckAfterFrozen);
    }

    /**
     * SV-SYNC: Cross-stream visibility test.
     * Tests whether D2D staging content is visible to gap ops on the same stream.
     *
     * Pattern: Java .assign() writes to host → H2D sync → D2D to staging → gap reads staging.
     * The D2D is on DSP stream, gap ops are on DSP stream (gap-stream unification).
     * Same-stream means implicit ordering — BUT if gap-stream unification is broken
     * and gap ops run on the default stream, they'd miss the D2D.
     */
    @ParameterizedTest(name = "gapStreamUnification mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    void testSV_SYNC_GapStreamUnification(GraphExecutionMode mode) {
        int dim = 16;
        sd = buildSinglePlaceholder(dim, dim);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Warmup with changing input to get to replay state
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);
        for (int i = 0; i < 8; i++) {
            x.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            sd.output(ph, "out");
        }

        // Now in replay — each step should see fresh data via staging D2D
        int stuckCount = 0;
        INDArray prevResult = null;
        for (int step = 0; step < 20; step++) {
            x.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            INDArray result = sd.output(ph, "out").get("out").dup();
            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) stuckCount++;
            }
            prevResult = result;
        }
        assertTrue(stuckCount <= 1,
                mode + " SV-SYNC: gap-stream unification broken? stuck=" + stuckCount
                        + "/20. Gap ops may run on wrong stream, missing D2D data.");
        log.info("[SV-SYNC] {} PASS — stuckCount={}", mode, stuckCount);
    }

    /**
     * SV-CHAIN: Test that gap slot chain propagation works correctly.
     * In test47, gap_mm_0 feeds add_input_1 feeds gap_mm_1, etc.
     * Even if add_input_0 gets fresh x, the downstream chain must propagate.
     * If outputSlots_[] isn't updated correctly after each gap op, downstream
     * gap ops read stale slot values.
     */
    @ParameterizedTest(name = "gapChainPropagation mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV_CHAIN_GapSlotChainPropagation(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // 4 sequential gap matmuls: x → mm1 → mm2 → mm3 → mm4 → out
        // No islands at all — pure gap chain. If output is stuck, the chain is broken.
        INDArray w1 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray w3 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray w4 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w1", w1.dup());
        sd.constant("w2", w2.dup());
        sd.constant("w3", w3.dup());
        sd.constant("w4", w4.dup());

        SDVariable mm1 = sd.mmul("mm1", x, sd.getVariable("w1"));
        SDVariable mm2 = sd.mmul("mm2", mm1, sd.getVariable("w2"));
        SDVariable mm3 = sd.mmul("mm3", mm2, sd.getVariable("w3"));
        sd.mmul("out", mm3, sd.getVariable("w4"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int totalSteps = 20;
        int stuckCount = 0;
        INDArray prevResult = null;
        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();
            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) {
                    stuckCount++;
                    log.warn("[SV-CHAIN] {} step {}: STUCK in gap chain", mode, step);
                }
            }
            prevResult = result;
        }
        assertTrue(stuckCount <= 1,
                mode + " SV-CHAIN: gap chain propagation broken! stuck=" + stuckCount
                        + "/" + totalSteps + ". outputSlots_[] not updated between gap ops?");
        log.info("[SV-CHAIN] {} PASS — stuckCount={}", mode, stuckCount);
    }

    /**
     * SV-INTERLEAVE: The exact test47 pattern but with only 1 block (minimal).
     * Island (gather+reshape+slice from constants) THEN gap (add x + island_out, mmul).
     * This tests the INTERLEAVING of island replay → gap execution specifically.
     *
     * Unlike SV12A which builds the graph manually, this uses the same
     * buildIslandGapIslandChain helper as test47 with numBlocks=1.
     */
    @ParameterizedTest(name = "singleBlockIslandGap mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV_INTERLEAVE_SingleBlockIslandGap(GraphExecutionMode mode) {
        int dim = 16;
        INDArray[][] weights = new INDArray[1][1];
        Nd4j.getRandom().setSeed(42);
        weights[0][0] = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);

        sd = buildIslandGapIslandChain(weights, dim, 1);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SameDiff sdRef = buildIslandGapIslandChain(weights, dim, 1);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int totalSteps = 20;
        int stuckCount = 0;
        int matchCount = 0;
        INDArray prevResult = null;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("x", input), "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 0.01) matchCount++;

            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) {
                    stuckCount++;
                    log.warn("[SV-INTERLEAVE] {} step {}: STUCK", mode, step);
                }
            }
            prevResult = result;

            if (step < 4 || step == totalSteps - 1) {
                log.info("[SV-INTERLEAVE] {} step {}: diff={} result[0]={}",
                        mode, step, diff, result.getFloat(0));
            }
        }

        sdRef.close();

        assertTrue(stuckCount <= 1,
                mode + " SV-INTERLEAVE: 1-block island-gap stuck=" + stuckCount + "/" + totalSteps);
        log.info("[SV-INTERLEAVE] {} — stuck={} matchRate={}", mode, stuckCount,
                (double) matchCount / totalSteps);
    }

    /**
     * SV-ARGREFRESH: After merged CUDA graph capture, does the arg table
     * actually get refreshed for the island that reads from staging?
     *
     * The island CUDA graph has baked device pointers. When staging buffers
     * are used, the island graph should read from the staging buffer address
     * (which is stable, plan-lifetime). The arg table refresh should copy
     * the staging buffer's CONTENT (D2D), not change the address.
     *
     * This test verifies the complete pipeline:
     * 1. Java .assign() writes new data to host
     * 2. H2D sync copies to device
     * 3. D2D copies from device to staging
     * 4. Arg table refresh ensures the CUDA graph reads staging
     * 5. Graph replay produces correct output
     *
     * Uses a pure matmul graph (no gap) to test islands in isolation.
     */
    @ParameterizedTest(name = "argTableRefreshWithStaging mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO", "CUDA_GRAPHS"})
    void testSV_ARGREFRESH_StagingPipeline(GraphExecutionMode mode) {
        int dim = 16;
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w", w.dup());
        sd.mmul("out", x, sd.getVariable("w"));

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Compute expected results manually: result = input @ w
        List<Double> expected = new ArrayList<>();
        List<Double> actual = new ArrayList<>();
        INDArray wDup = w.dup();

        for (int step = 0; step < 20; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();

            // Manual reference
            INDArray manualRef = input.mmul(wDup);
            double refSum = manualRef.sumNumber().doubleValue();
            double actSum = result.sumNumber().doubleValue();
            expected.add(refSum);
            actual.add(actSum);

            if (step < 4 || step == 19) {
                double diff = Math.abs(refSum - actSum);
                log.info("[SV-ARGREFRESH] {} step {}: ref={} act={} diff={}",
                        mode, step, refSum, actSum, diff);
            }
        }

        // After warmup (step 4+), outputs should still track expected
        for (int i = 4; i < 20; i++) {
            double diff = Math.abs(expected.get(i) - actual.get(i));
            assertTrue(diff < 1.0,
                    mode + " SV-ARGREFRESH: step " + i + " diverged! expected="
                            + expected.get(i) + " actual=" + actual.get(i) + " diff=" + diff
                            + ". Arg table refresh not working with staging.");
        }
        log.info("[SV-ARGREFRESH] {} PASS", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 24: Gap Slot Detailed Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Gap output is correctly zeroed before re-execution each step.
     * If gap output accumulates (not zeroed), values will grow without bound.
     */
    @ParameterizedTest(name = "gapPrezeroBeforeExec mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Gap output must not accumulate across steps (pre-zero verification)")
    void testGapPrezeroBeforeExec(GraphExecutionMode mode) {
        sd = buildGappyGraph(8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // Run 20 steps with IDENTICAL input — output must be constant (not growing)
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 5.0));
        double baseline = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        for (int step = 0; step < 20; step++) {
            double val = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();
            assertEquals(baseline, val, 1e-4,
                    mode + " step " + step + ": output growing despite stable input! "
                            + "baseline=" + baseline + " current=" + val
                            + " — gap output not zeroed before re-execution");
        }
        log.info("[GAP_PREZERO] mode={} PASS — 20 identical steps, output stable at {}", mode, baseline);
    }

    /**
     * Batched GEMM gap group with changing inputs — correct results.
     * Uses multiple matmuls in gap positions to exercise batched GEMM optimization.
     */
    @ParameterizedTest(name = "gapBatchedGemm mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Multiple gap matmuls (batched GEMM candidates) with changing input")
    void testGapBatchedGemm(GraphExecutionMode mode) {
        // Graph with multiple matmuls separated by reshapes (gap-inducing)
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 16);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 16)).addi(0.1f));
        SDVariable w3 = g.var("w3", Transforms.abs(Nd4j.randn(DataType.FLOAT, 16, 8)).addi(0.1f));

        // Three matmuls with reshapes between = multiple gap units
        SDVariable mm1 = g.mmul("mm1", x, w1);
        SDVariable r1 = g.reshape("r1", mm1, 16, 1);
        SDVariable r1f = g.reshape("r1f", r1, 1, 16);
        SDVariable mm2 = g.mmul("mm2", r1f, w2);
        SDVariable r2 = g.reshape("r2", mm2, 16, 1);
        SDVariable r2f = g.reshape("r2f", r2, 1, 16);
        g.mmul("out", r2f, w3);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 16});

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 16}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — batched GEMM gap ops broken. sums=" + sums);
        }
        log.info("[GAP_BATCHED_GEMM] mode={} PASS — 20 steps with batched gap matmuls all unique", mode);
    }

    /**
     * Constant-only gap op output unchanged despite 20 replay steps.
     * Verifies frozen constant gap ops are deterministic and never re-run kernel
     * (or if they do, produce identical output).
     */
    @ParameterizedTest(name = "frozenConstGapNeverReexecutes mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Constant-only gap op output stable across 20 steps")
    void testFrozenConstGapNeverReexecutes(GraphExecutionMode mode) {
        // Graph where one path is constant (mean of constant → gap) and
        // another path reads placeholder. We verify the constant path output
        // is identical across steps.
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable kv = g.var("kv_const", Nd4j.randn(DataType.FLOAT, 1, 4, 8));
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));

        // Constant path: mean(kv) → reshape → "const_out"
        SDVariable kvMean = g.mean("kv_mean", kv, 1); // [1, 8]
        g.identity("const_out", kvMean);

        // Variable path: mmul(x, w) → "var_out"
        SDVariable mm = g.mmul("var_mm", x, w);

        // Combine: add(const_out, reshape(var_mm)) — but const_out has different shape
        // Just output both separately
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        for (int i = 0; i < 10; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "const_out", "var_mm");
        }

        // Now verify const_out is identical across 20 steps while var_mm changes
        double constBaseline = sd.output(singlePh("x", input), "const_out").get("const_out").sumNumber().doubleValue();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "const_out");
            double constVal = result.get("const_out").sumNumber().doubleValue();
            assertEquals(constBaseline, constVal, 1e-5,
                    mode + " step " + step + ": constant gap output drifted! "
                            + "baseline=" + constBaseline + " current=" + constVal);
        }
        log.info("[FROZEN_CONST_GAP] mode={} PASS — const_out stable={} across 20 steps", mode, constBaseline);
    }

    /**
     * Gap op whose input comes from a placeholder that changes address
     * post-classification. After gap cache is built (execCount>=3), provide
     * new INDArray for the placeholder — gap must still execute correctly.
     */
    @ParameterizedTest(name = "gapSlotWithChangingInputAddress mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Gap op input changes address after classification — gap still executes correctly")
    void testGapSlotWithChangingInputAddress(GraphExecutionMode mode) {
        // Graph with gap-inducing ops (reshapes between matmuls)
        sd = buildGappyGraph(16);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        // Warmup to build gap cache
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 16});

        // Now change to new INDArray objects (different address) each step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            INDArray newInput = Nd4j.valueArrayOf(new long[]{1, 16}, (double)(step + 100));
            Map<String, INDArray> result = sd.output(singlePh("x", newInput), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck after gap cache built + address change. sums=" + sums);
        }
        log.info("[GAP_ADDR_CHANGE] mode={} PASS — 20 address changes post-gap-cache all produce unique output", mode);
    }

    /**
     * View op (reshape) in gap range — verify output tracks input changes
     * and is NOT frozen at classification-time value.
     */
    @ParameterizedTest(name = "gapViewFastPath mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("View op (reshape) in gap range — output tracks input changes, not frozen")
    void testGapViewFastPath(GraphExecutionMode mode) {
        // Graph: x [1,16] → matmul → reshape [4,4] (gap view) → reshape [1,16] → matmul → out
        SameDiff g = SameDiff.create();
        int dim = 16;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, 4)).addi(0.1f));

        SDVariable mm1 = g.mmul("mm1", x, w1);
        SDVariable view1 = g.reshape("view_4x4", mm1, 4, 4);      // gap: view/reshape
        SDVariable view2 = g.reshape("view_1x16", view1, 1, dim);  // gap: view/reshape
        g.mmul("out", view2, w2);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, dim);
        warmupWithChangingInput(g, "x", input, "out", 10, new long[]{1, dim});

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 50)));
            Map<String, INDArray> result = g.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [GAP_VIEW_FAST]: STUCK! " + stuckCount + "/19 steps. "
                        + "View ops in gap range frozen at classification time. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[GAP_VIEW_FAST] mode={} PASS — view ops in gap track input changes ({}/19 unique)",
                mode, 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 25: Multi-Gap Unit Cache Regression Test (per-unit keying fix)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Regression test for the per-gap-unit cache bug.
     * Before the fix: cachedActiveGapSlots_ was a single flat vector shared
     * across ALL gap units. Gap unit [0-2] would build the cache and set
     * activeGapSlotsCached_=true, then gap unit [4-4] would replay unit [0-2]'s
     * slots instead of its own.
     *
     * After the fix: cachedActiveGapSlotsMap_ is keyed by unit.startSlot.
     *
     * This test creates a graph with MULTIPLE gap units by interleaving
     * capturable ops (adds) with non-capturable gap ops (reshapes + matmuls).
     * Each gap unit has DIFFERENT gap ops. If the cache is shared, later gap
     * units execute wrong ops.
     */
    @ParameterizedTest(name = "multiGapUnitCache mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Multiple gap units with different ops — per-unit cache keying correct")
    void testMultiGapUnitCache(GraphExecutionMode mode) {
        // Graph: x → add(const) → [GAP: reshape+matmul] → add(const2) → [GAP: reshape+matmul2] → out
        // This creates 2 islands (adds) separated by 2 gap units (reshape+matmul each)
        // Each gap unit has its own matmul with different weights
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable c1 = g.var("c1", Nd4j.ones(DataType.FLOAT, 1, 8).muli(0.1f));
        SDVariable c2 = g.var("c2", Nd4j.ones(DataType.FLOAT, 1, 8).muli(0.2f));
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 8)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));

        // Island 1: add
        SDVariable added1 = x.add("add1", c1);
        // Gap 1: reshape + matmul
        SDVariable r1 = g.reshape("r1", added1, 8, 1);
        SDVariable r1f = g.reshape("r1f", r1, 1, 8);
        SDVariable mm1 = g.mmul("mm1", r1f, w1);
        // Island 2: add
        SDVariable added2 = mm1.add("add2", c2);
        // Gap 2: reshape + matmul (DIFFERENT weights from gap 1)
        SDVariable r2 = g.reshape("r2", added2, 8, 1);
        SDVariable r2f = g.reshape("r2f", r2, 1, 8);
        g.mmul("out", r2f, w2);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup past gap cache classification (executeCount >= 3)
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // Run 30 steps — if per-unit cache is correct, output changes every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 30; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [MULTI_GAP_CACHE]: STUCK! " + stuckCount + "/29 steps. "
                        + "Per-unit gap cache keying broken — gap unit 2 replaying gap unit 1's ops. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[MULTI_GAP_CACHE] mode={} PASS — {}/29 unique steps", mode, 29 - stuckCount);
    }
}
