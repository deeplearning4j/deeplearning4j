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
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP capture/replay tests for ops found in the vision encoder CUDA graph
 * that are NOT covered by existing DspCompositeReplayTest.
 *
 * Existing tests cover: gather, matmul, rmsNorm, softmax, attention, add, multiply.
 * The vision encoder (SmolDocling) ONNX graph additionally contains:
 *   shape_of, gather_nd, concat, split, zeros_like/ones_like,
 *   tile (broadcast_to), clip_by_value, where/select, reduce_sum, gt/gte/cast
 *
 * Each test embeds the target op(s) in a graph with matmul + rmsNorm so
 * that Triton compilation IS triggered and the DSP actually reaches
 * capture → freeze → replay. Without the matmul/rmsNorm backbone, the
 * graphs are too simple and stay in slot-by-slot forever.
 *
 * If any test fails at step ~3 (first replay), that op's Triton kernel
 * has a capture/replay bug (stale arg table, baked pointer, etc.).
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@DisplayName("DSP Vision Encoder Ops Replay Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DspVisionEncoderOpsReplayTest {

    private static final int STEPS = 20;
    private static final double TOL = 1e-4;
    private static final int DIM = 32;

    private final List<SameDiff> activeSds = new ArrayList<>();

    @AfterEach
    void cleanup() {
        for (SameDiff sd : activeSds) {
            try { sd.close(); } catch (Exception e) { /* ok */ }
        }
        activeSds.clear();
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();
        try {
            var nativeOps = Nd4j.getNativeOps();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
        } catch (Exception e) { /* ok on CPU */ }
    }

    private SameDiff track(SameDiff sd) {
        activeSds.add(sd);
        return sd;
    }

    @FunctionalInterface
    interface InputGenerator {
        Map<String, INDArray> generate(int step);
    }

    /**
     * Run a graph for STEPS iterations with varying inputs, compare against SLOT_BY_SLOT.
     */
    private int runAndCompare(SameDiff sd, SameDiff sdRef, GraphExecutionMode mode,
                              String outputName, InputGenerator inputGen) {
        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int matchCount = 0;
        int firstFail = -1;
        double worstDiff = 0;

        for (int i = 0; i < STEPS; i++) {
            Map<String, INDArray> inputs = inputGen.generate(i);

            Map<String, INDArray> ref = sdRef.output(inputs, outputName);
            Map<String, INDArray> actual = sd.output(inputs, outputName);

            INDArray refOut = ref.get(outputName);
            INDArray actOut = actual.get(outputName);

            assertNotNull(refOut, "ref output null at step " + i);
            assertNotNull(actOut, mode + " output null at step " + i);
            assertArrayEquals(refOut.shape(), actOut.shape(),
                    mode + " step " + i + ": shape mismatch ref=" + Arrays.toString(refOut.shape())
                            + " actual=" + Arrays.toString(actOut.shape()));

            double diff = refOut.sub(actOut).amaxNumber().doubleValue();
            if (diff < TOL) {
                matchCount++;
            } else {
                if (firstFail < 0) firstFail = i;
                log.warn("{} step {}: MISMATCH diff={}", mode, i, diff);
            }
            worstDiff = Math.max(worstDiff, diff);
        }

        int minMatch = (int) (STEPS * 0.9);
        assertTrue(matchCount >= minMatch,
                mode + ": only " + matchCount + "/" + STEPS + " steps match SLOT_BY_SLOT. "
                        + "First fail step=" + firstFail + " worstDiff=" + worstDiff);

        log.info("{}: {}/{} match, worstDiff={}", mode, matchCount, STEPS, worstDiff);
        return matchCount;
    }

    /**
     * Shared weight generation for the matmul+rmsNorm backbone.
     * Call once, pass to both sd and sdRef.
     */
    private static INDArray[] generateBackboneWeights() {
        Nd4j.getRandom().setSeed(42);
        return new INDArray[] {
            Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02),  // w1
            Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02),  // w2
            Nd4j.ones(DataType.FLOAT, DIM)                      // gamma
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  1. concat — embedded in matmul → concat → matmul → rmsNorm chain
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildConcatGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, DIM);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 1, DIM);

        // matmul each input
        SDVariable ma = sd.mmul("ma", a, sd.getVariable("w1"));
        SDVariable mb = sd.mmul("mb", b, sd.getVariable("w1"));

        // TARGET OP: concat along dim 0 → [2, DIM]
        SDVariable concat = sd.concat("concat", 0, ma, mb);

        // matmul → rmsNorm → output
        SDVariable proj = sd.mmul("proj", concat, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "1_concat_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(1)
    void test1_ConcatReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildConcatGraph(w);
        SameDiff sdRef = buildConcatGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray a = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1 * (step + 1));
            INDArray b = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1 * (step + 1));
            return Map.of("a", a, "b", b);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  2. split — matmul → split → add halves → matmul → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildSplitGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));  // [2, DIM]

        // TARGET OP: split along dim 0 → 2 x [1, DIM]
        SDVariable[] splits = sd.split(new String[]{"s0", "s1"}, m1, 2, 0);
        SDVariable combined = sd.math().add("add", splits[0], splits[1]);  // [1, DIM]

        SDVariable proj = sd.mmul("proj", combined, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "2_split_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(2)
    void test2_SplitReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildSplitGraph(w);
        SameDiff sdRef = buildSplitGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 2, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  3. zeros_like / ones_like — matmul → zerosLike → onesLike → add → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildZerosOnesLikeGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));

        // TARGET OPS: zeros_like, ones_like
        SDVariable zeros = sd.zerosLike("zeros", m1);
        SDVariable ones = sd.onesLike("ones", m1);
        SDVariable biased = sd.math().add("biased", m1, ones);  // m1 + 1
        SDVariable masked = sd.math().add("masked", biased, zeros);  // no-op but exercises the op

        SDVariable proj = sd.mmul("proj", masked, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "3_zerosOnesLike_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(3)
    void test3_ZerosOnesLikeReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildZerosOnesLikeGraph(w);
        SameDiff sdRef = buildZerosOnesLikeGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4. gather_nd — matmul → gatherNd → matmul → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildGatherNdGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());
        sd.constant("indices", Nd4j.createFromArray(new int[][]{{0}, {2}}));

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 4, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));  // [4, DIM]

        // TARGET OP: gather_nd rows 0 and 2 → [2, DIM]
        SDVariable gathered = sd.gatherNd("gathered", m1, sd.getVariable("indices"));

        SDVariable proj = sd.mmul("proj", gathered, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "4_gatherNd_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(4)
    void test4_GatherNdReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildGatherNdGraph(w);
        SameDiff sdRef = buildGatherNdGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 4, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  5. tile (broadcast_to) — matmul → tile → matmul → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildTileGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));  // [1, DIM]

        // TARGET OP: tile [1,DIM] → [4,DIM]
        SDVariable tiled = sd.tile("tiled", m1, 4, 1);

        SDVariable proj = sd.mmul("proj", tiled, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "5_tile_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(5)
    void test5_TileReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildTileGraph(w);
        SameDiff sdRef = buildTileGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  6. clipByValue — matmul → clipByValue → matmul → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildClipByValueGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));

        // TARGET OP: clipByValue
        SDVariable clipped = sd.clipByValue("clipped", m1, -1.0, 1.0);

        SDVariable proj = sd.mmul("proj", clipped, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "6_clipByValue_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(6)
    void test6_ClipByValueReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildClipByValueGraph(w);
        SameDiff sdRef = buildClipByValueGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(3.0 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  7. where/select + gte + cast — matmul → mask → where → matmul → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildWhereGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());
        sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1, DIM));

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));

        // TARGET OPS: gte → where (relu-like: keep positive, zero negative)
        SDVariable mask = sd.gte("mask", m1, 0.0);
        SDVariable selected = sd.where("selected", m1, sd.getVariable("zero"), mask);

        SDVariable proj = sd.mmul("proj", selected, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "7_where_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(7)
    void test7_WhereSelectReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildWhereGraph(w);
        SameDiff sdRef = buildWhereGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.5 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  8. gt + castTo (comparison + type cast) — matmul → gt → cast → mul → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildComparisonCastGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());
        sd.constant("threshold", Nd4j.scalar(DataType.FLOAT, 0.0));

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));

        // TARGET OPS: gt + castTo (boolean mask → float)
        SDVariable greater = sd.gt("greater", m1, sd.getVariable("threshold"));
        SDVariable greaterFloat = sd.castTo("cast", greater, DataType.FLOAT);
        SDVariable masked = sd.math().mul("masked", m1, greaterFloat);

        SDVariable proj = sd.mmul("proj", masked, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "8_comparison_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(8)
    void test8_ComparisonCastReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildComparisonCastGraph(w);
        SameDiff sdRef = buildComparisonCastGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, DIM);
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  9. reduce_sum — matmul → tile → sum → expand → matmul → rmsNorm
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildReduceSumGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 4, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));  // [4, DIM]

        // TARGET OP: reduce sum along axis 0
        SDVariable summed = sd.math().sum("summed", m1, 0);  // [DIM]
        SDVariable expanded = sd.expandDims("expanded", summed, 0);  // [1, DIM]

        SDVariable proj = sd.mmul("proj", expanded, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "9_reduceSum_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(9)
    void test9_ReduceSumReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildReduceSumGraph(w);
        SameDiff sdRef = buildReduceSumGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 4, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  10. shape_of — matmul → shape → cast → add scalar → matmul → rmsNorm
    //      (shape produces INT64, so we cast + use it as a scale factor)
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildShapeOfGraph(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));

        // TARGET OP: shape → cast to FLOAT → sum → scalar divide
        SDVariable shape = sd.shape("shape", m1);  // [2] = INT64
        SDVariable shapeFloat = sd.castTo("shape_float", shape, DataType.FLOAT);
        SDVariable shapeProd = sd.math().sum("shape_sum", shapeFloat);  // scalar
        SDVariable scaled = sd.math().div("scaled", m1, shapeProd);

        SDVariable proj = sd.mmul("proj", scaled, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "10_shapeOf_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(10)
    void test10_ShapeOfReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildShapeOfGraph(w);
        SameDiff sdRef = buildShapeOfGraph(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  11. Vision preprocess chain — combines shape, concat, split, clip,
    //      zeros/ones_like in a single graph with matmul backbone
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildVisionPreprocessChain(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", weights[0].dup());
        sd.constant("w2", weights[1].dup());
        sd.constant("gamma", weights[2].dup());

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, DIM);
        SDVariable m1 = sd.mmul("m1", input, sd.getVariable("w1"));  // [2, DIM]

        // TARGET OPS: clip → ones_like → add → concat → split → sum halves
        SDVariable clipped = sd.clipByValue("clipped", m1, -0.5, 0.5);
        SDVariable ones = sd.onesLike("ones", clipped);
        SDVariable biased = sd.math().add("biased", clipped, ones);

        SDVariable concat = sd.concat("concat", 0, biased, m1);  // [4, DIM]
        SDVariable[] splits = sd.split(new String[]{"sp0", "sp1"}, concat, 2, 0);  // 2 x [2, DIM]
        SDVariable combined = sd.math().add("combined", splits[0], splits[1]);

        SDVariable proj = sd.mmul("proj", combined, sd.getVariable("w2"));
        sd.nn().rmsNorm("out", proj, sd.getVariable("gamma"), 1e-5);
        return track(sd);
    }

    @ParameterizedTest(name = "11_visionPreprocessChain_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(11)
    void test11_VisionPreprocessChainReplay(GraphExecutionMode mode) {
        INDArray[] w = generateBackboneWeights();
        SameDiff sd = buildVisionPreprocessChain(w);
        SameDiff sdRef = buildVisionPreprocessChain(w);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 2, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  12. Vision transformer layer — mean → sub → rmsNorm → mmul → softmax
    //      This is the actual per-layer pattern in the vision encoder
    // ═══════════════════════════════════════════════════════════════════════

    private static INDArray[] generateTransformerWeights() {
        Nd4j.getRandom().setSeed(123);
        return new INDArray[] {
            Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02),  // wQ
            Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02),  // wK
            Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02),  // wV
            Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02),  // wO
            Nd4j.ones(DataType.FLOAT, DIM)                      // gamma
        };
    }

    private SameDiff buildVisionTransformerLayer(INDArray[] txWeights) {
        SameDiff sd = SameDiff.create();
        sd.constant("wQ", txWeights[0].dup());
        sd.constant("wK", txWeights[1].dup());
        sd.constant("wV", txWeights[2].dup());
        sd.constant("wO", txWeights[3].dup());
        sd.constant("gamma", txWeights[4].dup());

        // seq_len=16 patches, hidden=DIM
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16, DIM);

        // reduce_mean → subtract → rmsNorm (layernorm pattern)
        SDVariable mean = sd.math().mean("mean", input, 2);  // [1, 16]
        SDVariable meanExp = sd.expandDims("mean_exp", mean, 2);  // [1, 16, 1]
        SDVariable centered = sd.math().sub("centered", input, meanExp);
        SDVariable normed = sd.nn().rmsNorm("normed", centered, sd.getVariable("gamma"), 1e-5);

        // QKV projections (batched matmul)
        SDVariable Q = sd.mmul("Q", normed, sd.getVariable("wQ"));
        SDVariable K = sd.mmul("K", normed, sd.getVariable("wK"));
        SDVariable V = sd.mmul("V", normed, sd.getVariable("wV"));

        // Self-attention: Q @ K^T → softmax → @ V
        SDVariable Kt = sd.permute("Kt", K, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", Q, Kt);
        SDVariable scaled = sd.math().div("scaled", scores, Math.sqrt(DIM));
        SDVariable softmax = sd.nn().softmax("softmax", scaled, 2);
        SDVariable attnOut = sd.mmul("attn_out", softmax, V);

        // Output projection + residual
        SDVariable projected = sd.mmul("projected", attnOut, sd.getVariable("wO"));
        SDVariable residual = sd.math().add("residual", input, projected);
        sd.math().mean("out", residual, 1, 2);  // [1]
        return track(sd);
    }

    @ParameterizedTest(name = "12_visionTransformerLayer_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(12)
    void test12_VisionTransformerLayerReplay(GraphExecutionMode mode) {
        INDArray[] txW = generateTransformerWeights();
        SameDiff sd = buildVisionTransformerLayer(txW);
        SameDiff sdRef = buildVisionTransformerLayer(txW);
        runAndCompare(sd, sdRef, mode, "out", step -> {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16, DIM).muli(0.1 * (step + 1));
            return Map.of("input", input);
        });
    }
}
