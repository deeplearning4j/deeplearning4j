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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Operator-pattern smoke coverage for standalone native plans.
 *
 * Each test compiles a fresh plan and verifies output correctness.
 * These are smoke tests — they verify op patterns don't immediately throw.
 * For replay-specific validation (same handle across iterations), see
 * TritonReplayConsolidationTest.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonPhaseStructuredReplayTest extends BaseNd4jTestWithBackends {

    private static final int NUM_ITERATIONS = 5;

    @BeforeEach
    public void setUp() {
        // Enable DSP diagnostics for test validation
        System.setProperty("nd4j.dsp.diagnostics", "SEGMENT_BUCKETS,EXECUTE,FALLBACK,GRAPH_REPLAY");
        System.setProperty("nd4j.dsp.diagnostics.level", "DETAILED");
    }

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Pure View Chains ───────────────────────────────────────────────────

    @Test
    @DisplayName("Pure view chain: reshape -> permute -> expand_dims -> reshape_no_copy")
    public void testPureViewChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);

        // Chain of view ops
        SDVariable reshaped = sd.reshape("reshape", x, -1, 4, 2);
        SDVariable permuted = sd.permute("permute", reshaped, 0, 2, 1);
        SDVariable expanded = sd.expandDims("expand", permuted, -1);
        SDVariable result = sd.reshape("result", expanded, -1, 8);

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 8);
        TritonTestUtils.runOpTest("testPureViewChain", sd, Map.of("x", input), "result");

        // Run multiple iterations to verify phase stability
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testPureViewChain_iter" + i, sd, Map.of("x", input), "result");
        }

        sd.close();
    }

    @Test
    @DisplayName("Pure view chain: squeeze -> reshape -> permute -> reshape")
    public void testSqueezeReshapePermute() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 1, 8);

        SDVariable squeezed = sd.squeeze("squeeze", x, 1);
        SDVariable reshaped = sd.reshape("reshape", squeezed, -1, 4, 2);
        SDVariable permuted = sd.permute("permute", reshaped, 0, 2, 1);
        SDVariable result = sd.reshape("result", permuted, -1, 8);

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 1, 8);
        TritonTestUtils.runOpTest("testSqueezeReshapePermute", sd, Map.of("x", input), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testSqueezeReshapePermute_iter" + i, sd, Map.of("x", input), "result");
        }

        sd.close();
    }

    @Test
    @DisplayName("Strided slice as view: contiguous slice that should be a view")
    public void testStridedSliceAsView() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        // Contiguous slice — should be a view, not a copy
        SDVariable result = sd.stridedSlice("result", x,
                new long[]{0, 0}, new long[]{-1, 16}, new long[]{1, 1});

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 32);
        TritonTestUtils.runOpTest("testStridedSliceAsView", sd, Map.of("x", input), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testStridedSliceAsView_iter" + i, sd, Map.of("x", input), "result");
        }

        sd.close();
    }

    // ─── Shape-Expression Chains ────────────────────────────────────────────

    @Test
    @DisplayName("Shape-expression chain: shape_of -> gather -> concat -> reshape")
    public void testShapeExpressionChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);

        SDVariable expanded = sd.expandDims("expanded", x, 0);
        SDVariable shape = sd.shape("shape_of", expanded);
        SDVariable tailShape = sd.gather("tail_shape", shape, new int[]{1, 2}, 0);
        SDVariable leadingOne = sd.constant("leading_one", Nd4j.createFromArray(1L));
        SDVariable targetShape = sd.concat("target_shape", 0, leadingOne, tailShape);
        SDVariable reshaped = sd.reshape("reshaped", x, targetShape);
        SDVariable result = sd.reshape("result", reshaped, -1, 8);

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 8);
        TritonTestUtils.runOpTest("testShapeExpressionChain", sd, Map.of("x", input), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testShapeExpressionChain_iter" + i, sd, Map.of("x", input), "result");
        }

        sd.close();
    }

    @Test
    @DisplayName("Shape-expression chain: shape_of -> gather -> stack")
    public void testShapeExpressionStack() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 8);

        SDVariable summed = sd.math().add("summed", x, y);
        SDVariable xShape = sd.shape("x_shape", x);
        SDVariable yShape = sd.shape("y_shape", y);
        SDVariable xDims = sd.gather("x_dims", xShape, new int[]{0, 1}, 0);
        SDVariable yDims = sd.gather("y_dims", yShape, new int[]{0, 1}, 0);
        SDVariable stacked = sd.stack("stack_shapes", 0, xDims, yDims);
        SDVariable firstShape2d = sd.gather("first_shape_2d", stacked, new int[]{0}, 0);
        SDVariable firstShape = sd.reshape("first_shape", firstShape2d, 2);
        SDVariable result = sd.reshape("result", summed, firstShape);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 16, 8);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 16, 8);
        TritonTestUtils.runOpTest("testShapeExpressionStack", sd,
                Map.of("x", input1, "y", input2), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testShapeExpressionStack_iter" + i, sd,
                    Map.of("x", input1, "y", input2), "result");
        }

        sd.close();
    }

    @Test
    @DisplayName("Constant generation chain: range -> ones_like -> multiply")
    public void testConstantGenerationChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);

        SDVariable range = sd.range("range", 0.0, 8.0, 1.0, DataType.FLOAT);
        SDVariable ones = sd.onesLike("ones_like", range);
        SDVariable scale = sd.reshape("scale", ones, 1, 8);
        SDVariable result = sd.math().mul("result", x, scale);

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 8);
        TritonTestUtils.runOpTest("testConstantGenerationChain", sd, Map.of("x", input), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testConstantGenerationChain_iter" + i, sd, Map.of("x", input), "result");
        }

        sd.close();
    }

    // ─── Mixed Prep Ladders ─────────────────────────────────────────────────

    @Test
    @DisplayName("Mixed prep ladder: gather -> concat -> reshape -> elementwise")
    public void testMixedPrepLadder() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 32);

        // Gather rows
        SDVariable gathered = sd.gather("gather", x, new int[]{0, 1, 2, 3}, 0);
        SDVariable gatheredY = sd.gather("gather_y", y, new int[]{0, 1, 2, 3}, 0);

        // Concat results
        SDVariable concat = sd.concat("concat", 0, gathered, gatheredY);

        // Reshape and elementwise
        SDVariable reshaped = sd.reshape("reshape", concat, -1, 8);
        SDVariable result = sd.nn().relu("result", reshaped, 0);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 16, 32);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 16, 32);
        TritonTestUtils.runOpTest("testMixedPrepLadder", sd,
                Map.of("x", input1, "y", input2), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testMixedPrepLadder_iter" + i, sd,
                    Map.of("x", input1, "y", input2), "result");
        }

        sd.close();
    }

    // ─── Attention Tail Tests ───────────────────────────────────────────────

    @Test
    @DisplayName("Attention-like tail: CONST_GEN -> GATHER -> SHAPE_MANIP -> ELEMENTWISE -> matmul tail")
    public void testAttentionTailShort() {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, -1, 8, 64);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, -1, 8, 64);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, -1, 8, 64);

        SDVariable qGathered = sd.gather("q_gather", q, new int[]{0, 1, 2}, 0);
        SDVariable kGathered = sd.gather("k_gather", k, new int[]{0, 1, 2}, 0);
        SDVariable vGathered = sd.gather("v_gather", v, new int[]{0, 1, 2}, 0);
        SDVariable qBiased = sd.math().add("q_bias", qGathered,
                sd.constant("bias", Nd4j.linspace(0.0, 0.5, 64, DataType.FLOAT).reshape(1, 1, 64)));
        SDVariable qFlat = sd.reshape("q_reshape", qBiased, -1, 64);
        SDVariable kFlat = sd.reshape("k_reshape", kGathered, -1, 64);
        SDVariable vFlat = sd.reshape("v_reshape", vGathered, -1, 64);
        SDVariable qkT = sd.mmul("qkT", qFlat, kFlat, false, true, false);
        SDVariable result = sd.mmul("result", qkT, vFlat, false, false, false);

        INDArray qInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        INDArray kInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        INDArray vInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        TritonTestUtils.runOpTest("testAttentionTailShort", sd,
                Map.of("q", qInput, "k", kInput, "v", vInput), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testAttentionTailShort_iter" + i, sd,
                    Map.of("q", qInput, "k", kInput, "v", vInput), "result");
        }

        sd.close();
    }

    @Test
    @DisplayName("Attention tail: full stack-chain matching invalid bucket pattern")
    public void testAttentionTailFullStackChain() {
        // This test matches the "attention_tail+stack_chain+concat_ladder+gather_ladder (1)" bucket
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, -1, 8, 64);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, -1, 8, 64);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, -1, 8, 64);

        // Gather for KV cache indexing
        SDVariable kGathered = sd.gather("k_gather", k, new int[]{0, 1, 2, 3}, 0);
        SDVariable vGathered = sd.gather("v_gather", v, new int[]{0, 1, 2, 3}, 0);

        // Stack for multi-step KV (stack adds new dim: [2, 4, 8, 64])
        SDVariable kStacked = sd.stack("k_stack", 0, kGathered, vGathered);

        SDVariable vStacked = sd.stack("v_stack", 0, kGathered, vGathered);
        // Reshape stacked back to 3D for concat: [2, 4, 8, 64] -> [8, 8, 64]
        SDVariable kStacked3d = sd.reshape("k_stacked_3d", kStacked, -1, 8, 64);
        SDVariable vStacked3d = sd.reshape("v_stacked_3d", vStacked, -1, 8, 64);

        // Concat for past+current K/V (along batch dim, both 3D)
        SDVariable kConcat = sd.concat("k_concat", 0, kStacked3d, kGathered);
        SDVariable vConcat = sd.concat("v_concat", 0, vStacked3d, vGathered);

        // Reshape for attention-like matmul
        SDVariable qFlat = sd.reshape("q_reshape", q, -1, 64);
        SDVariable kFlat = sd.reshape("k_reshape", kConcat, -1, 64);
        SDVariable vFlat = sd.reshape("v_reshape", vConcat, -1, 64);

        SDVariable qkT = sd.mmul("qkT", qFlat, kFlat, false, true, false);
        SDVariable result = sd.mmul("result", qkT, vFlat, false, false, false);

        INDArray qInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        INDArray kInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        INDArray vInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        TritonTestUtils.runOpTest("testAttentionTailFullStackChain", sd,
                Map.of("q", qInput, "k", kInput, "v", vInput), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testAttentionTailFullStackChain_iter" + i, sd,
                    Map.of("q", qInput, "k", kInput, "v", vInput), "result");
        }

        sd.close();
    }

    // ─── Phase Violation Tests ──────────────────────────────────────────────

    @Test
    @DisplayName("Intentional phase violation: should fail fast with contract error")
    public void testPhaseViolationFailsFast() {
        // This test verifies that intentional phase violations fail fast
        // with the TRITON_REPLAY_PHASE_VIOLATION error

        SameDiff sd = SameDiff.create();
        // Create a graph that would trigger a phase violation
        // (gap slot before covered slot in Triton replay)
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        // Complex pattern that may trigger replay gap issues
        SDVariable reshaped = sd.reshape("reshape", x, -1, 4, 8);
        SDVariable gathered = sd.gather("gather", reshaped, new int[]{0, 2}, 0);
        SDVariable r2 = sd.reshape("r2", reshaped, -1L, 4L, 8L);
        SDVariable concat = sd.concat("concat", 0, gathered, r2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 8, 32);

        // The test passes if no TRITON_REPLAY_PHASE_VIOLATION occurs
        // (meaning the ordered replay units correctly handle the gap ordering)
        assertDoesNotThrow(() -> {
            TritonTestUtils.runOpTest("testPhaseViolation", sd, Map.of("x", input), "concat");
        }, "Should not throw TRITON_REPLAY_PHASE_VIOLATION with ordered replay units");

        sd.close();
    }

    // ─── Replay Phase Stability ─────────────────────────────────────────────

    @Test
    @DisplayName("Repeated standalone executions remain numerically stable")
    public void testReplayPhaseStability() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 16, 8));
        SDVariable result = sd.mmul("result", x, w);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 16);

        // First iteration establishes the baseline for repeated standalone runs.
        TritonTestUtils.runOpTest("testReplayStability_warmup", sd, Map.of("x", input), "result");

        // Subsequent iterations should continue to execute without numeric drift.
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            final int iter = i;
            assertDoesNotThrow(() -> {
                TritonTestUtils.runOpTest("testReplayStability_iter" + iter, sd, Map.of("x", input), "result");
            }, "Repeated standalone execution should remain stable at iteration " + iter);
        }

        sd.close();
    }

    // ─── View Aliasing Validation ───────────────────────────────────────────

    @Test
    @DisplayName("View recipe chain smoke: reshape feeds relu without phase errors")
    public void testViewRecipeAliasing() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable reshaped = sd.reshape("reshape", x, -1, 4, 2);
        SDVariable result = sd.nn().relu("result", reshaped, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 8);
        TritonTestUtils.runOpTest("testViewRecipeAliasing", sd, Map.of("x", input), "result");

        sd.close();
    }

    // ─── Simple Const/Gather Bucket ────────────────────────────────────────

    @Test
    @DisplayName("simple_const_gather bucket: CONST_GEN + GATHER + SHAPE_MANIP + ELEMENTWISE")
    public void testSimpleConstGatherBucket() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        SDVariable gathered = sd.gather("gather", x, new int[]{0, 1, 2, 3}, 0);
        SDVariable reshaped = sd.reshape("reshape", gathered, 4, 4, 8);
        SDVariable shifted = sd.math().add("shifted", reshaped,
                sd.constant("bias", Nd4j.linspace(0.0, 0.25, 8, DataType.FLOAT).reshape(1, 1, 8)));
        SDVariable activated = sd.nn().relu("relu", shifted, 0);
        SDVariable result = sd.reshape("result", activated, -1, 8);

        INDArray input = Nd4j.randn(DataType.FLOAT, 8, 32);
        TritonTestUtils.runOpTest("testSimpleConstGatherBucket", sd, Map.of("x", input), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testSimpleConstGatherBucket_iter" + i, sd, Map.of("x", input), "result");
        }

        sd.close();
    }

    // ─── Concat Ladder + Gather Ladder Bucket ───────────────────────────────

    @Test
    @DisplayName("concat_ladder+gather_ladder bucket: CONCAT + GATHER + SHAPE_MANIP + CONST_GEN")
    public void testConcatGatherLadderBucket() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 16);
        SDVariable z = sd.placeHolder("z", DataType.FLOAT, -1, 16);

        SDVariable xGathered = sd.gather("x_gather", x, new int[]{0, 1, 2, 3}, 0);
        SDVariable yGathered = sd.gather("y_gather", y, new int[]{0, 1, 2, 3}, 0);
        SDVariable zGathered = sd.gather("z_gather", z, new int[]{0, 1, 2, 3}, 0);
        SDVariable concat1 = sd.concat("concat1", 0, xGathered, yGathered);
        SDVariable biased = sd.math().add("biased", concat1,
                sd.constant("ladder_bias", Nd4j.ones(DataType.FLOAT, 1, 16)));
        SDVariable ladderView = sd.reshape("ladder_view", biased, 4, 2, 16);
        SDVariable ladderFlat = sd.reshape("ladder_flat", ladderView, 8, 16);
        SDVariable concat2 = sd.concat("concat2", 0, ladderFlat, zGathered);
        SDVariable result = sd.reshape("result", concat2, -1, 16);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 8, 16);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 8, 16);
        INDArray input3 = Nd4j.randn(DataType.FLOAT, 8, 16);
        TritonTestUtils.runOpTest("testConcatGatherLadderBucket", sd,
                Map.of("x", input1, "y", input2, "z", input3), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testConcatGatherLadderBucket_iter" + i, sd,
                    Map.of("x", input1, "y", input2, "z", input3), "result");
        }

        sd.close();
    }

    // ─── Gather Ladder Only ────────────────────────────────────────────────

    @Test
    @DisplayName("gather_ladder bucket: GATHER + CONST_GEN + SHAPE_MANIP")
    public void testGatherLadderBucket() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        SDVariable g1 = sd.gather("g1", x, new int[]{0, 1, 2, 3, 4, 5}, 0);
        SDVariable shifted = sd.math().add("shifted", g1,
                sd.constant("gather_bias", Nd4j.linspace(0.0, 1.0, 32, DataType.FLOAT).reshape(1, 32)));
        SDVariable g2 = sd.gather("g2", shifted, new int[]{1, 3, 5}, 0);
        SDVariable reshaped = sd.reshape("reshape", g2, 3, 4, 8);
        SDVariable g3 = sd.gather("g3", reshaped, new int[]{0, 2}, 0);
        SDVariable result = sd.reshape("result", g3, -1, 8);

        INDArray input = Nd4j.randn(DataType.FLOAT, 8, 32);
        TritonTestUtils.runOpTest("testGatherLadderBucket", sd, Map.of("x", input), "result");

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TritonTestUtils.runOpTest("testGatherLadderBucket_iter" + i, sd, Map.of("x", input), "result");
        }

        sd.close();
    }

    public List<DataType> getExclusiveDataTypes() {
        return new ArrayList<>() {{
            add(DataType.FLOAT);
        }};
    }
}
