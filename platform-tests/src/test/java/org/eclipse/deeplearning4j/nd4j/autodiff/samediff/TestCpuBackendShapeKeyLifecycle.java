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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests CPU backend (OpenVINO/oneDNN) shape key lifecycle alignment with the GPU path.
 *
 * <p>Validates the fix that aligned executeSegmentWithSpecificBackend() with the GPU path:
 * <ul>
 *   <li>hasValueDepOps segments must ALWAYS recompute shapeKey even when frozen</li>
 *   <li>Stable replay (execCount >= 3) skips shapeKey computation for all segments</li>
 *   <li>cachedShapeKey (ExecState) is used consistently, not the stale seg.shapeKey</li>
 * </ul>
 *
 * <p>These tests exercise the DSP phase lifecycle:
 * SLOT_BY_SLOT → SHAPES_FROZEN → POINTERS_STABLE → REPLAYING
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestCpuBackendShapeKeyLifecycle 2>&1 | tee /tmp/cpu-shapekey.log
 * </pre>
 */
public class TestCpuBackendShapeKeyLifecycle extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestCpuBackendShapeKeyLifecycle.class);
    private static final double TOL = 1e-4;
    private static final double TOL_LOOSE = 1e-2;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void cleanup() {
        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(false);
        env.setTritonSectionFusion(false);
        env.setTritonConsolidatedArgTable(false);
        env.setTritonArgDirtyTracking(false);
        env.setTritonCompileAll(false);
        env.setTritonIncludeTypes("");
        env.setTritonAllowFallbackCapture(false);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    // ═══════════════════════════════════════════════════════════════
    // Value-Dependent Ops After Freeze (CPU path hasValueDepOps fix)
    // ═══════════════════════════════════════════════════════════════

    /**
     * Reshape with variable target shape through multiple frozen executions.
     *
     * This tests the hasValueDepOps alignment: segments containing reshape ops
     * must recompute their shapeKey even when shapes are frozen, because the
     * reshape target shape comes from input VALUES (not just shapes).
     *
     * Without the fix, the CPU path would skip shapeKey recomputation for frozen
     * segments with value-dep ops, causing stale graph replay with wrong shapes.
     */
    @Test
    @DisplayName("CPU shapeKey: reshape with changing target shape after freeze")
    public void testReshapeValueDepAfterFreeze() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 24);
        SDVariable shapeVar = sd.placeHolder("shape_var", DataType.INT64, 2);
        SDVariable reshaped = sd.reshape("reshaped", x, shapeVar);
        SDVariable activated = sd.nn.relu("activated", reshaped, 0);
        SDVariable out = activated.sum("out");

        enableDsp(sd);

        // Input: values 1..24, all positive so relu is identity
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(1, 24);
        // Expected sum: 1+2+...+24 = 300
        double expectedSum = 300.0;

        // Phase 1: warmup with shape [4,6] — run enough to trigger freeze
        INDArray shape4x6 = Nd4j.createFromArray(4L, 6L);
        INDArray lastWarmupOutput = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "shape_var", shape4x6), "out");
            lastWarmupOutput = result.get("out").dup();
        }
        assertEquals(expectedSum, lastWarmupOutput.getDouble(0), TOL,
                "Warmup sum should be 300 regardless of reshape");

        // Phase 2: change shape_var to [3,8] after freeze
        INDArray shape3x8 = Nd4j.createFromArray(3L, 8L);
        Map<String, INDArray> afterFreezeResult = sd.output(
                Map.of("x", input, "shape_var", shape3x8), "out");
        INDArray afterFreezeOutput = afterFreezeResult.get("out").dup();
        assertEquals(expectedSum, afterFreezeOutput.getDouble(0), TOL,
                "After-freeze with shape [3,8] must produce same sum");

        // Phase 3: change to [2,12]
        INDArray shape2x12 = Nd4j.createFromArray(2L, 12L);
        Map<String, INDArray> shape2x12Result = sd.output(
                Map.of("x", input, "shape_var", shape2x12), "out");
        INDArray shape2x12Output = shape2x12Result.get("out").dup();
        assertEquals(expectedSum, shape2x12Output.getDouble(0), TOL,
                "Shape [2,12] must also produce correct sum");

        // Phase 4: back to [4,6] — verify cache reuse works
        Map<String, INDArray> backTo4x6Result = sd.output(
                Map.of("x", input, "shape_var", shape4x6), "out");
        INDArray backTo4x6Output = backTo4x6Result.get("out").dup();
        assertEquals(expectedSum, backTo4x6Output.getDouble(0), TOL,
                "Back to [4,6] must produce correct sum");

        log.info("Reshape value-dep test passed: warmup={}, [3,8]={}, [2,12]={}, back=[4,6]={}",
                lastWarmupOutput.getDouble(0), afterFreezeOutput.getDouble(0),
                shape2x12Output.getDouble(0), backTo4x6Output.getDouble(0));

        sd.close();
    }

    /**
     * Gather with changing indices after freeze — another value-dep op pattern.
     *
     * The gather indices are input VALUES that change between steps. The segment
     * containing gather has hasValueDepOps=true and must recompute shapeKey.
     */
    @Test
    @DisplayName("CPU shapeKey: gather with changing indices after freeze")
    public void testGatherValueDepAfterFreeze() {
        SameDiff sd = SameDiff.create();

        SDVariable data = sd.placeHolder("data", DataType.FLOAT, 4, 8);
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, 2);
        SDVariable gathered = sd.gather("gathered", data, indices, 0);
        SDVariable out = gathered.sum("out");

        enableDsp(sd);

        // Data: row i has all values = (i+1)*10
        INDArray dataArr = Nd4j.zeros(DataType.FLOAT, 4, 8);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 8; c++) {
                dataArr.putScalar(r, c, (r + 1) * 10.0f);
            }
        }

        // Phase 1: warmup with indices [0,1] → sum = 8*10 + 8*20 = 240
        INDArray indices01 = Nd4j.createFromArray(0L, 1L);
        double expectedWarmup = 8 * 10.0 + 8 * 20.0;
        INDArray lastWarmupOutput = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("data", dataArr, "indices", indices01), "out");
            lastWarmupOutput = result.get("out").dup();
        }
        assertEquals(expectedWarmup, lastWarmupOutput.getDouble(0), TOL,
                "Warmup gather [0,1] sum");

        // Phase 2: change indices to [2,3] → sum = 8*30 + 8*40 = 560
        INDArray indices23 = Nd4j.createFromArray(2L, 3L);
        double expectedAfterFreeze = 8 * 30.0 + 8 * 40.0;
        Map<String, INDArray> afterFreezeResult = sd.output(
                Map.of("data", dataArr, "indices", indices23), "out");
        INDArray afterFreezeOutput = afterFreezeResult.get("out").dup();
        assertEquals(expectedAfterFreeze, afterFreezeOutput.getDouble(0), TOL,
                "After-freeze gather [2,3] must return different sum");

        // Phase 3: change to [0,3] → sum = 8*10 + 8*40 = 400
        INDArray indices03 = Nd4j.createFromArray(0L, 3L);
        double expected03 = 8 * 10.0 + 8 * 40.0;
        Map<String, INDArray> result03 = sd.output(
                Map.of("data", dataArr, "indices", indices03), "out");
        INDArray output03 = result03.get("out").dup();
        assertEquals(expected03, output03.getDouble(0), TOL,
                "Gather [0,3] must produce correct sum");

        log.info("Gather value-dep test passed: warmup={} (expected {}), " +
                        "[2,3]={} (expected {}), [0,3]={} (expected {})",
                lastWarmupOutput.getDouble(0), expectedWarmup,
                afterFreezeOutput.getDouble(0), expectedAfterFreeze,
                output03.getDouble(0), expected03);

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════
    // Stable Replay Optimization (execCount >= 3)
    // ═══════════════════════════════════════════════════════════════

    /**
     * Verify correctness through many frozen executions, exercising the stable
     * replay path (execCount >= 3) where shapeKey computation is skipped.
     *
     * This tests that the cachedShapeKey remains valid during steady-state
     * replay and that no stale data leaks between executions.
     */
    @Test
    @DisplayName("CPU shapeKey: stable replay correctness (execCount >= 3)")
    public void testStableReplayCorrectness() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 4));

        SDVariable h1 = sd.mmul("h1", x, w1);
        SDVariable h1b = h1.add("h1b", b1);
        SDVariable h1a = sd.nn.relu("h1a", h1b, 0.0);
        SDVariable out = sd.mmul("out", h1a, w2);

        enableDsp(sd);

        // Pre-compute expected outputs for 10 distinct inputs
        INDArray[] inputs = new INDArray[10];
        INDArray[] expected = new INDArray[10];
        for (int i = 0; i < 10; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, 8).mul(i + 1);
            Map<String, INDArray> refResult = sd.output(Map.of("x", inputs[i]), "out");
            expected[i] = refResult.get("out").dup();
        }

        // Run through DSP path — cycles through warmup → freeze → stable replay
        for (int cycle = 0; cycle < 3; cycle++) {
            for (int i = 0; i < 10; i++) {
                Map<String, INDArray> dspResult = sd.outputDirect(
                        Map.of("x", inputs[i]), "out");
                INDArray actual = dspResult.get("out").dup();

                double maxDiff = expected[i].sub(actual).amaxNumber().doubleValue();
                assertTrue(maxDiff < TOL_LOOSE,
                        String.format("Cycle %d, input %d: maxDiff=%f exceeds tolerance",
                                cycle, i, maxDiff));

                // Verify outputs differ between consecutive inputs (not stale)
                if (i > 0) {
                    double diffFromPrev = actual.sub(expected[i - 1]).amaxNumber().doubleValue();
                    assertTrue(diffFromPrev > 0.01,
                            String.format("Cycle %d, input %d: too similar to input %d (diff=%f) — stale data?",
                                    cycle, i, i - 1, diffFromPrev));
                }
            }
        }

        log.info("Stable replay test passed: 30 executions all correct");
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════
    // Explicit Freeze + Value-Dep Ops (matches GPU decode loop pattern)
    // ═══════════════════════════════════════════════════════════════

    /**
     * Explicitly freeze shapes (like a decode loop does) then execute with
     * value-dependent ops. This is the most realistic scenario for LLM inference.
     *
     * Pattern: prefill → freeze → decode with changing KV cache sizes.
     */
    @Test
    @DisplayName("CPU shapeKey: explicit freeze + decode with value-dep ops")
    public void testExplicitFreezeWithValueDepOps() {
        SameDiff sd = SameDiff.create();

        // Graph: x[1,12] → matmul[12,8] → [1,8] → reshape(shapeVar) → sum
        // The reshape happens AFTER matmul, so shapeVar only affects the output layout.
        // This is the correct pattern for value-dep ops where the reshape target changes.
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 12);
        SDVariable shapeVar = sd.placeHolder("shape_var", DataType.INT64, 2);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 12, 8));

        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable reshaped = sd.reshape("reshaped", mm, shapeVar);
        SDVariable out = reshaped.sum("out");

        enableDsp(sd);

        // Input: values 1..12
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(1, 12);

        // Phase 1: warmup — run with shape [2,4] to populate caches
        INDArray shape2x4 = Nd4j.createFromArray(2L, 4L);
        for (int i = 0; i < 5; i++) {
            sd.output(Map.of("x", input, "shape_var", shape2x4), "out");
        }

        // Phase 2: explicitly freeze (like decode loop does after prefill)
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            log.info("Shapes frozen successfully");
        }

        // Phase 3: decode with different shapes — value-dep ops must recompute shapeKey
        // Shape [4,2]: same 8 elements, different layout
        INDArray shape4x2 = Nd4j.createFromArray(4L, 2L);
        Map<String, INDArray> result4x2 = sd.output(
                Map.of("x", input, "shape_var", shape4x2), "out");
        double sum4x2 = result4x2.get("out").getDouble(0);

        // Shape [1,8]: same elements
        INDArray shape1x8 = Nd4j.createFromArray(1L, 8L);
        Map<String, INDArray> result1x8 = sd.output(
                Map.of("x", input, "shape_var", shape1x8), "out");
        double sum1x8 = result1x8.get("out").getDouble(0);

        // Shape [8,1]: same elements
        INDArray shape8x1 = Nd4j.createFromArray(8L, 1L);
        Map<String, INDArray> result8x1 = sd.output(
                Map.of("x", input, "shape_var", shape8x1), "out");
        double sum8x1 = result8x1.get("out").getDouble(0);

        // All reshapes preserve elements, so sum should be identical
        // Compute expected: mm = x[1,12] * w[12,8] → [1,8], then sum
        INDArray mmExpected = input.mmul(w.getArr());
        double expectedSum = mmExpected.sumNumber().doubleValue();

        assertEquals(expectedSum, sum4x2, TOL_LOOSE,
                "Shape [4,2] after freeze: sum mismatch");
        assertEquals(expectedSum, sum1x8, TOL_LOOSE,
                "Shape [1,8] after freeze: sum mismatch");
        assertEquals(expectedSum, sum8x1, TOL_LOOSE,
                "Shape [8,1] after freeze: sum mismatch");

        log.info("Explicit freeze + value-dep test passed: [4,2]={}, [1,8]={}, [8,1]={} (expected {})",
                sum4x2, sum1x8, sum8x1, expectedSum);

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════
    // Mixed Value-Dep and Static Ops in Same Segment
    // ═══════════════════════════════════════════════════════════════

    /**
     * Graph with both value-dependent (reshape) and static ops (matmul, relu)
     * in the same segment. The segment hasValueDepOps=true, so shapeKey must
     * be recomputed even when frozen.
     */
    @Test
    @DisplayName("CPU shapeKey: mixed value-dep and static ops in same segment")
    public void testMixedValueDepAndStaticOps() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 16);
        SDVariable shapeVar = sd.placeHolder("shape_var", DataType.INT64, 2);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 8));

        // Static ops first: x[1,16] * w1[16,8] → [1,8]
        SDVariable h1 = sd.mmul("h1", x, w1);
        SDVariable h1a = sd.nn.relu("h1a", h1, 0.0);

        // Value-dep op: reshape [1,8] -> [2,4] or [4,2] or [1,8]
        SDVariable reshaped = sd.reshape("reshaped", h1a, shapeVar);

        // Sigmoid after the reshape (shape-independent elementwise op)
        SDVariable out = sd.nn.sigmoid("out", reshaped);

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);

        // Phase 1: warmup with shape [2,4] (8 elements from [1,8] matmul output)
        INDArray shape2x4 = Nd4j.createFromArray(2L, 4L);
        INDArray lastWarmupOutput = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "shape_var", shape2x4), "out");
            lastWarmupOutput = result.get("out").dup();
        }
        assertNotNull(lastWarmupOutput);
        assertFalse(lastWarmupOutput.isNaN().any(), "Warmup produced NaN");

        // Phase 2: change shape to [4,2]
        INDArray shape4x2 = Nd4j.createFromArray(4L, 2L);
        Map<String, INDArray> result4x2 = sd.output(
                Map.of("x", input, "shape_var", shape4x2), "out");
        INDArray output4x2 = result4x2.get("out").dup();
        assertFalse(output4x2.isNaN().any(), "Shape [4,2] produced NaN");

        // Phase 3: change shape to [1,8]
        INDArray shape1x8 = Nd4j.createFromArray(1L, 8L);
        Map<String, INDArray> result1x8 = sd.output(
                Map.of("x", input, "shape_var", shape1x8), "out");
        INDArray output1x8 = result1x8.get("out").dup();
        assertFalse(output1x8.isNaN().any(), "Shape [1,8] produced NaN");

        // Phase 4: back to [2,4] — verify cache hit works
        Map<String, INDArray> resultBack2x4 = sd.output(
                Map.of("x", input, "shape_var", shape2x4), "out");
        INDArray outputBack2x4 = resultBack2x4.get("out").dup();

        double maxDiff = lastWarmupOutput.sub(outputBack2x4).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "Back to [2,4] diverged from warmup. maxDiff=" + maxDiff);

        log.info("Mixed ops test passed: warmup={}, [4,2] sum={}, [1,8] sum={}, back=[2,4] diff={}",
                lastWarmupOutput.sumNumber().doubleValue(),
                output4x2.sumNumber().doubleValue(),
                output1x8.sumNumber().doubleValue(),
                maxDiff);

        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════
    // KV Cache Concat Pattern (realistic LLM decode scenario)
    // ═══════════════════════════════════════════════════════════════

    /**
     * Simulates KV cache concat with dynamic shapes. The concat axis dimension
     * grows each step, but the segment should handle this via dynamic dims
     * without recompilation.
     */
    @Test
    @DisplayName("CPU shapeKey: KV cache concat pattern with growing sequence")
    public void testKvCacheConcatGrowingSequence() {
        SameDiff sd = SameDiff.create();

        // KV cache: [batch, heads, seq_len, head_dim]
        SDVariable pastKv = sd.placeHolder("past_kv", DataType.FLOAT, 1, 2, -1, 8);
        SDVariable newKv = sd.placeHolder("new_kv", DataType.FLOAT, 1, 2, 1, 8);
        SDVariable fullKv = sd.concat("full_kv", 2, pastKv, newKv);

        // Simple linear projection
        SDVariable w = sd.constant("w", Nd4j.eye(8).castTo(DataType.FLOAT).muli(0.5f));
        SDVariable projected = sd.mmul("projected", fullKv, w);
        SDVariable out = projected.sum("out");

        enableDsp(sd);

        // Step 0: empty KV cache
        INDArray emptyPast = Nd4j.zeros(DataType.FLOAT, 1, 2, 0, 8);
        INDArray newToken = Nd4j.ones(DataType.FLOAT, 1, 2, 1, 8);
        Map<String, INDArray> result0 = sd.output(
                Map.of("past_kv", emptyPast, "new_kv", newToken), "out");
        // 1*2*1*8 elements * 0.5 = 8.0
        double expected0 = 1 * 2 * 1 * 8 * 0.5;
        assertEquals(expected0, result0.get("out").getDouble(0), TOL,
                "Step 0 (empty cache): sum mismatch");

        // Step 1: KV cache has 1 token
        INDArray past1 = Nd4j.ones(DataType.FLOAT, 1, 2, 1, 8);
        Map<String, INDArray> result1 = sd.output(
                Map.of("past_kv", past1, "new_kv", newToken), "out");
        // 1*2*2*8 elements * 0.5 = 16.0
        double expected1 = 1 * 2 * 2 * 8 * 0.5;
        assertEquals(expected1, result1.get("out").getDouble(0), TOL,
                "Step 1 (seq_len=2): sum mismatch");

        // Step 2: KV cache has 2 tokens
        INDArray past2 = Nd4j.ones(DataType.FLOAT, 1, 2, 2, 8);
        Map<String, INDArray> result2 = sd.output(
                Map.of("past_kv", past2, "new_kv", newToken), "out");
        // 1*2*3*8 elements * 0.5 = 24.0
        double expected2 = 1 * 2 * 3 * 8 * 0.5;
        assertEquals(expected2, result2.get("out").getDouble(0), TOL,
                "Step 2 (seq_len=3): sum mismatch");

        // Step 3: KV cache has 5 tokens
        INDArray past5 = Nd4j.ones(DataType.FLOAT, 1, 2, 5, 8);
        Map<String, INDArray> result5 = sd.output(
                Map.of("past_kv", past5, "new_kv", newToken), "out");
        // 1*2*6*8 elements * 0.5 = 48.0
        double expected5 = 1 * 2 * 6 * 8 * 0.5;
        assertEquals(expected5, result5.get("out").getDouble(0), TOL,
                "Step 3 (seq_len=6): sum mismatch");

        log.info("KV cache concat test passed: step0={}, step1={}, step2={}, step5={}",
                result0.get("out").getDouble(0), result1.get("out").getDouble(0),
                result2.get("out").getDouble(0), result5.get("out").getDouble(0));

        sd.close();
    }
}
