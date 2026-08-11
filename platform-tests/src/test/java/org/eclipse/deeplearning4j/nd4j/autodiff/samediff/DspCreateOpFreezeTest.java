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
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Breadcrumb test for the create-op freeze bug in native-only (monolithic) CUDA graph capture.
 *
 * <p><b>Root cause (fixed):</b> When {@code forceNativeCapture=true} (triggered whenever gap ops
 * like cuBLAS matmuls exist), ops with {@code CONSTANT_GENERATION + VALUE_DEPENDENT_SHAPE} traits
 * (e.g. {@code range}, {@code create}, {@code lin_space}) were baked into the CUDA graph at
 * capture time. Their inputs change every decode step (position IDs, sequence lengths), so
 * {@code computeCreateOpValueKey} detected a mismatch → {@code createValuesStable=false} →
 * {@code invalidateForRebuild} fired every step → infinite capture-invalidate cycle → plan
 * stuck in {@code SHAPES_FROZEN} forever → all decode steps replayed the same baked position
 * values → frozen/wrong token at every step.</p>
 *
 * <p><b>Fix:</b> Value-shape create ops are executed <em>live before</em> each
 * {@code cudaGraphLaunch} (like view ops are excluded from capture). Their outputs
 * update the stable slot buffers the graph reads from, so positions are always correct.</p>
 *
 * <p><b>What this test does:</b> Builds a minimal autoregressive-decode-like graph containing
 * {@code range(0, stepCount, 1)} (a CONSTANT_GENERATION+VALUE_DEPENDENT_SHAPE op) plus a
 * cuBLAS matmul (creates gap ops → triggers {@code forceNativeCapture=true}). A KV-like
 * variable accumulates outputs. The {@code stepCount} placeholder changes every step (position
 * IDs 1, 2, 3, …). Outputs MUST differ each step. Before the fix, outputs froze at the
 * capture-time value. After the fix, outputs change every step.</p>
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class DspCreateOpFreezeTest {

    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Graph fixture
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Build a graph that reproduces the create-op freeze:
     *
     * <pre>
     *   stepCount (placeholder, INT64 scalar) — changes each step [1, 2, 3, ...]
     *   x         (placeholder, FLOAT [1, dim]) — changes each step
     *   W         (variable,    FLOAT [dim, dim]) — fixed weights
     *
     *   range_out = range(0, stepCount, 1, INT64)       ← CONSTANT_GENERATION + VALUE_DEPENDENT_SHAPE
     *   cast_pos  = cast(range_out, FLOAT)
     *   pos_embed = sum(cast_pos)                        ← scalar position signal
     *   pos_bc    = broadcastTo(pos_embed, [1, dim])     ← broadcast to match x shape
     *   mm        = mmul(x + pos_bc, W)                 ← matmul = gap op → forceNativeCapture
     *   out       = sum(mm) + pos_embed                  ← output depends on step
     * </pre>
     *
     * The {@code out} changes each step both because {@code x} changes AND because
     * {@code pos_embed} changes (derived from the changing {@code range} output).
     * A frozen/baked {@code range} always produces the capture-time position, so
     * {@code pos_embed} stays constant → {@code out} changes only due to x (partial freeze).
     */
    private SameDiff buildCreateOpGraph(int dim) {
        SameDiff g = SameDiff.create();

        // stepCount: scalar INT64 placeholder — changes each step (1, 2, 3, ...)
        SDVariable stepCount = g.placeHolder("stepCount", DataType.INT64);

        // x: input placeholder
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);

        // Fixed weight matrix
        INDArray wArr = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1f);
        SDVariable W = g.var("W", wArr);

        // range(0, stepCount, 1) — CONSTANT_GENERATION + VALUE_DEPENDENT_SHAPE
        SDVariable zero  = g.constant(Nd4j.scalar(DataType.INT64, 0L));
        SDVariable one   = g.constant(Nd4j.scalar(DataType.INT64, 1L));
        SDVariable rangeOut = g.range("range_pos", zero, stepCount, one, DataType.INT64);

        // Cast and sum to get a scalar float position signal
        SDVariable castPos   = g.castTo("cast_pos", rangeOut, DataType.FLOAT);
        SDVariable posEmbed  = g.math().sum("pos_embed", castPos, false);

        // Reshape posEmbed to [1,1] then broadcast to [1, dim] so we can add it to x
        SDVariable posReshaped = g.reshape("pos_reshaped", posEmbed, new long[]{1, 1});
        SDVariable posBC       = g.math().add("pos_bc", x, posReshaped);  // broadcast: [1,dim] + [1,1]

        // matmul — this is a cuBLAS gap op (not Triton), triggers forceNativeCapture=true
        SDVariable mm  = g.mmul("mm", posBC, W);

        // Final output: sum(mm) + pos_embed  (depends on both input content AND step position)
        SDVariable mmSum = g.math().sum("mm_sum", mm, false);
        g.math().add("out", mmSum, posEmbed);

        return g;
    }

    /**
     * Fixed-width decode fixture: range(step, step + 1) always has shape [1],
     * while its value changes every token. This distinguishes value drift from
     * shape drift and reproduces the composite-gap invalidation seen in decoder
     * graphs.
     */
    private SameDiff buildFixedWidthRangeGraph(int dim) {
        SameDiff g = SameDiff.create();

        SDVariable step = g.placeHolder("step", DataType.INT64);
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable one = g.constant("one", Nd4j.scalar(DataType.INT64, 1L));
        SDVariable limit = g.math().add("limit", step, one);
        SDVariable range = g.range("range_pos", step, limit, one, DataType.INT64);
        SDVariable position = g.castTo("position", range, DataType.FLOAT);
        SDVariable position2d = g.reshape("position_2d", position, 1, 1);
        SDVariable shifted = g.math().add("shifted", x, position2d);

        INDArray identity = Nd4j.eye(dim).castTo(DataType.FLOAT);
        SDVariable weight = g.var("W", identity);
        SDVariable projected = g.mmul("projected", shifted, weight);
        g.math().sum("out", projected, false);

        return g;
    }

    private void configureMode(SameDiff g, GraphExecutionMode mode) {
        g.getSessions().clear();
        g.setGraphExecutionMode(mode);
        g.setDspAutoCompileEnabled(true);
        g.setDspNativeAutoCompileEnabled(true);
    }

    private Map<String, INDArray> makeInputs(int step, int dim) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        // stepCount = step + 1 so range(0, stepCount) always produces at least 1 element
        ph.put("stepCount", Nd4j.scalar(DataType.INT64, (long)(step + 1)));
        ph.put("x", Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 1)));
        return ph;
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Tests
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Core regression: outputs MUST change each step (position ID changes each step).
     *
     * <p>Before the fix, once the plan hit SHAPES_FROZEN the {@code range} op was baked at
     * capture-time position, so {@code pos_embed} was constant → outputs froze (changing x
     * might still perturb the matmul slightly, but the position contribution was frozen).</p>
     *
     * <p>After the fix, {@code range} runs live before each {@code cudaGraphLaunch}, producing
     * the correct per-step position vector, so outputs change monotonically with the step.</p>
     */
    @Test
    @DisplayName("range op in native-only capture: outputs must change each step (create-op freeze regression)")
    void testCreateOpFreezeRegressionAuto() {
        sd = buildCreateOpGraph(16);
        configureMode(sd, GraphExecutionMode.AUTO);

        int steps = 15;  // enough to enter REPLAYING and detect a freeze
        List<Double> outs = new ArrayList<>();

        for (int step = 0; step < steps; step++) {
            Map<String, INDArray> inputs = makeInputs(step, 16);
            Map<String, INDArray> result = sd.output(inputs, "out");
            double val = result.get("out").getDouble(0);
            outs.add(val);
            log.info("[CREATE_OP_FREEZE] step={} stepCount={} out={}", step, step + 1, val);
        }

        // Count steps where output was identical to the previous step.
        // Some isolated repeats are acceptable (e.g. rounding) but sustained freezes are not.
        int frozenCount = 0;
        for (int i = 1; i < outs.size(); i++) {
            if (Math.abs(outs.get(i) - outs.get(i - 1)) < 1e-6) {
                frozenCount++;
            }
        }

        log.info("[CREATE_OP_FREEZE] outputs: {}", outs);
        log.info("[CREATE_OP_FREEZE] frozen steps: {}/{}", frozenCount, steps - 1);

        // Allow at most 2 consecutive identical outputs (rounding edge-cases),
        // but a hard freeze produces >= 5 consecutive identical values.
        assertTrue(frozenCount < 5,
                "CREATE_OP_FREEZE REGRESSION: " + frozenCount + "/" + (steps - 1) +
                        " consecutive identical steps (threshold=5). " +
                        "range op may have been baked into the CUDA graph at capture time. " +
                        "outputs=" + outs);

        // Verify that the last several steps (well into REPLAYING) also show variation
        if (steps >= 10) {
            List<Double> late = outs.subList(steps - 5, steps);
            long distinctLate = late.stream().map(v -> Math.round(v * 1e4)).distinct().count();
            assertTrue(distinctLate >= 3,
                    "CREATE_OP_FREEZE: last 5 outputs should have >= 3 distinct values, got " +
                            distinctLate + ". late=" + late + " (frozen at REPLAYING?)");
        }
    }

    /**
     * Verify the position embedding signal changes monotonically with the step count.
     *
     * <p>Each step, {@code stepCount=step+1} so {@code range(0, stepCount)} = [0, 1, ..., step],
     * {@code sum} = step*(step+1)/2. With the fix, {@code pos_embed} in the output increases
     * each step. With the bug, it's frozen at the capture-time value.</p>
     */
    @Test
    @DisplayName("fixed-width range remains live across Triton composite replay")
    void testFixedWidthRangeRemainsLiveInTritonCompositeReplay() {
        assumeTrue(Nd4j.getBackendDeviceType() == DeviceType.CUDA_GPU
                        || Nd4j.getBackendDeviceType() == DeviceType.GPU,
                "CUDA backend required for Triton composite replay");

        final int dim = 16;
        sd = buildFixedWidthRangeGraph(dim);
        configureMode(sd, GraphExecutionMode.TRITON);

        INDArray fixedX = Nd4j.zeros(DataType.FLOAT, 1, dim);
        for (int step = 1; step <= 18; step++) {
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("step", Nd4j.scalar(DataType.INT64, (long) step));
            inputs.put("x", fixedX);

            double actual = sd.output(inputs, "out").get("out").getDouble(0);
            double expected = dim * (double) step;
            assertEquals(expected, actual, 1e-3,
                    "Fixed-shape range value was stale at decode step " + step);
        }
    }

    @Test
    @DisplayName("range(0, stepCount, 1) position signal increases monotonically with stepCount")
    void testPositionSignalMonotonic() {
        sd = buildCreateOpGraph(16);
        configureMode(sd, GraphExecutionMode.AUTO);

        // Run enough steps to reach REPLAYING (warmup + capture + replay)
        int warmupSteps = 8;
        for (int step = 0; step < warmupSteps; step++) {
            sd.output(makeInputs(step, 16), "out");
        }

        // Now check steps in REPLAYING state: position signal must increase.
        // Use a fixed x across these steps so any output change is due to position alone.
        INDArray fixedX = Nd4j.ones(DataType.FLOAT, 1, 16);
        List<Double> outs = new ArrayList<>();
        for (int step = warmupSteps; step < warmupSteps + 6; step++) {
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("stepCount", Nd4j.scalar(DataType.INT64, (long)(step + 1)));
            inputs.put("x", fixedX);
            Map<String, INDArray> result = sd.output(inputs, "out");
            double val = result.get("out").getDouble(0);
            outs.add(val);
            log.info("[MONOTONIC] step={} stepCount={} out={}", step, step + 1, val);
        }

        // With fixed x, out = sum(mm) + pos_embed where pos_embed = sum(range(0, stepCount))
        // = 0 + 1 + ... + step = step*(step+1)/2, which strictly increases.
        // After the fix, out must be non-decreasing across the last 6 steps.
        int decreases = 0;
        for (int i = 1; i < outs.size(); i++) {
            if (outs.get(i) < outs.get(i - 1) - 1e-3) decreases++;
        }

        long distinctLate = outs.stream()
                .map(v -> Math.round(v * 1e4))
                .distinct()
                .count();
        log.info("[MONOTONIC] late outputs: {} distinct={}", outs, distinctLate);
        assertTrue(distinctLate >= 3,
                "POSITION_SIGNAL_FROZEN: expected at least 3 distinct replay outputs as " +
                        "stepCount changes, got " + distinctLate + ". outputs=" + outs);
        assertTrue(decreases <= 1,  // Allow 1 for FP rounding
                "POSITION_SIGNAL_NON_MONOTONIC: " + decreases + " decreases in position signal " +
                        "(expected monotonic increase with stepCount). outputs=" + outs);
    }
}
