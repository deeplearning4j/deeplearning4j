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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * A DSP plan rewrites element-wise runs into fused chain heads. Run through emulated replay,
 * where the native slot executor executes the fused heads, a plan must produce exactly what
 * the same graph produces op by op without a plan: during warmup, across the freeze that runs
 * fusion again, and in the frozen steps after it. Where the operands leave the fused kernel's
 * contract at run time, the members execute one by one and the result must not change.
 */
@NativeTag
@Tag("dsp")
public class FusedElementwiseChainDspParityTest extends BaseNd4jTestWithBackends {

    private static final DataType[] STORAGE = {DataType.HALF, DataType.BFLOAT16, DataType.FLOAT, DataType.DOUBLE};

    /** Enough steps to freeze the plan and execute it frozen. */
    private static final int STEPS = 12;

    private static final double[] SPECIALS = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
            -0.0, 0.0, 1000, -1000, 1e-7};

    private boolean dynamicShapePlanWasEnabled;

    @FunctionalInterface
    private interface Graph {
        void build(SameDiff sd, SDVariable x, DataType dtype);
    }

    @BeforeEach
    public void enableDynamicShapePlan() {
        dynamicShapePlanWasEnabled = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void restoreDynamicShapePlan() {
        Nd4j.getExecutioner().commit();
        InferenceSession.setDynamicShapePlanEnabled(dynamicShapePlanWasEnabled);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    /** One run of five: declarable and legacy members, a swapped subtract. */
    @Test
    public void testMixedChain() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " mixed chain", dtype, new long[]{-1, 33}, steps(dtype, 0, STEPS, 4, 33),
                    (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        SDVariable w = ramp(sd, "w", t, 33, -1.25, 0.078125);
                        sd.math.tanh(w.sub(sd.nn.sigmoid(x.mul(y)))).div("out", y);
                    }, new int[]{1, 4}, "out");
        }
    }

    /** Twelve fusible members: a full run of eight, then a run of four. */
    @Test
    public void testLongChainSplitsIntoRuns() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " twelve members", dtype, new long[]{-1, 33}, steps(dtype, 0, STEPS, 3, 33),
                    (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        SDVariable w = ramp(sd, "w", t, 33, -1.25, 0.078125);
                        SDVariable v = sd.math.tanh(x.mul(y).add(w)).mul(y);
                        v = sd.math.sqrt(sd.math.abs(sd.nn.sigmoid(v).sub(w)));
                        sd.math.exp("out", sd.math.neg(sd.math.log1p(v.add(y))));
                    }, new int[]{2, 10}, "out");
        }
    }

    /** A requested intermediate is published, so the run ends at it. */
    @Test
    public void testRequestedIntermediateEndsRun() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " requested intermediate", dtype, new long[]{-1, 33},
                    steps(dtype, 0, STEPS, 4, 33), (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        SDVariable w = ramp(sd, "w", t, 33, -1.25, 0.078125);
                        SDVariable mid = sd.nn.sigmoid("mid", x.mul(y));
                        sd.math.tanh(mid).mul("out", w);
                    }, new int[]{2, 2}, "mid", "out");
        }
    }

    /**
     * The chain value as the second operand of subtract, divide and squaredsubtract maps to
     * the swapped code. maximum and minimum have none, so the minimum ends the run.
     */
    @Test
    public void testSwappedOperands() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " swapped operands", dtype, new long[]{-1, 33},
                    steps(dtype, 0, STEPS, 4, 33), (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        SDVariable w = ramp(sd, "w", t, 33, -1.25, 0.078125);
                        SDVariable v = y.div(w.sub(x.mul(y)));
                        v = w.squaredDifference(v.rsub(w).rdiv(y));
                        sd.math.min("out", w, sd.math.max(v, w));
                    }, new int[]{1, 6}, "out");
        }
    }

    /** clipbyvalue members; a chain carries one bounds pair, so other bounds start a new run. */
    @Test
    public void testClipMembers() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " clip", dtype, new long[]{-1, 33}, steps(dtype, 0, STEPS, 4, 33),
                    (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        SDVariable w = ramp(sd, "w", t, 33, -1.25, 0.078125);
                        sd.math.clipByValue(x.mul(y), -0.3, 0.7).add("out", w);
                    }, new int[]{1, 2}, "out");
            assertDspMatchesEager(dtype + " two clip bounds", dtype, new long[]{-1, 33}, steps(dtype, 0, STEPS, 4, 33),
                    (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        SDVariable w = ramp(sd, "w", t, 33, -1.25, 0.078125);
                        SDVariable v = sd.math.tanh(sd.math.clipByValue(x.mul(y), -0.3, 0.7));
                        sd.math.clipByValue(v, -0.5, 0.5).add("out", w);
                    }, new int[]{2, 3}, "out");
        }
    }

    /**
     * The fused relu clamps to +0. A relu with cutoff -0 returns -0 for negative inputs, which
     * tanh keeps, so it must stay out of the run.
     */
    @Test
    public void testReluCutoffs() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " relu cutoff +0", dtype, new long[]{-1, 33}, steps(dtype, 0, STEPS, 4, 33),
                    (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        sd.math.tanh("out", sd.nn.relu(x, 0.0).mul(y));
                    }, new int[]{1, 2}, "out");
            assertDspMatchesEager(dtype + " relu cutoff -0", dtype, new long[]{-1, 33}, steps(dtype, 0, STEPS, 4, 33),
                    (sd, x, t) -> {
                        SDVariable y = ramp(sd, "y", t, 33, 0.75, -0.046875);
                        sd.math.tanh("out", sd.nn.relu(x, -0.0).mul(y));
                    }, new int[]{1, 1}, "out");
        }
    }

    /** Softmax's subtract-exp run between two reductions, broadcasting the max along rows. */
    @Test
    public void testRunBetweenReductions() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " softmax", dtype, new long[]{-1, 16}, steps(dtype, 0, STEPS, 3, 16),
                    (sd, x, t) -> {
                        SDVariable e = sd.math.exp(x.sub(x.max(true, 1)));
                        e.div("out", e.sum(true, 1));
                    }, new int[]{1, 1}, "out");
        }
    }

    /** The chain value is also each member's secondary operand. */
    @Test
    public void testChainInputAsSecondary() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " input as secondary", dtype, new long[]{-1, 33},
                    steps(dtype, 0, STEPS, 4, 33), (sd, x, t) -> sd.math.tanh("out", x.mul(x).sub(x)),
                    new int[]{1, 2}, "out");
        }
    }

    /** The head reads a permuted view; the fused output is dense. */
    @Test
    public void testPermutedChainValue() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " permuted chain value", dtype, new long[]{-1, 33},
                    steps(dtype, 0, STEPS, 4, 33), (sd, x, t) -> {
                        SDVariable column = sd.constant("column",
                                Nd4j.createFromArray(values(33, 0.75, -0.046875)).castTo(t).reshape(33, 1));
                        sd.math.tanh("out", sd.permute(x, 1, 0).mul(column));
                    }, new int[]{1, 1}, "out");
        }
    }

    /**
     * Secondaries that grow the member's output past the chain value's shape: the fused kernel
     * only writes the chain value's shape, so the members run one by one.
     */
    @Test
    public void testOutputLargerThanChainValue() {
        for (DataType dtype : STORAGE) {
            assertDspMatchesEager(dtype + " [8] * [1,1,1]", dtype, new long[]{8}, steps(dtype, 0, STEPS, 8),
                    (sd, x, t) -> sd.nn.sigmoid("out",
                            x.mul(sd.constant("c", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 1.5, t)))),
                    null, "out");
            assertDspMatchesEager(dtype + " [1,8] * [3,1]", dtype, new long[]{1, 8}, steps(dtype, 0, STEPS, 1, 8),
                    (sd, x, t) -> sd.math.tanh("out",
                            x.mul(sd.constant("c", Nd4j.createFromArray(0.5, -2.0, 0.0).castTo(t).reshape(3, 1)))),
                    null, "out");
        }
    }

    /** The batch changes after the freeze. */
    @Test
    public void testChangingBatch() {
        for (DataType dtype : STORAGE) {
            List<INDArray> inputs = new ArrayList<>(steps(dtype, 0, STEPS, 3, 16));
            inputs.addAll(steps(dtype, STEPS, 4, 5, 16));
            inputs.addAll(steps(dtype, STEPS + 4, 4, 3, 16));
            assertDspMatchesEager(dtype + " changing batch", dtype, new long[]{-1, 16}, inputs, (sd, x, t) -> {
                SDVariable y = ramp(sd, "y", t, 16, 0.75, -0.09375);
                SDVariable w = ramp(sd, "w", t, 16, -1.25, 0.15625);
                sd.math.tanh(x.mul(y)).add("out", w);
            }, new int[]{1, 2}, "out");
        }
    }

    /** An empty batch after the freeze, then a non-empty one again. */
    @Test
    public void testEmptyBatchAfterFreeze() {
        for (DataType dtype : STORAGE) {
            List<INDArray> inputs = new ArrayList<>(steps(dtype, 0, STEPS, 3, 16));
            inputs.addAll(steps(dtype, STEPS, 2, 0, 16));
            inputs.addAll(steps(dtype, STEPS + 2, 3, 3, 16));
            assertDspMatchesEager(dtype + " empty batch", dtype, new long[]{-1, 16}, inputs, (sd, x, t) -> {
                SDVariable y = ramp(sd, "y", t, 16, 0.75, -0.09375);
                SDVariable w = ramp(sd, "w", t, 16, -1.25, 0.15625);
                sd.math.tanh(x.mul(y)).add("out", w);
            }, new int[]{1, 2}, "out");
        }
    }

    /**
     * Builds the graph twice: a reference that runs op by op without a plan, and a plan run
     * through emulated replay. Every step's outputs must match bit for bit.
     *
     * @param fusedRuns the expected number of fused heads and tails after the last step, or
     *                  null where the members may run one by one
     */
    private static void assertDspMatchesEager(String context, DataType dtype, long[] placeholderShape,
                                              List<INDArray> inputs, Graph graph, int[] fusedRuns,
                                              String... outputs) {
        try (SameDiff reference = SameDiff.create(); SameDiff dsp = SameDiff.create()) {
            graph.build(reference, reference.placeHolder("x", dtype, placeholderShape), dtype);
            graph.build(dsp, dsp.placeHolder("x", dtype, placeholderShape), dtype);
            reference.setDspAutoCompileEnabled(false);
            reference.setDspNativeAutoCompileEnabled(false);
            DspDebugger debugger = DspDebugger.attach(dsp);
            dsp.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);

            for (int step = 0; step < inputs.size(); step++) {
                INDArray input = inputs.get(step);
                INDArray original = input.dup();
                Map<String, INDArray> placeholders = Collections.singletonMap("x", input);
                Map<String, INDArray> expected = reference.output(placeholders, outputs);
                Map<String, INDArray> actual = dsp.output(placeholders, outputs);
                String stepContext = context + " step " + step + " " + Arrays.toString(input.shape());
                for (String name : outputs) {
                    assertNotNull(actual.get(name), stepContext + ": no " + name);
                    assertBitwise(stepContext + " " + name, expected.get(name), actual.get(name));
                }
                assertBitwise(stepContext + ": placeholder unchanged", original, input);
            }

            DynamicShapePlanExecutor eager = reference.getOrCreateSession().getDynamicShapePlanExecutor();
            assertTrue(eager == null || eager.getNativePlanHandle() == null || eager.getNativePlanHandle().isNull(),
                    context + ": the reference must run op by op, without a plan");
            DspPlanAssertions.assertPhaseReached(dsp, PlanPhase.SHAPES_FROZEN, context);
            DspPlanAssertions.assertNoPhaseContractViolations(dsp, context);
            DspPlanAssertions.assertNoSegmentFailures(dsp, context);
            DspPlanAssertions.assertNoFusionDanglingTails(dsp, context);
            if (fusedRuns != null) {
                DspDebugger.PlanReport report = debugger.analyzePlan();
                assertNull(report.errorMessage, context);
                long heads = report.slots.stream().filter(DspDebugger.SlotInfo::isFusedChainHead).count();
                long tails = report.slots.stream().filter(DspDebugger.SlotInfo::isFusedChainTail).count();
                assertEquals(fusedRuns[0], heads, context + ": fused heads\n" + report);
                assertEquals(fusedRuns[1], tails, context + ": fused tails\n" + report);
            }
        }
    }

    private static SDVariable ramp(SameDiff sd, String name, DataType dtype, int length, double start, double step) {
        return sd.constant(name, Nd4j.createFromArray(values(length, start, step)).castTo(dtype));
    }

    private static double[] values(int length, double start, double step) {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) values[i] = start + i * step;
        return values;
    }

    /** Inputs of one shape for steps first..first+count-1, with the specials at shifting positions. */
    private static List<INDArray> steps(DataType dtype, int first, int count, long... shape) {
        List<INDArray> inputs = new ArrayList<>();
        for (int step = first; step < first + count; step++) {
            long length = 1;
            for (long d : shape) length *= d;
            if (length == 0) {
                inputs.add(Nd4j.create(dtype, shape));
                continue;
            }
            double[] values = new double[(int) length];
            for (int i = 0; i < values.length; i++) values[i] = ((i * 37L + step * 11L) % 97 - 48) / 8.0;
            for (int k = 0; k < SPECIALS.length && k < values.length; k++) {
                values[(k * 5 + step) % values.length] = SPECIALS[k];
            }
            inputs.add(Nd4j.createFromArray(values).castTo(dtype).reshape(shape));
        }
        return inputs;
    }

    /** Bit-for-bit equality; every NaN matches every NaN. */
    private static void assertBitwise(String context, INDArray expected, INDArray actual) {
        assertEquals(expected.dataType(), actual.dataType(), context + ": dtype");
        assertArrayEquals(expected.shape(), actual.shape(), context + ": shape");
        if (expected.isEmpty()) {
            assertTrue(actual.isEmpty(), context + ": empty");
            return;
        }
        double[] e = expected.dup('c').data().asDouble();
        double[] a = actual.dup('c').data().asDouble();
        boolean isDouble = expected.dataType() == DataType.DOUBLE;
        StringBuilder mismatches = new StringBuilder();
        int count = 0;
        for (int i = 0; i < e.length; i++) {
            boolean same = isDouble
                    ? Double.doubleToLongBits(e[i]) == Double.doubleToLongBits(a[i])
                    : Float.floatToIntBits((float) e[i]) == Float.floatToIntBits((float) a[i]);
            if (same) continue;
            if (count++ < 8) {
                mismatches.append("\n  [").append(i).append("] expected=").append(e[i]).append(" actual=").append(a[i]);
            }
        }
        if (count > 0) fail(context + ": " + count + " of " + e.length + " elements differ" + mismatches);
    }
}
