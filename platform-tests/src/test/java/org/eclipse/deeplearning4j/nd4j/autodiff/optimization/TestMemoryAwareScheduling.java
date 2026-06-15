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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for size-weighted memory-aware scheduling.
 *
 * The scheduler uses Kahn's algorithm with a priority queue that prefers ops
 * freeing the most bytes (not just the most variables). This reduces peak
 * memory by prioritizing the release of large tensors.
 */
@Tag(TagNames.SAMEDIFF)
public class TestMemoryAwareScheduling {

    /**
     * Two independent branches consuming the same input: one produces a large
     * intermediate (matmul → 256-wide), the other produces a small intermediate
     * (bias add → 4-wide). The scheduler should prefer consuming the large
     * intermediate first to free its buffer sooner.
     *
     * Graph:
     *   x (batch, 64)
     *     → matmul(x, W_large) → large_out (batch, 256) → tanh → out_large
     *     → matmul(x, W_small) → small_out (batch, 4)   → tanh → out_small
     *   result = out_large.sum() + out_small.sum()
     *
     * With size-weighted scheduling, the large branch's tanh should be scheduled
     * before the small branch's tanh (when both are ready), freeing the 256-wide
     * buffer before the 4-wide buffer.
     */
    @Test
    void testLargeTensorFreedFirst() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);

        // Large branch: 64 → 256
        SDVariable wLarge = sd.constant("w_large", Nd4j.randn(DataType.FLOAT, 64, 256));
        SDVariable largeMm = sd.mmul("large_mm", x, wLarge);
        SDVariable largeTanh = sd.math().tanh("large_tanh", largeMm);
        SDVariable largeSum = largeTanh.sum("large_sum");

        // Small branch: 64 → 4
        SDVariable wSmall = sd.constant("w_small", Nd4j.randn(DataType.FLOAT, 64, 4));
        SDVariable smallMm = sd.mmul("small_mm", x, wSmall);
        SDVariable smallTanh = sd.math().tanh("small_tanh", smallMm);
        SDVariable smallSum = smallTanh.sum("small_sum");

        SDVariable result = largeSum.add("result", smallSum);
        sd.setOutputs("result");

        // Compile plan and inspect slot ordering
        DynamicShapePlan plan = compilePlan(sd, "result");
        DynamicShapeSlot[] slots = plan.getSlots();

        // Find the slot indices for large_tanh and small_tanh by output variable name
        int largeTanhSlot = findSlotByOutput(slots, "large_tanh");
        int smallTanhSlot = findSlotByOutput(slots, "small_tanh");

        // Both tanh ops should exist in the plan
        assertTrue(largeTanhSlot >= 0, "large_tanh should be in the plan");
        assertTrue(smallTanhSlot >= 0, "small_tanh should be in the plan");

        // Size-weighted scheduling: large_tanh (frees 256-wide buffer) should come
        // before small_tanh (frees 4-wide buffer) when both are ready.
        // Note: this is a soft ordering — dependencies may force a different order.
        // We only assert when the matmuls are independent (which they are here).
        System.out.println("large_tanh at slot " + largeTanhSlot + ", small_tanh at slot " + smallTanhSlot);
        assertTrue(largeTanhSlot < smallTanhSlot,
                "Size-weighted scheduling should place large_tanh (256-wide) before small_tanh (4-wide). " +
                "Got large_tanh at slot " + largeTanhSlot + ", small_tanh at slot " + smallTanhSlot);
    }

    /**
     * Verify that size-weighted scheduling produces correct results across all DSP modes.
     * The scheduling order must not affect numerical correctness.
     */
    @ParameterizedTest(name = "correctness_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    void testSchedulingCorrectness(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        // Two branches with different intermediate sizes
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 128));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 8));
        SDVariable mm1 = sd.mmul("mm1", x, w1);
        SDVariable mm2 = sd.mmul("mm2", x, w2);
        SDVariable t1 = sd.math().tanh("t1", mm1);
        SDVariable t2 = sd.math().tanh("t2", mm2);
        SDVariable s1 = t1.sum("s1");
        SDVariable s2 = t2.sum("s2");
        SDVariable result = s1.add("result", s2);
        sd.setOutputs("result");
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 32);

        // Compute expected: matmul + tanh + sum for each branch, then add
        INDArray w1Arr = sd.getVariable("w1").getArr();
        INDArray w2Arr = sd.getVariable("w2").getArr();
        INDArray mm1Expected = input.mmul(w1Arr);
        INDArray mm2Expected = input.mmul(w2Arr);
        INDArray t1Expected = Nd4j.math().tanh(mm1Expected.dup());
        INDArray t2Expected = Nd4j.math().tanh(mm2Expected.dup());
        double expected = t1Expected.sumNumber().doubleValue() + t2Expected.sumNumber().doubleValue();

        INDArray output = sd.output(Map.of("x", input), "result").get("result");
        double actual = output.getDouble(0);
        double diff = Math.abs(expected - actual);
        assertTrue(diff < 0.1, mode + ": scheduling correctness diff = " + diff + " (expected < 0.1)");
    }

    /**
     * Verify that the scheduler handles a linear chain correctly — no reordering
     * is possible since each op depends on the previous.
     */
    @Test
    void testLinearChain_NoReordering() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable t = sd.math().tanh("t1", x);
        SDVariable s = sd.nn().sigmoid("s1", t);
        SDVariable n = sd.math().neg("output", s);
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");
        DynamicShapeSlot[] slots = plan.getSlots();

        // In a linear chain, ordering is fixed: t1 → s1 → output
        int t1Slot = findSlotByOutput(slots, "t1");
        int s1Slot = findSlotByOutput(slots, "s1");
        int outSlot = findSlotByOutput(slots, "output");

        assertTrue(t1Slot >= 0 && s1Slot >= 0 && outSlot >= 0, "All ops should be in the plan");
        assertTrue(t1Slot < s1Slot, "t1 must come before s1");
        assertTrue(s1Slot < outSlot, "s1 must come before output");

        // Verify execution correctness
        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 64);
        INDArray expected = Nd4j.math().tanh(input.dup());
        expected = Nd4j.nn().sigmoid(expected.dup());
        expected = expected.neg();

        INDArray result = sd.output(Map.of("x", input), "output").get("output");
        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5, "Linear chain scheduling diff = " + maxDiff);
    }

    /**
     * Wide fan-out: one input feeds many independent branches of varying sizes.
     * The scheduler should process branches that free larger intermediates first.
     */
    @Test
    void testWideFanOut_LargestFirst() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        // 3 branches with decreasing output sizes: 512, 64, 8
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 512));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 32, 8));

        SDVariable mm1 = sd.mmul("mm_512", x, w1);
        SDVariable mm2 = sd.mmul("mm_64", x, w2);
        SDVariable mm3 = sd.mmul("mm_8", x, w3);

        // Each branch: tanh → sum (the tanh consumes mm output, freeing it)
        SDVariable t1 = sd.math().tanh("tanh_512", mm1);
        SDVariable t2 = sd.math().tanh("tanh_64", mm2);
        SDVariable t3 = sd.math().tanh("tanh_8", mm3);

        SDVariable s1 = t1.sum("sum_512");
        SDVariable s2 = t2.sum("sum_64");
        SDVariable s3 = t3.sum("sum_8");

        SDVariable result = s1.add("r1", s2).add("result", s3);
        sd.setOutputs("result");

        DynamicShapePlan plan = compilePlan(sd, "result");
        DynamicShapeSlot[] slots = plan.getSlots();

        // Find tanh slots by output variable name
        int tanh512 = findSlotByOutput(slots, "tanh_512");
        int tanh64 = findSlotByOutput(slots, "tanh_64");
        int tanh8 = findSlotByOutput(slots, "tanh_8");

        assertTrue(tanh512 >= 0 && tanh64 >= 0 && tanh8 >= 0,
                "All tanh ops should be in the plan");

        System.out.println("tanh_512 at slot " + tanh512 + ", tanh_64 at slot " + tanh64 +
                ", tanh_8 at slot " + tanh8);

        // Size-weighted: tanh_512 should come before tanh_64, which should come before tanh_8
        // (each tanh frees its matmul input, and 512 > 64 > 8)
        assertTrue(tanh512 < tanh64,
                "tanh_512 should be scheduled before tanh_64. Got " + tanh512 + " vs " + tanh64);
        assertTrue(tanh64 < tanh8,
                "tanh_64 should be scheduled before tanh_8. Got " + tanh64 + " vs " + tanh8);

        // Verify execution correctness
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 32);
        INDArray result1 = sd.output(Map.of("x", input), "result").get("result");
        assertNotNull(result1, "Output should not be null");
    }

    private int findSlotByOutput(DynamicShapeSlot[] slots, String varName) {
        for (int i = 0; i < slots.length; i++) {
            String[] outputNames = slots[i].getOutputVarNames();
            if (outputNames != null) {
                for (String name : outputNames) {
                    if (varName.equals(name)) return i;
                }
            }
        }
        return -1;
    }

    private DynamicShapePlan compilePlan(SameDiff sd, String... outputs) {
        java.util.List<String> outputList = java.util.Arrays.asList(outputs);
        ForwardExecutionDAG dag = new ForwardExecutionDAGBuilder(sd)
                .buildForwardDAG(outputList);
        return DynamicShapePlanCompiler.compile(sd, dag, new java.util.HashSet<>(outputList));
    }
}
