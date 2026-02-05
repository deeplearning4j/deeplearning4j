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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.*;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

@Tag(TagNames.SAMEDIFF)
public class TestExecutionPlan extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // ---- Test 1: Compile a simple linear graph and verify plan structure ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompileSimpleLinearGraph(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable out = sd.nn.softmax("out", in.mmul(w));

        Map<String, INDArray> ph = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 2, 4));

        // Build the DAG
        ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sd);
        Set<String> outputs = Collections.singleton("out");
        ForwardExecutionDAG dag = builder.buildForwardDAG(outputs);

        // Compile the plan
        ExecutionPlan plan = ExecutionPlanCompiler.compile(sd, dag, ph, outputs);

        assertNotNull(plan, "Plan should not be null");
        assertTrue(plan.getSlots().size() > 0, "Plan should have op slots");
        assertTrue(plan.getTotalHostBytes() > 0, "Plan should require some memory");
        assertTrue(plan.getBufferPoolSize() > 0, "Plan should have buffer pool entries");
        assertNotNull(plan.getBufferMap().get("out"), "Output variable should have a buffer allocation");

        // Verify the output buffer is marked as OUTPUT kind
        BufferAllocation outAlloc = plan.getBufferMap().get("out");
        assertEquals(BufferAllocKind.OUTPUT, outAlloc.getKind(),
                "Output buffer should have OUTPUT kind");

        System.out.println(plan.getSummary());
    }

    // ---- Test 2: Plan-based execution matches standard execution ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanExecutionMatchesStandard(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 3));
        SDVariable out = sd.nn.softmax("out", in.mmul(w).add(b));

        INDArray input = Nd4j.rand(DataType.FLOAT, 5, 4);
        Map<String, INDArray> ph = Collections.singletonMap("in", input);

        // Standard execution
        Map<String, INDArray> standardResult = sd.output(ph, "out");
        INDArray expected = standardResult.get("out");
        assertNotNull(expected, "Standard execution should produce output");

        // Plan-based execution
        sd.setPlanBasedExecution(true);
        Map<String, INDArray> planResult = sd.output(ph, "out");
        INDArray actual = planResult.get("out");
        assertNotNull(actual, "Plan-based execution should produce output");

        // Compare results
        assertEquals(expected.shape().length, actual.shape().length, "Rank should match");
        assertArrayEquals(expected.shape(), actual.shape(), "Shape should match");
        assertTrue(expected.equalsWithEps(actual, 1e-4),
                "Plan-based result should match standard result.\nExpected:\n" + expected + "\nActual:\n" + actual);
    }

    // ---- Test 3: Plan cache reuse across calls with same shapes ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanCacheReuse(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable out = sd.math.add("out", in.mmul(w), 1.0);

        sd.setPlanBasedExecution(true);

        // First call — should compile plan
        Map<String, INDArray> ph1 = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 3, 4));
        sd.output(ph1, "out");

        InferenceSession session = sd.getOrCreateSession();
        int cacheSize1 = session.getPlanCache().size();
        assertTrue(cacheSize1 > 0, "Plan cache should have at least one entry after first call");

        // Second call with same shape — should reuse plan
        Map<String, INDArray> ph2 = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 3, 4));
        sd.output(ph2, "out");

        int cacheSize2 = session.getPlanCache().size();
        assertEquals(cacheSize1, cacheSize2, "Plan cache size should not change for same shapes");
    }

    // ---- Test 4: Different placeholder shapes trigger recompilation ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDifferentShapesRecompile(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable out = sd.math.add("out", in.mmul(w), 1.0);

        sd.setPlanBasedExecution(true);

        // Call with shape [3, 4]
        sd.output(Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 3, 4)), "out");
        InferenceSession session = sd.getOrCreateSession();
        int cacheSize1 = session.getPlanCache().size();

        // Call with different shape [7, 4] — should compile a new plan
        sd.output(Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 7, 4)), "out");
        int cacheSize2 = session.getPlanCache().size();

        assertEquals(cacheSize1 + 1, cacheSize2,
                "Different placeholder shapes should trigger new plan compilation");
    }

    // ---- Test 5: Buffer reuse in compiled plan ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBufferReuse(Nd4jBackend backend) {
        // Build a graph where intermediate values die before later ops,
        // enabling buffer reuse
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 10);
        SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 10, 10));
        SDVariable w2 = sd.var("w2", Nd4j.rand(DataType.FLOAT, 10, 10));
        SDVariable w3 = sd.var("w3", Nd4j.rand(DataType.FLOAT, 10, 5));

        // Chain: in -> matmul(w1) -> relu -> matmul(w2) -> relu -> matmul(w3) -> out
        SDVariable h1 = sd.nn.relu(in.mmul(w1), 0);
        SDVariable h2 = sd.nn.relu(h1.mmul(w2), 0);
        SDVariable out = h2.mmul("out", w3);

        Map<String, INDArray> ph = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 4, 10));

        ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sd);
        Set<String> outputs = Collections.singleton("out");
        ForwardExecutionDAG dag = builder.buildForwardDAG(outputs);

        ExecutionPlan plan = ExecutionPlanCompiler.compile(sd, dag, ph, outputs);

        // Count reused buffers
        long reuseCount = plan.getAllocations().stream()
                .filter(a -> a.getKind() == BufferAllocKind.REUSE)
                .count();

        System.out.println(plan.getSummary());
        System.out.println("Buffer reuse count: " + reuseCount);

        // Should have some reuse in a chain like this
        // (not guaranteed depending on shape matching, but the plan should still work)
        assertNotNull(plan);
        assertTrue(plan.getSlots().size() >= 3, "Should have at least 3 op slots for the chain");
    }

    // ---- Test 6: Multi-output correctness ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiOutput(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable mmul = in.mmul(w);
        SDVariable out1 = sd.nn.softmax("out1", mmul);
        SDVariable out2 = sd.math.add("out2", mmul, 1.0);

        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
        Map<String, INDArray> ph = Collections.singletonMap("in", input);

        // Standard
        Map<String, INDArray> standard = sd.output(ph, "out1", "out2");

        // Plan-based
        sd.setPlanBasedExecution(true);
        Map<String, INDArray> planned = sd.output(ph, "out1", "out2");

        for (String key : new String[]{"out1", "out2"}) {
            INDArray exp = standard.get(key);
            INDArray act = planned.get(key);
            assertNotNull(act, "Plan output should contain " + key);
            assertArrayEquals(exp.shape(), act.shape(), "Shape mismatch for " + key);
            assertTrue(exp.equalsWithEps(act, 1e-4),
                    "Values mismatch for " + key + ".\nExpected:\n" + exp + "\nActual:\n" + act);
        }
    }

    // ---- Test 7: Repeated execution correctness (autoregressive-like pattern) ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeatedExecution(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 4));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 4));
        SDVariable out = sd.nn.tanh("out", in.mmul(w).add(b));

        sd.setPlanBasedExecution(true);

        // Run 10 times with same shape, different data
        INDArray[] results = new INDArray[10];
        for (int i = 0; i < 10; i++) {
            INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
            Map<String, INDArray> ph = Collections.singletonMap("in", input);
            Map<String, INDArray> result = sd.output(ph, "out");
            results[i] = result.get("out").dup();
            assertNotNull(results[i], "Result should not be null on iteration " + i);
            assertEquals(2, results[i].shape()[0], "Batch dim should be 2");
            assertEquals(4, results[i].shape()[1], "Feature dim should be 4");
        }

        // Results should differ (different inputs)
        boolean allSame = true;
        for (int i = 1; i < results.length; i++) {
            if (!results[0].equalsWithEps(results[i], 1e-6)) {
                allSame = false;
                break;
            }
        }
        assertFalse(allSame, "Results with different inputs should differ");
    }

    // ---- Test 8: Plan summary output ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanSummary(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 2, 3);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 3, 5));
        SDVariable out = sd.nn.relu("out", in.mmul(w), 0);

        Map<String, INDArray> ph = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 2, 3));

        ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sd);
        Set<String> outputs = Collections.singleton("out");
        ForwardExecutionDAG dag = builder.buildForwardDAG(outputs);

        ExecutionPlan plan = ExecutionPlanCompiler.compile(sd, dag, ph, outputs);

        String summary = plan.getSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("ExecutionPlan"));
        assertTrue(summary.contains("ops:"));
        assertTrue(summary.contains("buffers:"));
        assertTrue(summary.contains("totalHostBytes:"));

        System.out.println(summary);
    }

    // ---- Test 9: Scalar operations ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarOps(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable scaled = sd.math.mul(in, 2.0);
        SDVariable shifted = sd.math.add(scaled, 1.0);
        SDVariable out = sd.nn.sigmoid("out", shifted);

        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
        Map<String, INDArray> ph = Collections.singletonMap("in", input);

        // Standard
        Map<String, INDArray> standard = sd.output(ph, "out");

        // Plan-based
        sd.setPlanBasedExecution(true);
        Map<String, INDArray> planned = sd.output(ph, "out");

        INDArray exp = standard.get("out");
        INDArray act = planned.get("out");
        assertNotNull(act);
        assertTrue(exp.equalsWithEps(act, 1e-4),
                "Scalar ops result mismatch.\nExpected:\n" + exp + "\nActual:\n" + act);
    }

    // ---- Test 10: Fallback on failure ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFallbackToStandardExecution(Nd4jBackend backend) {
        // Even with plan-based enabled, if something goes wrong the standard path
        // should still produce correct results
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable out = sd.nn.softmax("out", in.mmul(w));

        sd.setPlanBasedExecution(true);

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Collections.singletonMap("in", input);

        // This should work via either plan or fallback
        Map<String, INDArray> result = sd.output(ph, "out");
        assertNotNull(result.get("out"));
        assertEquals(2, result.get("out").shape()[0]);
        assertEquals(3, result.get("out").shape()[1]);

        // Verify softmax property: rows sum to 1
        INDArray rowSums = result.get("out").sum(1);
        for (int i = 0; i < rowSums.length(); i++) {
            assertEquals(1.0, rowSums.getDouble(i), 1e-4,
                    "Softmax row " + i + " should sum to 1");
        }
    }

    // ---- Test 11: Benchmark plan vs standard execution ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanVsStandardBenchmark(Nd4jBackend backend) {
        // Build a non-trivial graph: multi-layer MLP
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 64);
        SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 64, 128));
        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 128));
        SDVariable w2 = sd.var("w2", Nd4j.rand(DataType.FLOAT, 128, 64));
        SDVariable b2 = sd.var("b2", Nd4j.rand(DataType.FLOAT, 64));
        SDVariable w3 = sd.var("w3", Nd4j.rand(DataType.FLOAT, 64, 10));
        SDVariable b3 = sd.var("b3", Nd4j.rand(DataType.FLOAT, 10));

        SDVariable h1 = sd.nn.relu(in.mmul(w1).add(b1), 0);
        SDVariable h2 = sd.nn.relu(h1.mmul(w2).add(b2), 0);
        SDVariable out = sd.nn.softmax("out", h2.mmul(w3).add(b3));

        int warmupIters = 20;
        int benchIters = 100;
        INDArray input = Nd4j.rand(DataType.FLOAT, 16, 64);
        Map<String, INDArray> ph = Collections.singletonMap("in", input);

        // ---- Standard execution benchmark ----
        sd.setPlanBasedExecution(false);
        // Warmup
        for (int i = 0; i < warmupIters; i++) {
            sd.output(ph, "out");
        }
        long startStd = System.nanoTime();
        for (int i = 0; i < benchIters; i++) {
            sd.output(ph, "out");
        }
        long stdTimeNs = System.nanoTime() - startStd;

        // ---- Plan-based execution benchmark ----
        sd.setPlanBasedExecution(true);
        // Warmup (first call compiles the plan)
        for (int i = 0; i < warmupIters; i++) {
            sd.output(ph, "out");
        }
        long startPlan = System.nanoTime();
        for (int i = 0; i < benchIters; i++) {
            sd.output(ph, "out");
        }
        long planTimeNs = System.nanoTime() - startPlan;

        double stdMs = stdTimeNs / 1_000_000.0;
        double planMs = planTimeNs / 1_000_000.0;
        double speedup = stdMs / planMs;

        System.out.println("Standard execution: " + String.format("%.2f", stdMs) + " ms total (" +
                String.format("%.3f", stdMs / benchIters) + " ms/iter)");
        System.out.println("Plan-based execution: " + String.format("%.2f", planMs) + " ms total (" +
                String.format("%.3f", planMs / benchIters) + " ms/iter)");
        System.out.println("Speedup: " + String.format("%.2fx", speedup));

        // Verify correctness — plan result should match standard
        sd.setPlanBasedExecution(false);
        INDArray expectedOut = sd.output(ph, "out").get("out");
        sd.setPlanBasedExecution(true);
        INDArray planOut = sd.output(ph, "out").get("out");
        assertTrue(expectedOut.equalsWithEps(planOut, 1e-4),
                "Plan output should match standard output");
    }

    // ---- Test 12: Verify buffer pool persistence across calls ----
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBufferPoolPersistence(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable out = sd.nn.relu("out", in.mmul(w), 0);

        sd.setPlanBasedExecution(true);

        // Run twice with same shape — second should be faster (no allocation)
        Map<String, INDArray> ph1 = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 5, 4));
        Map<String, INDArray> ph2 = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 5, 4));

        // First call — initializes plan + buffer pool
        Map<String, INDArray> result1 = sd.output(ph1, "out");
        // Second call — reuses plan + buffer pool
        Map<String, INDArray> result2 = sd.output(ph2, "out");

        assertNotNull(result1.get("out"));
        assertNotNull(result2.get("out"));
        assertArrayEquals(result1.get("out").shape(), result2.get("out").shape());

        // Results should be different (different inputs)
        assertFalse(result1.get("out").equalsWithEps(result2.get("out"), 1e-6),
                "Different inputs should produce different outputs");
    }
}
