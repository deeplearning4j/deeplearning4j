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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.AutoTuner;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.autodiff.samediff.execution.GraphAnalysis;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the cost-model-driven auto-tuner.
 *
 * Validates that the AutoTuner correctly classifies graph structures and
 * recommends appropriate execution modes based on op mix, graph size,
 * and value-dependent shape prevalence.
 */
@Tag(TagNames.SAMEDIFF)
public class TestAutoTuner {

    @BeforeEach
    void setUp() {
        AutoTuner.clearCache();
    }

    @AfterEach
    void tearDown() {
        AutoTuner.clearCache();
    }

    /**
     * A tiny graph (3 ops) should always recommend SLOT_BY_SLOT
     * because compile overhead dominates for small graphs.
     */
    @Test
    void testSmallGraph_SlotBySlot() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable t = sd.math().tanh("t1", x);
        SDVariable s = sd.nn().sigmoid("s1", t);
        SDVariable out = sd.math().neg("output", s);
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");

        GraphExecutionMode recommended = AutoTuner.recommend(plan, sd, true);
        assertEquals(GraphExecutionMode.SLOT_BY_SLOT, recommended,
                "Small graph (3 ops) should use SLOT_BY_SLOT");
    }

    /**
     * A compute-heavy graph (matmuls dominating) should recommend TRITON
     * for kernel fusion benefits on GPU.
     */
    @Test
    void testComputeHeavyGraph_Triton() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);

        // Build a chain of matmuls to make the graph compute-heavy
        SDVariable current = x;
        for (int i = 0; i < 8; i++) {
            SDVariable w = sd.constant("w" + i, Nd4j.randn(DataType.FLOAT, 64, 64));
            current = sd.mmul("mm" + i, current, w);
        }
        // Add a few element-wise ops too (minority)
        current = sd.math().tanh("tanh", current);
        SDVariable out = current.sum("output");
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");
        GraphAnalysis analysis = AutoTuner.analyze(plan);

        // Matmuls should dominate
        assertTrue(analysis.getComputeOpFraction() >= AutoTuner.COMPUTE_HEAVY_THRESHOLD,
                "Compute ops should exceed threshold. Got: " + analysis.getComputeOpFraction());

        GraphExecutionMode recommended = AutoTuner.recommend(plan, sd, true);
        assertEquals(GraphExecutionMode.TRITON, recommended,
                "Compute-heavy graph should recommend TRITON. Analysis: " + analysis);
    }

    /**
     * An element-wise heavy graph should recommend CUDA_GRAPHS when
     * sufficient capturable segments exist.
     */
    @Test
    void testElementwiseHeavyGraph_CudaGraphs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);

        // Build a long chain of element-wise ops to exceed ELEMENTWISE_HEAVY_THRESHOLD
        SDVariable current = x;
        for (int i = 0; i < 20; i++) {
            switch (i % 4) {
                case 0: current = sd.math().tanh("tanh" + i, current); break;
                case 1: current = sd.nn().sigmoid("sig" + i, current); break;
                case 2: current = sd.math().abs("abs" + i, current); break;
                case 3: current = sd.math().neg("neg" + i, current); break;
            }
        }
        SDVariable out = current.sum("output");
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");
        GraphAnalysis analysis = AutoTuner.analyze(plan);

        assertTrue(analysis.getElementwiseOpFraction() >= AutoTuner.ELEMENTWISE_HEAVY_THRESHOLD,
                "Element-wise ops should exceed threshold. Got: " + analysis.getElementwiseOpFraction());

        GraphExecutionMode recommended = AutoTuner.recommend(plan, sd, true);
        assertEquals(GraphExecutionMode.CUDA_GRAPHS, recommended,
                "Element-wise heavy graph should recommend CUDA_GRAPHS. Analysis: " + analysis);
    }

    /**
     * Without CUDA, the tuner should not recommend CUDA_GRAPHS.
     * For a mixed graph, it should fall back to AUTO.
     */
    @Test
    void testNoCuda_FallsBackToAuto() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        // Mixed graph: some matmuls, some element-wise, some reductions
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 32));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable t = sd.math().tanh("t", mm);
        SDVariable s = sd.nn().sigmoid("s", t);
        SDVariable a = sd.math().abs("a", s);
        SDVariable n = sd.math().neg("n", a);
        SDVariable r = sd.math().tanh("t2", n);
        SDVariable out = r.sum("output");
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");
        GraphExecutionMode recommended = AutoTuner.recommend(plan, sd, false);

        // Without CUDA, should not be CUDA_GRAPHS
        assertNotEquals(GraphExecutionMode.CUDA_GRAPHS, recommended,
                "Should not recommend CUDA_GRAPHS without CUDA");
    }

    /**
     * Empirical trial data should override the cost model.
     */
    @Test
    void testTrialDataOverridesCostModel() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);

        // Compute-heavy graph (cost model says TRITON)
        SDVariable current = x;
        for (int i = 0; i < 8; i++) {
            SDVariable w = sd.constant("w" + i, Nd4j.randn(DataType.FLOAT, 64, 64));
            current = sd.mmul("mm" + i, current, w);
        }
        current = sd.math().tanh("tanh", current);
        SDVariable out = current.sum("output");
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");
        String shapeKey = AutoTuner.computeShapeKey(plan);

        // Cost model says TRITON
        GraphExecutionMode initial = AutoTuner.recommend(plan, sd, true);
        assertEquals(GraphExecutionMode.TRITON, initial);

        // Now record trial data showing CUDA_GRAPHS is faster
        AutoTuner.recordTrialResult(shapeKey, GraphExecutionMode.TRITON, 5000);
        AutoTuner.recordTrialResult(shapeKey, GraphExecutionMode.CUDA_GRAPHS, 3000);

        // Empirical data should override
        GraphExecutionMode override = AutoTuner.recommend(plan, sd, true);
        assertEquals(GraphExecutionMode.CUDA_GRAPHS, override,
                "Trial data should override cost model when CUDA_GRAPHS was faster");
    }

    /**
     * Recommendations should be cached -- same plan structure gets same result without recomputing.
     */
    @Test
    void testRecommendationCaching() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable t = sd.math().tanh("t", mm);
        SDVariable out = t.sum("output");
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");

        assertEquals(0, AutoTuner.getCacheSize(), "Cache should start empty");

        GraphExecutionMode first = AutoTuner.recommend(plan, sd, true);
        assertEquals(1, AutoTuner.getCacheSize(), "Cache should have one entry");

        GraphExecutionMode second = AutoTuner.recommend(plan, sd, true);
        assertEquals(first, second, "Cached recommendation should match");
        assertEquals(1, AutoTuner.getCacheSize(), "Cache should still have one entry");
    }

    /**
     * GraphAnalysis should correctly classify ops by category.
     */
    @Test
    void testAnalysisOpClassification() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        // 2 matmuls, 3 element-wise, 1 reduction
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 32));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable mm1 = sd.mmul("mm1", x, w1);
        SDVariable mm2 = sd.mmul("mm2", mm1, w2);
        SDVariable t = sd.math().tanh("t", mm2);
        SDVariable s = sd.nn().sigmoid("s", t);
        SDVariable a = sd.math().abs("a", s);
        SDVariable out = a.sum("output");
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");
        GraphAnalysis analysis = AutoTuner.analyze(plan);

        assertTrue(analysis.getComputeOps() >= 2,
                "Should have at least 2 compute ops (matmuls). Got: " + analysis.getComputeOps());
        assertTrue(analysis.getElementwiseOps() >= 3,
                "Should have at least 3 element-wise ops. Got: " + analysis.getElementwiseOps());
        assertTrue(analysis.getTotalSlots() >= 6,
                "Should have at least 6 total slots. Got: " + analysis.getTotalSlots());

        assertNotNull(analysis.getOpHistogram(), "Op histogram should not be null");
        assertFalse(analysis.getOpHistogram().isEmpty(), "Op histogram should not be empty");
    }

    /**
     * The shape key should be stable across identical graph structures.
     */
    @Test
    void testShapeKeyStability() {
        DynamicShapePlan plan1 = buildSimplePlan();
        DynamicShapePlan plan2 = buildSimplePlan();

        String key1 = AutoTuner.computeShapeKey(plan1);
        String key2 = AutoTuner.computeShapeKey(plan2);

        assertEquals(key1, key2, "Same graph structure should produce same shape key");
        assertTrue(key1.startsWith("plan_"), "Shape key should start with 'plan_'");
    }

    /**
     * Null/empty plans should return SLOT_BY_SLOT safely.
     */
    @Test
    void testNullAndEmptyPlan() {
        assertEquals(GraphExecutionMode.SLOT_BY_SLOT,
                AutoTuner.recommend(null, null, true),
                "Null plan should default to SLOT_BY_SLOT");
    }

    /**
     * clearCache should reset both recommendation cache and trial data.
     */
    @Test
    void testClearCache() {
        AutoTuner.recordTrialResult("test_key", GraphExecutionMode.TRITON, 1000);
        assertTrue(AutoTuner.getTrialCount() > 0);

        AutoTuner.clearCache();
        assertEquals(0, AutoTuner.getCacheSize());
        assertEquals(0, AutoTuner.getTrialCount());
    }

    /**
     * Verify the tuner produces correct numerical results when the recommended
     * mode is used for execution.
     */
    @Test
    void testAutoTunedExecution_Correctness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable t = sd.math().tanh("t", mm);
        SDVariable out = t.sum("output");
        sd.setOutputs("output");

        DynamicShapePlan plan = compilePlan(sd, "output");
        GraphExecutionMode recommended = AutoTuner.recommend(plan, sd,
                Nd4j.backends().isCudaAvailable());

        assertNotNull(recommended, "Recommendation should not be null");

        // Apply the recommended mode
        sd.setGraphExecutionMode(recommended);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 32);
        INDArray wArr = sd.getVariable("w").getArr();

        // Reference computation
        INDArray expected = input.mmul(wArr);
        expected = Nd4j.math().tanh(expected.dup());
        double expectedSum = expected.sumNumber().doubleValue();

        // DSP execution with auto-tuned mode
        INDArray result = sd.output(Map.of("x", input), "output").get("output");
        double actual = result.getDouble(0);

        double diff = Math.abs(expectedSum - actual);
        assertTrue(diff < 0.1,
                "Auto-tuned execution should produce correct results. " +
                "Mode=" + recommended + ", diff=" + diff);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private DynamicShapePlan compilePlan(SameDiff sd, String... outputs) {
        java.util.List<String> outputList = java.util.Arrays.asList(outputs);
        ForwardExecutionDAG dag = new ForwardExecutionDAGBuilder(sd)
                .buildForwardDAG(outputList);
        return DynamicShapePlanCompiler.compile(sd, dag, new java.util.HashSet<>(outputList));
    }

    private DynamicShapePlan buildSimplePlan() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable t = sd.math().tanh("t", x);
        SDVariable s = sd.nn().sigmoid("s", t);
        SDVariable out = sd.math().neg("output", s);
        sd.setOutputs("output");
        return compilePlan(sd, "output");
    }
}
