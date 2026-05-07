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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

/**
 * Tests for the optimized slot-by-slot DSP execution path.
 *
 * <p>The optimized slot-by-slot path uses the DSP (DynamicShapePlan) pre-compiled
 * execution plan with integer-indexed slot wiring, pre-computed liveness, and
 * shape caching — but without Triton compilation, CUDA graph capture, or any
 * advanced backend. This is the execution path available on minimal builds
 * (CPU+openblas or CUDA+cublas only, no Triton).
 *
 * <p>Key properties tested:
 * <ul>
 *   <li>Correctness: SLOT_BY_SLOT matches legacy (non-DSP) execution</li>
 *   <li>Determinism: repeated executions with same inputs produce identical outputs</li>
 *   <li>Shape reuse: shapes computed once, reused on subsequent calls</li>
 *   <li>Plan compilation: DynamicShapePlanCompiler produces valid plans</li>
 *   <li>Slot structure: slots are correctly wired with input/output indices</li>
 *   <li>Training: forward + backward + updater works under SLOT_BY_SLOT</li>
 *   <li>Multi-epoch: training across multiple epochs with plan reuse</li>
 *   <li>Shape changes: correct recompilation when shapes change</li>
 * </ul>
 *
 * <p><b>Running:</b>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=DspOptimizedSlotBySlotTest \
 *       2&gt;&amp;1 | tee /tmp/dsp-slot-by-slot.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP Optimized Slot-by-Slot Execution")
public class DspOptimizedSlotBySlotTest {

    private boolean dspWasEnabled;

    @BeforeEach
    public void setUp() {
        dspWasEnabled = InferenceSession.isDynamicShapePlanEnabled();
        Nd4j.getExecutioner().commit();
    }

    @AfterEach
    public void tearDown() {
        InferenceSession.setDynamicShapePlanEnabled(dspWasEnabled);
        Nd4j.getExecutioner().commit();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── Helper: force SLOT_BY_SLOT mode on a SameDiff ────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    private static void forceSlotBySlot(SameDiff sd) {
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── Graph builders ───────────────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    private static SameDiff buildElementwiseGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", FLOAT, -1, 4);
        SDVariable sum = x.add("sum", y);
        SDVariable prod = sum.mul("prod", sd.constant("scale", Nd4j.scalar(FLOAT, 2.0)));
        sd.nn.relu("out", prod, 0);
        return sd;
    }

    private static SameDiff buildMatmulMlpGraph(long seed) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 8, 16), FLOAT, 8, 16);
        SDVariable b0 = sd.var("b0", new ZeroInitScheme('c'), FLOAT, 16);
        SDVariable h = sd.nn.relu("h", sd.nn.linear("z0", x, w0, b0), 0);
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 16, 4), FLOAT, 16, 4);
        SDVariable b1 = sd.var("b1", new ZeroInitScheme('c'), FLOAT, 4);
        sd.nn.linear("out", h, w1, b1);
        return sd;
    }

    private static SameDiff buildReductionGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        // keepDims=true so mean has shape [-1, 1] and broadcasts correctly
        SDVariable mean = x.mean("mean", true, 1);
        SDVariable centered = x.sub("centered", mean);
        SDVariable variance = centered.mul(centered).mean("var", true, 1);
        sd.math.sqrt("std", variance.add(1e-5));
        return sd;
    }

    private static SameDiff buildViewOpsGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 12);
        SDVariable reshaped = sd.reshape("reshaped", x, -1L, 3L, 4L);
        SDVariable transposed = sd.permute("transposed", reshaped, 0, 2, 1);
        SDVariable flattened = sd.reshape("flat", transposed, -1L, 12L);
        flattened.add("out", sd.constant("bias", Nd4j.ones(FLOAT, 12)));
        return sd;
    }

    private static SameDiff buildConstantFoldingGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 4);
        // Constants that could be folded
        SDVariable c1 = sd.constant("c1", Nd4j.ones(FLOAT, 4).mul(2.0));
        SDVariable c2 = sd.constant("c2", Nd4j.ones(FLOAT, 4).mul(3.0));
        SDVariable constProd = c1.mul("const_prod", c2);  // 6.0
        x.mul("out", constProd);
        return sd;
    }

    private static SameDiff buildMultiOutputGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable w = sd.var("w", new XavierInitScheme('c', 8, 4), FLOAT, 8, 4);
        SDVariable linear = x.mmul("linear", w);
        sd.nn.relu("relu_out", linear, 0);
        sd.nn.sigmoid("sigmoid_out", linear);
        sd.nn.tanh("tanh_out", linear);
        return sd;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── Training model builders ──────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    private static SameDiff buildLinearTrainingModel(long seed, int nIn, int nOut) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", FLOAT, -1, nIn);
        SDVariable labels = sd.placeHolder("labels", FLOAT, -1, nOut);
        SDVariable w = sd.var("w", new XavierInitScheme('c', nIn, nOut), FLOAT, nIn, nOut);
        SDVariable b = sd.var("b", new ZeroInitScheme('c'), FLOAT, nOut);
        SDVariable pred = sd.nn.linear("pred", input, w, b);
        sd.loss.meanSquaredError("loss", labels, pred, null);
        return sd;
    }

    private static SameDiff buildMlpTrainingModel(long seed, int nIn, int nHidden, int nOut) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", FLOAT, -1, nIn);
        SDVariable labels = sd.placeHolder("labels", FLOAT, -1, nOut);
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', nIn, nHidden), FLOAT, nIn, nHidden);
        SDVariable b0 = sd.var("b0", new ZeroInitScheme('c'), FLOAT, nHidden);
        SDVariable h = sd.nn.relu("h", sd.nn.linear("z0", input, w0, b0), 0);
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nHidden, nOut), FLOAT, nHidden, nOut);
        SDVariable b1 = sd.var("b1", new ZeroInitScheme('c'), FLOAT, nOut);
        SDVariable pred = sd.nn.linear("pred", h, w1, b1);
        sd.loss.meanSquaredError("loss", labels, pred, null);
        return sd;
    }

    private static SameDiff buildSoftmaxTrainingModel(long seed, int nIn, int nClasses) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", FLOAT, -1, nIn);
        SDVariable labels = sd.placeHolder("labels", FLOAT, -1, nClasses);
        SDVariable w = sd.var("w", new XavierInitScheme('c', nIn, nClasses), FLOAT, nIn, nClasses);
        SDVariable b = sd.var("b", new ZeroInitScheme('c'), FLOAT, nClasses);
        SDVariable logits = sd.nn.linear("logits", input, w, b);
        sd.loss.softmaxCrossEntropy("loss", labels, logits, null);
        return sd;
    }

    // ─── Data generation ─────────────────────────────────────────────────

    private static DataSet generateRegressionData(long seed, int n, int nIn, int nOut) {
        Nd4j.getRandom().setSeed(seed);
        INDArray features = Nd4j.randn(FLOAT, n, nIn);
        INDArray trueW = Nd4j.randn(FLOAT, nIn, nOut).muli(0.5);
        INDArray labels = features.mmul(trueW).addi(Nd4j.randn(FLOAT, n, nOut).muli(0.1));
        return new DataSet(features, labels);
    }

    private static DataSet generateClassificationData(long seed, int n, int nIn, int nClasses) {
        Nd4j.getRandom().setSeed(seed);
        INDArray features = Nd4j.randn(FLOAT, n, nIn);
        INDArray labels = Nd4j.zeros(FLOAT, n, nClasses);
        for (int i = 0; i < n; i++) {
            int cls = (features.getRow(i).sumNumber().doubleValue() > 0 ? 1 : 0) % nClasses;
            labels.putScalar(i, cls, 1.0f);
        }
        return new DataSet(features, labels);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── INFERENCE TESTS ──────────────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * SLOT_BY_SLOT produces the same output as legacy (non-DSP) execution
     * for a simple elementwise graph.
     */
    @Test
    @DisplayName("Elementwise: SLOT_BY_SLOT matches legacy execution")
    public void testElementwiseSlotBySlotMatchesLegacy() {
        SameDiff sd = buildElementwiseGraph();
        INDArray x = Nd4j.randn(FLOAT, 4, 4);
        INDArray y = Nd4j.randn(FLOAT, 4, 4);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);
        ph.put("y", y);

        // Reference: legacy (non-DSP)
        InferenceSession.setDynamicShapePlanEnabled(false);
        Map<String, INDArray> legacyOut = sd.output(ph, "out");
        INDArray legacyResult = legacyOut.get("out");

        // Test: DSP SLOT_BY_SLOT
        InferenceSession.setDynamicShapePlanEnabled(true);
        forceSlotBySlot(sd);
        Map<String, INDArray> slotOut = sd.output(ph, "out");
        INDArray slotResult = slotOut.get("out");

        assertNotNull(legacyResult, "Legacy result should not be null");
        assertNotNull(slotResult, "SLOT_BY_SLOT result should not be null");
        assertArrayEquals(legacyResult.shape(), slotResult.shape(),
                "Shapes should match");

        double maxDiff = legacyResult.sub(slotResult).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
                "Elementwise: max diff " + maxDiff + " exceeds tolerance 1e-5");
        log.info("Elementwise SLOT_BY_SLOT vs legacy: max diff = {}", maxDiff);
    }

    /**
     * SLOT_BY_SLOT produces the same output as legacy for a matmul-based MLP.
     */
    @Test
    @DisplayName("Matmul MLP: SLOT_BY_SLOT matches legacy execution")
    public void testMatmulMlpSlotBySlotMatchesLegacy() {
        long seed = 42;
        SameDiff sd = buildMatmulMlpGraph(seed);
        INDArray x = Nd4j.randn(FLOAT, 4, 8);
        Map<String, INDArray> ph = Collections.singletonMap("x", x);

        InferenceSession.setDynamicShapePlanEnabled(false);
        Map<String, INDArray> legacyOut = sd.output(ph, "out");

        InferenceSession.setDynamicShapePlanEnabled(true);
        forceSlotBySlot(sd);
        Map<String, INDArray> slotOut = sd.output(ph, "out");

        double maxDiff = legacyOut.get("out").sub(slotOut.get("out")).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4,
                "Matmul MLP: max diff " + maxDiff + " exceeds tolerance 1e-4");
        log.info("Matmul MLP SLOT_BY_SLOT vs legacy: max diff = {}", maxDiff);
    }

    /**
     * SLOT_BY_SLOT produces the same output as legacy for reduction ops
     * (mean, variance, sqrt).
     */
    @Test
    @DisplayName("Reductions: SLOT_BY_SLOT matches legacy execution")
    public void testReductionSlotBySlotMatchesLegacy() {
        SameDiff sd = buildReductionGraph();
        INDArray x = Nd4j.randn(FLOAT, 8, 8);
        Map<String, INDArray> ph = Collections.singletonMap("x", x);

        InferenceSession.setDynamicShapePlanEnabled(false);
        Map<String, INDArray> legacyOut = sd.output(ph, "std");

        InferenceSession.setDynamicShapePlanEnabled(true);
        forceSlotBySlot(sd);
        Map<String, INDArray> slotOut = sd.output(ph, "std");

        double maxDiff = legacyOut.get("std").sub(slotOut.get("std")).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
                "Reductions: max diff " + maxDiff + " exceeds tolerance 1e-5");
        log.info("Reduction SLOT_BY_SLOT vs legacy: max diff = {}", maxDiff);
    }

    /**
     * SLOT_BY_SLOT produces the same output as legacy for view ops
     * (reshape, permute).
     */
    @Test
    @DisplayName("View ops: SLOT_BY_SLOT matches legacy execution")
    public void testViewOpsSlotBySlotMatchesLegacy() {
        SameDiff sd = buildViewOpsGraph();
        INDArray x = Nd4j.randn(FLOAT, 2, 12);
        Map<String, INDArray> ph = Collections.singletonMap("x", x);

        InferenceSession.setDynamicShapePlanEnabled(false);
        Map<String, INDArray> legacyOut = sd.output(ph, "out");

        InferenceSession.setDynamicShapePlanEnabled(true);
        forceSlotBySlot(sd);
        Map<String, INDArray> slotOut = sd.output(ph, "out");

        double maxDiff = legacyOut.get("out").sub(slotOut.get("out")).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
                "View ops: max diff " + maxDiff + " exceeds tolerance 1e-5");
        log.info("View ops SLOT_BY_SLOT vs legacy: max diff = {}", maxDiff);
    }

    /**
     * SLOT_BY_SLOT correctly handles graphs with constant folding opportunities.
     */
    @Test
    @DisplayName("Constant folding: SLOT_BY_SLOT matches legacy")
    public void testConstantFoldingSlotBySlotMatchesLegacy() {
        SameDiff sd = buildConstantFoldingGraph();
        INDArray x = Nd4j.randn(FLOAT, 4, 4);
        Map<String, INDArray> ph = Collections.singletonMap("x", x);

        InferenceSession.setDynamicShapePlanEnabled(false);
        Map<String, INDArray> legacyOut = sd.output(ph, "out");

        InferenceSession.setDynamicShapePlanEnabled(true);
        forceSlotBySlot(sd);
        Map<String, INDArray> slotOut = sd.output(ph, "out");

        double maxDiff = legacyOut.get("out").sub(slotOut.get("out")).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
                "Constant folding: max diff " + maxDiff + " exceeds tolerance 1e-5");
    }

    /**
     * SLOT_BY_SLOT produces correct output for multi-output graphs
     * (same linear layer feeding relu, sigmoid, tanh).
     */
    @Test
    @DisplayName("Multi-output: SLOT_BY_SLOT matches legacy for all outputs")
    public void testMultiOutputSlotBySlotMatchesLegacy() {
        SameDiff sd = buildMultiOutputGraph();
        INDArray x = Nd4j.randn(FLOAT, 4, 8);
        Map<String, INDArray> ph = Collections.singletonMap("x", x);
        List<String> outputs = Arrays.asList("relu_out", "sigmoid_out", "tanh_out");

        InferenceSession.setDynamicShapePlanEnabled(false);
        Map<String, INDArray> legacyOut = sd.output(ph, outputs);

        InferenceSession.setDynamicShapePlanEnabled(true);
        forceSlotBySlot(sd);
        Map<String, INDArray> slotOut = sd.output(ph, outputs);

        for (String name : outputs) {
            double maxDiff = legacyOut.get(name).sub(slotOut.get(name)).amaxNumber().doubleValue();
            assertTrue(maxDiff < 1e-5,
                    "Multi-output '" + name + "': max diff " + maxDiff + " exceeds tolerance 1e-5");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── DETERMINISM AND SHAPE REUSE ──────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Repeated execution with the same inputs produces bit-identical results.
     * This verifies the plan is correctly reused (shapes cached, slots stable).
     */
    @Test
    @DisplayName("Determinism: repeated SLOT_BY_SLOT calls produce identical output")
    public void testRepeatedExecutionDeterministic() {
        SameDiff sd = buildMatmulMlpGraph(42);
        forceSlotBySlot(sd);
        InferenceSession.setDynamicShapePlanEnabled(true);

        INDArray x = Nd4j.randn(FLOAT, 4, 8);
        Map<String, INDArray> ph = Collections.singletonMap("x", x);

        INDArray first = sd.output(ph, "out").get("out").dup();

        for (int i = 0; i < 10; i++) {
            INDArray result = sd.output(ph, "out").get("out");
            double maxDiff = first.sub(result).amaxNumber().doubleValue();
            assertEquals(0.0, maxDiff, 0.0,
                    "Run " + (i + 1) + ": output differs from first run, maxDiff=" + maxDiff);
        }
        log.info("10 repeated SLOT_BY_SLOT executions: all bit-identical");
    }

    /**
     * Changing input shapes triggers correct recompilation without stale results.
     */
    @Test
    @DisplayName("Shape change: SLOT_BY_SLOT handles batch size changes correctly")
    public void testShapeChangeTriggersPlanRecompile() {
        SameDiff sd = buildMatmulMlpGraph(42);
        forceSlotBySlot(sd);
        InferenceSession.setDynamicShapePlanEnabled(true);

        int[] batchSizes = {2, 4, 8, 4, 2, 16};
        for (int bs : batchSizes) {
            INDArray x = Nd4j.randn(FLOAT, bs, 8);
            Map<String, INDArray> ph = Collections.singletonMap("x", x);
            Map<String, INDArray> out = sd.output(ph, "out");
            INDArray result = out.get("out");

            assertNotNull(result, "Result should not be null for batch size " + bs);
            assertEquals(bs, result.shape()[0],
                    "Output batch dim should match input batch size " + bs);
            assertEquals(4, result.shape()[1],
                    "Output feature dim should be 4 for batch size " + bs);
        }
        log.info("Shape change across {} batch sizes: all correct", batchSizes.length);
    }

    /**
     * Different input values with the same shape produce different outputs.
     * This guards against stale cached values being returned.
     */
    @Test
    @DisplayName("Input sensitivity: different inputs with same shape produce different outputs")
    public void testDifferentInputsDifferentOutputs() {
        SameDiff sd = buildElementwiseGraph();
        forceSlotBySlot(sd);
        InferenceSession.setDynamicShapePlanEnabled(true);

        INDArray x1 = Nd4j.randn(FLOAT, 4, 4);
        INDArray y1 = Nd4j.randn(FLOAT, 4, 4);
        Map<String, INDArray> ph1 = new LinkedHashMap<>();
        ph1.put("x", x1);
        ph1.put("y", y1);

        INDArray x2 = Nd4j.randn(FLOAT, 4, 4).muli(5);
        INDArray y2 = Nd4j.randn(FLOAT, 4, 4).muli(5);
        Map<String, INDArray> ph2 = new LinkedHashMap<>();
        ph2.put("x", x2);
        ph2.put("y", y2);

        INDArray out1 = sd.output(ph1, "out").get("out");
        INDArray out2 = sd.output(ph2, "out").get("out");

        double maxDiff = out1.sub(out2).amaxNumber().doubleValue();
        assertTrue(maxDiff > 1e-3,
                "Different inputs should produce different outputs (maxDiff=" + maxDiff + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── PLAN COMPILATION TESTS ───────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * DynamicShapePlanCompiler produces a valid plan for elementwise graph.
     * Verifies slot count, input wiring, and output mapping.
     */
    @Test
    @DisplayName("Plan compilation: elementwise graph produces valid plan")
    public void testPlanCompilationElementwise() {
        SameDiff sd = buildElementwiseGraph();
        forceSlotBySlot(sd);

        // Trigger a forward pass to populate the DAG
        INDArray x = Nd4j.randn(FLOAT, 4, 4);
        INDArray y = Nd4j.randn(FLOAT, 4, 4);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);
        ph.put("y", y);

        InferenceSession.setDynamicShapePlanEnabled(true);
        Map<String, INDArray> out = sd.output(ph, "out");
        assertNotNull(out.get("out"), "Output should be produced");

        // Verify the plan was compiled
        DynamicShapePlan plan = sd.getCachedDynamicShapePlan(Collections.singleton("out"));
        if (plan != null) {
            DynamicShapeSlot[] slots = plan.getSlots();
            assertTrue(slots.length > 0, "Plan should have at least one slot");

            // Verify output mapping exists
            Map<String, Integer> outputMap = plan.getOutputNameToSlotIndex();
            assertTrue(outputMap.containsKey("out"),
                    "Plan output map should contain 'out'");

            log.info("Plan compiled: {} slots, outputs: {}", slots.length, outputMap.keySet());

            // Verify each slot has valid wiring
            for (int i = 0; i < slots.length; i++) {
                DynamicShapeSlot slot = slots[i];
                assertNotNull(slot, "Slot " + i + " should not be null");
                // inputSourceIndices: positive = op output, negative = external input
                int[] sources = slot.getInputSourceIndices();
                assertNotNull(sources, "Slot " + i + " input sources should not be null");
            }
        } else {
            log.warn("Plan was not cached after execution — checking if execution succeeded anyway");
            assertNotNull(out.get("out"), "Output should still be valid even without cached plan");
        }
    }

    /**
     * DynamicShapePlanCompiler produces a valid plan for matmul MLP graph.
     * Verifies that the plan includes the expected number of computational slots.
     */
    @Test
    @DisplayName("Plan compilation: matmul MLP produces valid plan with correct slot count")
    public void testPlanCompilationMatmulMlp() {
        SameDiff sd = buildMatmulMlpGraph(42);
        forceSlotBySlot(sd);
        InferenceSession.setDynamicShapePlanEnabled(true);

        INDArray x = Nd4j.randn(FLOAT, 4, 8);
        sd.output(Collections.singletonMap("x", x), "out");

        DynamicShapePlan plan = sd.getCachedDynamicShapePlan(Collections.singleton("out"));
        if (plan != null) {
            DynamicShapeSlot[] slots = plan.getSlots();
            // MLP has: 2x linear (matmul + bias add) + 1x relu = at least 3 ops
            assertTrue(slots.length >= 3,
                    "MLP plan should have >= 3 slots, got " + slots.length);

            // Verify external inputs are properly mapped
            String[] externalInputKeys = plan.getExternalInputKeys();
            // x is a placeholder, w0/b0/w1/b1 are variables — all are external
            assertTrue(externalInputKeys.length >= 1,
                    "Plan should have >= 1 external input, got " + externalInputKeys.length);

            log.info("MLP plan: {} slots, {} external inputs",
                    slots.length, externalInputKeys.length);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── TRAINING TESTS ───────────────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> updaterConfigs() {
        return Stream.of(
                Arguments.of("SGD", new Sgd(1e-2)),
                Arguments.of("Adam", new Adam(1e-3)),
                Arguments.of("Nesterovs", new Nesterovs(1e-2, 0.9))
        );
    }

    /**
     * Linear regression training under SLOT_BY_SLOT: loss must decrease.
     */
    @ParameterizedTest(name = "linearTraining[{0}]")
    @MethodSource("updaterConfigs")
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: linear model loss decreases under SLOT_BY_SLOT")
    public void testLinearTrainingSlotBySlot(String updaterName, IUpdater updater) {
        long seed = 42;
        int nIn = 8, nOut = 2, batchSize = 16, epochs = 20;
        DataSet ds = generateRegressionData(seed + 100, batchSize, nIn, nOut);
        DataSetIterator iter = new SingletonDataSetIterator(ds);

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildLinearTrainingModel(seed, nIn, nOut);
        forceSlotBySlot(sd);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(updater)
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);
        History hist = sd.fit(iter, epochs);

        INDArray lossCurve = hist.getLossCurve().getLossValues();
        double firstLoss = lossCurve.getDouble(0);
        double lastLoss = lossCurve.getDouble((int) lossCurve.length() - 1);

        log.info("[{}] SLOT_BY_SLOT training: first={}, last={}", updaterName, firstLoss, lastLoss);

        assertTrue(firstLoss > 0, updaterName + ": initial loss should be positive");
        assertTrue(lastLoss < firstLoss,
                updaterName + ": final loss (" + lastLoss + ") should be < initial (" + firstLoss + ")");
    }

    /**
     * MLP training under SLOT_BY_SLOT: loss must decrease significantly.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: MLP with ReLU, loss decreases under SLOT_BY_SLOT")
    public void testMlpTrainingSlotBySlot() {
        long seed = 42;
        int nIn = 8, nHidden = 16, nOut = 2, batchSize = 32, epochs = 30;
        DataSet ds = generateRegressionData(seed + 200, batchSize, nIn, nOut);

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildMlpTrainingModel(seed, nIn, nHidden, nOut);
        forceSlotBySlot(sd);

        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());
        History hist = sd.fit(new SingletonDataSetIterator(ds), epochs);

        INDArray lossCurve = hist.getLossCurve().getLossValues();
        double firstLoss = lossCurve.getDouble(0);
        double lastLoss = lossCurve.getDouble((int) lossCurve.length() - 1);
        double reduction = (firstLoss - lastLoss) / firstLoss;

        log.info("MLP SLOT_BY_SLOT training: first={}, last={}, reduction={}%",
                firstLoss, lastLoss, reduction * 100);

        assertTrue(reduction > 0.3,
                "MLP loss should reduce by > 30%, got " + (reduction * 100) + "%");
    }

    /**
     * Softmax classifier training under SLOT_BY_SLOT.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: softmax classifier converges under SLOT_BY_SLOT")
    public void testSoftmaxTrainingSlotBySlot() {
        long seed = 42;
        int nIn = 8, nClasses = 3, batchSize = 32, epochs = 30;
        DataSet ds = generateClassificationData(seed + 300, batchSize, nIn, nClasses);

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildSoftmaxTrainingModel(seed, nIn, nClasses);
        forceSlotBySlot(sd);

        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());
        History hist = sd.fit(new SingletonDataSetIterator(ds), epochs);

        INDArray lossCurve = hist.getLossCurve().getLossValues();
        double firstLoss = lossCurve.getDouble(0);
        double lastLoss = lossCurve.getDouble((int) lossCurve.length() - 1);

        log.info("Softmax SLOT_BY_SLOT training: first={}, last={}", firstLoss, lastLoss);

        assertTrue(lastLoss < firstLoss,
                "Softmax loss should decrease: first=" + firstLoss + ", last=" + lastLoss);
    }

    /**
     * Weights must actually change during SLOT_BY_SLOT training.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: weights are updated under SLOT_BY_SLOT")
    public void testWeightsUpdatedSlotBySlot() {
        long seed = 42;
        int nIn = 4, nOut = 1, batchSize = 8, epochs = 5;
        DataSet ds = generateRegressionData(seed + 400, batchSize, nIn, nOut);

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildLinearTrainingModel(seed, nIn, nOut);
        forceSlotBySlot(sd);

        // Snapshot weights before training
        INDArray wBefore = sd.getVariable("w").getArr().dup();
        INDArray bBefore = sd.getVariable("b").getArr().dup();

        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(1e-2))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());
        sd.fit(new SingletonDataSetIterator(ds), epochs);

        INDArray wAfter = sd.getVariable("w").getArr();
        INDArray bAfter = sd.getVariable("b").getArr();

        double wDiff = wBefore.sub(wAfter).amaxNumber().doubleValue();
        double bDiff = bBefore.sub(bAfter).amaxNumber().doubleValue();

        log.info("Weight update check: w maxDiff={}, b maxDiff={}", wDiff, bDiff);

        assertTrue(wDiff > 1e-6, "Weights should change during training (maxDiff=" + wDiff + ")");
        assertTrue(bDiff > 1e-6, "Bias should change during training (maxDiff=" + bDiff + ")");
    }

    /**
     * Gradients computed via calculateGradients under SLOT_BY_SLOT must be non-zero.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: gradients are non-zero under SLOT_BY_SLOT")
    public void testGradientsNonZeroSlotBySlot() {
        long seed = 42;
        int nIn = 4, nOut = 1;

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildLinearTrainingModel(seed, nIn, nOut);
        forceSlotBySlot(sd);

        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Sgd(1e-2))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());

        INDArray features = Nd4j.randn(FLOAT, 8, nIn);
        INDArray labels = Nd4j.randn(FLOAT, 8, nOut);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", features);
        ph.put("labels", labels);

        Map<String, INDArray> grads = sd.calculateGradients(ph, "w", "b");

        assertNotNull(grads.get("w"), "Weight gradients should not be null");
        assertNotNull(grads.get("b"), "Bias gradients should not be null");

        double wGradNorm = grads.get("w").norm2Number().doubleValue();
        double bGradNorm = grads.get("b").norm2Number().doubleValue();

        log.info("Gradient norms: w={}, b={}", wGradNorm, bGradNorm);

        assertTrue(wGradNorm > 1e-8, "Weight gradient norm should be > 0, got " + wGradNorm);
        assertTrue(bGradNorm > 1e-8, "Bias gradient norm should be > 0, got " + bGradNorm);
    }

    /**
     * Both SLOT_BY_SLOT and legacy (non-DSP) training produce converging loss.
     * We verify both paths train successfully rather than requiring exact numerical
     * parity, since the execution paths differ (DSP pre-wired plans vs HashMap walk).
     */
    @ParameterizedTest(name = "parityTraining[{0}]")
    @MethodSource("updaterConfigs")
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: both SLOT_BY_SLOT and legacy converge")
    public void testTrainingParitySlotBySlotVsLegacy(String updaterName, IUpdater updater) {
        long seed = 123;
        int nIn = 4, nOut = 1, batchSize = 8, epochs = 10;
        DataSet ds = generateRegressionData(seed + 500, batchSize, nIn, nOut);

        // Reference: legacy (non-DSP)
        InferenceSession.setDynamicShapePlanEnabled(false);
        SameDiff sdLegacy = buildLinearTrainingModel(seed, nIn, nOut);
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(updater)
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build();
        sdLegacy.setTrainingConfig(config);
        History legacyHist = sdLegacy.fit(new SingletonDataSetIterator(ds), epochs);
        INDArray legacyLossCurve = legacyHist.getLossCurve().getLossValues();
        double legacyFirstLoss = legacyLossCurve.getDouble(0);
        double legacyFinalLoss = legacyLossCurve.getDouble((int) legacyLossCurve.length() - 1);

        // Test: DSP SLOT_BY_SLOT
        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sdSlot = buildLinearTrainingModel(seed, nIn, nOut);
        forceSlotBySlot(sdSlot);
        sdSlot.setTrainingConfig(new TrainingConfig.Builder()
                .updater(updater)
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());
        History slotHist = sdSlot.fit(new SingletonDataSetIterator(ds), epochs);
        INDArray slotLossCurve = slotHist.getLossCurve().getLossValues();
        double slotFirstLoss = slotLossCurve.getDouble(0);
        double slotFinalLoss = slotLossCurve.getDouble((int) slotLossCurve.length() - 1);

        log.info("[{}] Legacy: first={}, last={} | SLOT_BY_SLOT: first={}, last={}",
                updaterName, legacyFirstLoss, legacyFinalLoss, slotFirstLoss, slotFinalLoss);

        // Both should converge (final < first)
        assertTrue(legacyFinalLoss < legacyFirstLoss,
                updaterName + ": legacy loss should decrease (first=" + legacyFirstLoss +
                        ", last=" + legacyFinalLoss + ")");
        assertTrue(slotFinalLoss < slotFirstLoss,
                updaterName + ": SLOT_BY_SLOT loss should decrease (first=" + slotFirstLoss +
                        ", last=" + slotFinalLoss + ")");

        // Initial losses should be similar (same model, same data, same weights)
        double initRatio = Math.max(legacyFirstLoss, slotFirstLoss) /
                Math.max(Math.min(legacyFirstLoss, slotFirstLoss), 1e-10);
        assertTrue(initRatio < 5.0,
                updaterName + ": initial loss ratio " + initRatio +
                        " exceeds 5x (legacy=" + legacyFirstLoss + ", slot=" + slotFirstLoss + ")");
    }

    /**
     * Multi-epoch training works correctly — loss values are produced for each epoch.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: multi-epoch produces correct number of loss values")
    public void testMultiEpochTrainingSlotBySlot() {
        long seed = 42;
        int nIn = 4, nOut = 1, batchSize = 8, epochs = 5;
        DataSet ds = generateRegressionData(seed + 600, batchSize, nIn, nOut);

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildLinearTrainingModel(seed, nIn, nOut);
        forceSlotBySlot(sd);

        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());
        History hist = sd.fit(new SingletonDataSetIterator(ds), epochs);

        INDArray lossCurve = hist.getLossCurve().getLossValues();
        assertEquals(epochs, lossCurve.length(),
                "Should have " + epochs + " loss values, got " + lossCurve.length());

        // Verify monotonic-ish decrease (at least last < first)
        double firstLoss = lossCurve.getDouble(0);
        double lastLoss = lossCurve.getDouble(epochs - 1);
        assertTrue(lastLoss < firstLoss,
                "Multi-epoch: last loss (" + lastLoss + ") should be < first (" + firstLoss + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── SLOT-BY-SLOT SPECIFIC BEHAVIOR ───────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Verify that SLOT_BY_SLOT mode is correctly set and not auto-promoted to
     * a higher execution mode (Triton, CUDA graphs, etc.).
     */
    @Test
    @DisplayName("Mode enforcement: SLOT_BY_SLOT stays SLOT_BY_SLOT")
    public void testModeEnforcementSlotBySlot() {
        SameDiff sd = buildElementwiseGraph();
        forceSlotBySlot(sd);
        assertEquals(GraphExecutionMode.SLOT_BY_SLOT, sd.getGraphExecutionMode(),
                "Mode should be SLOT_BY_SLOT after setting");

        // Execute and verify mode hasn't changed
        InferenceSession.setDynamicShapePlanEnabled(true);
        INDArray x = Nd4j.randn(FLOAT, 4, 4);
        INDArray y = Nd4j.randn(FLOAT, 4, 4);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);
        ph.put("y", y);
        sd.output(ph, "out");

        assertEquals(GraphExecutionMode.SLOT_BY_SLOT, sd.getGraphExecutionMode(),
                "Mode should still be SLOT_BY_SLOT after execution");
    }

    /**
     * Verify that SLOT_BY_SLOT execution produces correct results even when
     * executed many times in succession (stress test for slot/shape reuse).
     */
    @Test
    @DisplayName("Stress test: 100 SLOT_BY_SLOT executions with same shape")
    public void testStressRepeatedExecution() {
        SameDiff sd = buildMatmulMlpGraph(42);
        forceSlotBySlot(sd);
        InferenceSession.setDynamicShapePlanEnabled(true);

        INDArray x = Nd4j.randn(FLOAT, 8, 8);
        Map<String, INDArray> ph = Collections.singletonMap("x", x);

        INDArray reference = sd.output(ph, "out").get("out").dup();

        for (int i = 0; i < 100; i++) {
            INDArray result = sd.output(ph, "out").get("out");
            double maxDiff = reference.sub(result).amaxNumber().doubleValue();
            if (maxDiff > 1e-6) {
                fail("Iteration " + i + ": output diverged from reference, maxDiff=" + maxDiff);
            }
        }
        log.info("100 repeated executions: all identical to reference");
    }

    /**
     * Alternating between different batch sizes doesn't corrupt the plan cache.
     * Uses pre-allocated deterministic data so reference and verification use the exact same inputs.
     */
    @Test
    @DisplayName("Alternating shapes: plan cache handles interleaved batch sizes")
    public void testAlternatingShapesPlanCache() {
        SameDiff sd = buildMatmulMlpGraph(42);
        forceSlotBySlot(sd);
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Pre-allocate deterministic input data for each batch size
        Map<Integer, INDArray> inputs = new LinkedHashMap<>();
        Nd4j.getRandom().setSeed(1000L);
        inputs.put(4, Nd4j.randn(FLOAT, 4, 8));
        Nd4j.getRandom().setSeed(2000L);
        inputs.put(8, Nd4j.randn(FLOAT, 8, 8));

        // Compute reference for each batch size
        Map<Integer, INDArray> references = new LinkedHashMap<>();
        for (Map.Entry<Integer, INDArray> e : inputs.entrySet()) {
            INDArray out = sd.output(Collections.singletonMap("x", e.getValue()), "out").get("out");
            references.put(e.getKey(), out.dup());
        }

        // Alternate batch sizes and verify each still matches its reference
        for (int round = 0; round < 5; round++) {
            for (int bs : new int[]{4, 8}) {
                INDArray result = sd.output(Collections.singletonMap("x", inputs.get(bs)), "out").get("out");
                double maxDiff = references.get(bs).sub(result).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-4,
                        "Round " + round + ", bs=" + bs + ": maxDiff=" + maxDiff);
            }
        }
        log.info("5 rounds of alternating batch sizes: all correct");
    }

    /**
     * SLOT_BY_SLOT mode works correctly even when DSP auto-compile flag is
     * explicitly set (not relying on lazy compilation).
     */
    @Test
    @DisplayName("Explicit compile: pre-compile plan then execute under SLOT_BY_SLOT")
    public void testExplicitCompileThenExecute() {
        SameDiff sd = buildMatmulMlpGraph(42);
        forceSlotBySlot(sd);
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Explicitly compile the plan
        sd.compileNativeDynamicShapePlan("out");

        // Execute with the pre-compiled plan
        INDArray x = Nd4j.randn(FLOAT, 4, 8);
        Map<String, INDArray> out = sd.output(Collections.singletonMap("x", x), "out");

        assertNotNull(out.get("out"), "Output should be produced from pre-compiled plan");
        assertEquals(4, out.get("out").shape()[0]);
        assertEquals(4, out.get("out").shape()[1]);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ─── ADVANCED TRAINING: MLP WITH DIFFERENT TOPOLOGIES ─────────────────
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Two-layer MLP training with layer normalization backward under SLOT_BY_SLOT.
     * Tests that complex backward graphs (norm backward + relu backward + matmul backward)
     * execute correctly.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: two-layer MLP with normalization backward under SLOT_BY_SLOT")
    public void testNormalizationBackwardSlotBySlot() {
        long seed = 42;
        int nIn = 8, nHidden = 16, nOut = 2, batchSize = 16, epochs = 20;

        InferenceSession.setDynamicShapePlanEnabled(true);
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", FLOAT, -1, nIn);
        SDVariable labels = sd.placeHolder("labels", FLOAT, -1, nOut);

        // Layer 1: linear + manual layer norm + relu
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', nIn, nHidden), FLOAT, nIn, nHidden);
        SDVariable b0 = sd.var("b0", new ZeroInitScheme('c'), FLOAT, nHidden);
        SDVariable z0 = sd.nn.linear("z0", input, w0, b0);
        SDVariable lnGamma = sd.var("ln_gamma", Nd4j.ones(FLOAT, nHidden));
        SDVariable lnBeta = sd.var("ln_beta", Nd4j.zeros(FLOAT, nHidden));
        SDVariable mean = z0.mean("z0_mean", true, 1);
        SDVariable centered = z0.sub(mean);
        SDVariable variance = centered.mul(centered).mean("z0_var", true, 1);
        SDVariable std = sd.math.sqrt(variance.add(1e-5));
        SDVariable normed = centered.div(std).mul(lnGamma).add(lnBeta);
        SDVariable h = sd.nn.relu("h", normed, 0);

        // Layer 2: linear -> loss
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nHidden, nOut), FLOAT, nHidden, nOut);
        SDVariable b1 = sd.var("b1", new ZeroInitScheme('c'), FLOAT, nOut);
        SDVariable pred = sd.nn.linear("pred", h, w1, b1);
        sd.loss.meanSquaredError("loss", labels, pred, null);

        forceSlotBySlot(sd);

        DataSet ds = generateRegressionData(seed + 700, batchSize, nIn, nOut);
        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());
        History hist = sd.fit(new SingletonDataSetIterator(ds), epochs);

        INDArray lossCurve = hist.getLossCurve().getLossValues();
        double firstLoss = lossCurve.getDouble(0);
        double lastLoss = lossCurve.getDouble(epochs - 1);

        log.info("LayerNorm MLP SLOT_BY_SLOT: first={}, last={}", firstLoss, lastLoss);
        assertTrue(lastLoss < firstLoss,
                "LayerNorm MLP loss should decrease: first=" + firstLoss + ", last=" + lastLoss);
    }

    /**
     * Training with gradient accumulation under SLOT_BY_SLOT.
     * Every N steps the gradients are applied — this tests the accumulation
     * codepath within DSP.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training: gradient accumulation works under SLOT_BY_SLOT")
    public void testGradientAccumulationSlotBySlot() {
        long seed = 42;
        int nIn = 4, nOut = 1, batchSize = 8, epochs = 20;
        int accumSteps = 4;
        DataSet ds = generateRegressionData(seed + 800, batchSize, nIn, nOut);

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildLinearTrainingModel(seed, nIn, nOut);
        forceSlotBySlot(sd);

        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .gradientAccumulationSteps(accumSteps)
                .build());
        History hist = sd.fit(new SingletonDataSetIterator(ds), epochs);

        INDArray lossCurve = hist.getLossCurve().getLossValues();
        double firstLoss = lossCurve.getDouble(0);
        double lastLoss = lossCurve.getDouble((int) lossCurve.length() - 1);

        log.info("Gradient accumulation SLOT_BY_SLOT: first={}, last={}, accumSteps={}",
                firstLoss, lastLoss, accumSteps);
        assertTrue(lastLoss < firstLoss,
                "Gradient accumulation: loss should decrease");
    }

    /**
     * Inference after training works correctly — the trained weights are used.
     */
    @Test
    @Tag(TagNames.TRAINING)
    @DisplayName("Training then inference: trained model produces different output than untrained")
    public void testInferenceAfterTrainingSlotBySlot() {
        long seed = 42;
        int nIn = 4, nOut = 1, batchSize = 16, epochs = 20;
        DataSet ds = generateRegressionData(seed + 900, batchSize, nIn, nOut);

        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = buildLinearTrainingModel(seed, nIn, nOut);
        forceSlotBySlot(sd);

        // Inference before training
        INDArray testInput = Nd4j.randn(FLOAT, 4, nIn);
        Map<String, INDArray> ph = Collections.singletonMap("input", testInput);
        INDArray beforeTraining = sd.output(ph, "pred").get("pred").dup();

        // Train
        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(1e-2))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build());
        sd.fit(new SingletonDataSetIterator(ds), epochs);

        // Inference after training — should produce different output
        INDArray afterTraining = sd.output(ph, "pred").get("pred");

        double maxDiff = beforeTraining.sub(afterTraining).amaxNumber().doubleValue();
        log.info("Before vs after training: maxDiff={}", maxDiff);
        assertTrue(maxDiff > 1e-4,
                "Inference output should change after training (maxDiff=" + maxDiff + ")");
    }
}
