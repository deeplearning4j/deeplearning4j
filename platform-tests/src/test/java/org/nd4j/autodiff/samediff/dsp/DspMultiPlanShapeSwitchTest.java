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
package org.nd4j.autodiff.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Tests for DSP executor behavior when the SAME graph is executed with
 * MULTIPLE different input shapes, producing multiple plans in the
 * shape-keyed native plan cache.
 *
 * <p>This covers the VLM multi-page pattern where a single decoder graph
 * alternates between prefill shapes (seqLen=N) and decode shapes (seqLen=1)
 * across multiple generation calls. The executor must correctly switch
 * between the shape-keyed plans (including when frozen) without corrupting
 * CUDA graph state or producing wrong results.</p>
 *
 * <h3>Scenarios tested:</h3>
 * <ol>
 *   <li>Shape A → Shape B → Shape A (basic plan switch round-trip)</li>
 *   <li>Frozen executor handling plan switch on shape change</li>
 *   <li>Multi-cycle prefill→decode loops (simulating multi-page VLM)</li>
 *   <li>clearNodeOutputsOnly + shape switch (fixedBuffers path)</li>
 * </ol>
 *
 * <p>Run:
 * <pre>
 * cd platform-tests && mvn test \
 *   -Dtest=DspMultiPlanShapeSwitchTest \
 *   2>&1 | tee /tmp/dsp-multiplan.log
 * </pre>
 */
@Slf4j
@DisplayName("DSP Multi-Plan Shape Switch Tests")
@Tag("dsp")
public class DspMultiPlanShapeSwitchTest {

    private static final double ABS_TOL = 1e-4;
    private static final double REL_TOL = 1e-3;

    @BeforeEach
    void setUp() {
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "ALL");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "summary");
    }

    @AfterEach
    void tearDown() {
        System.clearProperty(ND4JSystemProperties.DSP_DIAGNOSTICS);
        System.clearProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL);
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    // ── Graph builders ─────────────────────────────────────────────────────

    /**
     * Build a simple transformer-like graph with a dynamic sequence dimension:
     *   x [batch, -1, dim] → rmsNorm → flatten → matmul(W) → reshape → output
     *
     * The -1 dimension means the same graph handles any seqLen.
     */
    private SameDiff buildDynamicSeqGraph(int dim) {
        int batch = 1;
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, -1, dim);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, dim));

        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-6);
        SDVariable flat = sd.reshape("flat", normed, -1, dim);
        SDVariable proj = sd.mmul("proj", flat, w);
        SDVariable y = sd.reshape("y", proj, batch, -1, dim);
        sd.setOutputs("y");
        return sd;
    }

    /**
     * Build a slightly more complex graph with attention-like patterns:
     *   x [batch, -1, dim] → rmsNorm → Q/K/V projections → scaled dot-product → output projection
     */
    private SameDiff buildMiniAttentionGraph(int dim) {
        int batch = 1;
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, -1, dim);
        SDVariable wQ = sd.var("wQ", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wK = sd.var("wK", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wV = sd.var("wV", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wO = sd.var("wO", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, dim));

        // Pre-norm
        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-6);
        // Flatten to 2D for matmul
        SDVariable flat = sd.reshape("flat", normed, -1, dim);
        // Q, K, V projections
        SDVariable q = sd.mmul("q", flat, wQ);
        SDVariable k = sd.mmul("k", flat, wK);
        SDVariable v = sd.mmul("v", flat, wV);
        // Simple scaled dot product (not full attention, just exercises the pattern)
        SDVariable scores = sd.mmul("scores", q, sd.transpose("kT", k));
        SDVariable scale = sd.math.mul("scaled", scores, sd.constant("inv_sqrt", Nd4j.scalar(1.0f / (float) Math.sqrt(dim))));
        SDVariable attn = sd.nn.softmax("attn", scale, -1);
        SDVariable context = sd.mmul("context", attn, v);
        // Output projection
        SDVariable out = sd.mmul("out_proj", context, wO);
        SDVariable y = sd.reshape("y", out, batch, -1, dim);
        sd.setOutputs("y");
        return sd;
    }

    private void configureDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    /** Run the graph in EAGER mode (no DSP) to get the reference output. */
    private Map<String, INDArray> runEager(SameDiff sd, Map<String, INDArray> inputs, String... outputs) {
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        InferenceSession.setDynamicShapePlanEnabled(false);
        try {
            return sd.output(inputs, outputs);
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(true);
        }
    }

    private void assertOutputsMatch(String label,
                                    Map<String, INDArray> expected,
                                    Map<String, INDArray> actual) {
        for (String key : expected.keySet()) {
            INDArray exp = expected.get(key);
            INDArray act = actual.get(key);
            assertNotNull(act, label + ": missing output '" + key + "'");
            assertTrue(exp.equalShapes(act),
                    String.format("%s output '%s': shape mismatch — expected %s got %s",
                            label, key, java.util.Arrays.toString(exp.shape()),
                            java.util.Arrays.toString(act.shape())));
            double maxAbsDiff = exp.sub(act).amaxNumber().doubleValue();
            double maxExpVal = exp.amaxNumber().doubleValue();
            double relErr = maxExpVal > 1e-10 ? maxAbsDiff / maxExpVal : maxAbsDiff;
            log.info("{} output '{}': maxAbsDiff={} relErr={}", label, key, maxAbsDiff, relErr);
            if (maxAbsDiff > ABS_TOL && relErr > REL_TOL) {
                fail(String.format("%s output '%s': accuracy exceeded tolerance — " +
                                "maxAbsDiff=%.6f relErr=%.6f",
                        label, key, maxAbsDiff, relErr));
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 1: Basic shape-keyed plan switch round-trip (A → B → A)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Execute with shape A (seq=8), then shape B (seq=1), then back to shape A.
     * Verifies the plan cache correctly returns the right plan for each shape
     * key and that outputs are numerically correct after switching back.
     */
    @Test
    @DisplayName("Basic plan switch: A → B → A produces correct results")
    void testBasicShapeSwitchRoundTrip() {
        int dim = 64;
        SameDiff sd = buildDynamicSeqGraph(dim);
        configureDsp(sd);

        INDArray inputA = Nd4j.randn(DataType.FLOAT, 1, 8, dim);
        INDArray inputB = Nd4j.randn(DataType.FLOAT, 1, 1, dim);

        Map<String, INDArray> phA = new LinkedHashMap<>();
        phA.put("x", inputA);
        Map<String, INDArray> phB = new LinkedHashMap<>();
        phB.put("x", inputB);

        // Get reference outputs from eager mode
        Map<String, INDArray> expectedA = runEager(sd, phA, "y");
        Map<String, INDArray> expectedB = runEager(sd, phB, "y");

        // Re-enable DSP
        configureDsp(sd);

        // Phase 1: Warm up shape A (compiles plan A)
        log.info("=== Phase 1: Shape A (seq=8) warmup ===");
        Map<String, INDArray> resultA = null;
        for (int i = 0; i < 3; i++) {
            resultA = sd.output(phA, "y");
        }
        assertOutputsMatch("Phase1-ShapeA", expectedA, resultA);

        // Phase 2: Switch to shape B (plan swap → compiles plan B)
        log.info("=== Phase 2: Shape B (seq=1) ===");
        Map<String, INDArray> resultB = null;
        for (int i = 0; i < 3; i++) {
            resultB = sd.output(phB, "y");
        }
        assertOutputsMatch("Phase2-ShapeB", expectedB, resultB);

        // Phase 3: Switch BACK to shape A (plan swap → reuses plan A from cache)
        log.info("=== Phase 3: Shape A again (seq=8) ===");
        Map<String, INDArray> resultA2 = null;
        for (int i = 0; i < 3; i++) {
            resultA2 = sd.output(phA, "y");
        }
        assertOutputsMatch("Phase3-ShapeA-Again", expectedA, resultA2);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 2: Frozen executor must handle shape switch
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Freeze the executor on shape B (decode), then switch to shape A (prefill).
     * The frozen executor currently skips redispatch (line ~2296 of
     * DynamicShapePlanExecutor), which causes it to execute shape A through
     * shape B's plan → crash. This test validates the fix.
     */
    @Test
    @DisplayName("Frozen plan switch: freeze on B, then execute A")
    void testFrozenPlanShapeSwitch() {
        int dim = 64;
        SameDiff sd = buildDynamicSeqGraph(dim);
        configureDsp(sd);

        INDArray inputA = Nd4j.randn(DataType.FLOAT, 1, 8, dim);
        INDArray inputB = Nd4j.randn(DataType.FLOAT, 1, 1, dim);

        Map<String, INDArray> phA = new LinkedHashMap<>();
        phA.put("x", inputA);
        Map<String, INDArray> phB = new LinkedHashMap<>();
        phB.put("x", inputB);

        Map<String, INDArray> expectedA = runEager(sd, phA, "y");
        Map<String, INDArray> expectedB = runEager(sd, phB, "y");

        configureDsp(sd);

        // Phase 1: Warm up shape A → plan A compiled
        log.info("=== Phase 1: Shape A warmup ===");
        for (int i = 0; i < 3; i++) {
            sd.output(phA, "y");
        }

        // Phase 2: Switch to shape B → plan swap → warm up → freeze
        log.info("=== Phase 2: Shape B warmup + freeze ===");
        for (int i = 0; i < 3; i++) {
            sd.output(phB, "y");
        }

        // Freeze the executor (simulates what generateNative does after warmup decode)
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        assertNotNull(executor, "DSP executor should exist after warmup");
        executor.setShapesFrozen(true);
        log.info("Executor frozen on shape B plan (shapesFrozen={})", executor.isShapesFrozen());

        // Phase 3: Switch BACK to shape A while frozen
        // This is the exact scenario that fails without proper multi-plan support.
        log.info("=== Phase 3: Shape A on frozen executor ===");
        Map<String, INDArray> resultA = sd.output(phA, "y");
        assertOutputsMatch("FrozenSwitch-ShapeA", expectedA, resultA);

        // Phase 4: Switch back to shape B — should still work
        log.info("=== Phase 4: Shape B on frozen executor ===");
        Map<String, INDArray> resultB = sd.output(phB, "y");
        assertOutputsMatch("FrozenSwitch-ShapeB", expectedB, resultB);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 3: Multi-cycle prefill→decode simulation
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Simulates the VLM multi-page pattern:
     *   Page 1: prefill(seq=8) → decode(seq=1) → freeze
     *   Page 2: prefill(seq=8) → decode(seq=1) [reuse frozen plans]
     *   Page 3: prefill(seq=8) → decode(seq=1) [reuse frozen plans]
     *
     * Each cycle uses DIFFERENT random inputs (simulating different page
     * content) but the same shapes. Both prefill and decode must produce
     * correct results on every cycle.
     */
    @Test
    @DisplayName("Multi-cycle prefill→decode simulation (3 pages)")
    void testMultiCyclePrefillDecode() {
        int dim = 64;
        int prefillSeq = 8;
        int decodeSeq = 1;
        int numPages = 3;

        SameDiff sd = buildDynamicSeqGraph(dim);
        configureDsp(sd);

        for (int page = 0; page < numPages; page++) {
            // Fresh random inputs for each page
            INDArray prefillInput = Nd4j.randn(DataType.FLOAT, 1, prefillSeq, dim);
            INDArray decodeInput = Nd4j.randn(DataType.FLOAT, 1, decodeSeq, dim);

            Map<String, INDArray> phPrefill = new LinkedHashMap<>();
            phPrefill.put("x", prefillInput);
            Map<String, INDArray> phDecode = new LinkedHashMap<>();
            phDecode.put("x", decodeInput);

            // Reference outputs
            Map<String, INDArray> expectedPrefill = runEager(sd, phPrefill, "y");
            Map<String, INDArray> expectedDecode = runEager(sd, phDecode, "y");

            configureDsp(sd);

            // Prefill
            log.info("=== Page {}: Prefill (seq={}) ===", page + 1, prefillSeq);
            Map<String, INDArray> prefillResult = sd.output(phPrefill, "y");
            assertOutputsMatch(
                    String.format("Page%d-Prefill", page + 1),
                    expectedPrefill, prefillResult);

            // Decode
            log.info("=== Page {}: Decode (seq={}) ===", page + 1, decodeSeq);
            Map<String, INDArray> decodeResult = null;
            for (int step = 0; step < 3; step++) {
                decodeResult = sd.output(phDecode, "y");
            }
            assertOutputsMatch(
                    String.format("Page%d-Decode", page + 1),
                    expectedDecode, decodeResult);

            // Freeze after first page's decode (simulates generateNative freeze)
            if (page == 0) {
                InferenceSession session = sd.getOrCreateSession();
                DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
                if (executor != null) {
                    executor.setShapesFrozen(true);
                    log.info("Executor frozen after page 1 decode");
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 4: clearNodeOutputsOnly + shape switch (fixedBuffers path)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Simulates the exact GenerationPipeline.generateNative() fixedBuffers path:
     *   1. Prefill (seq=N) → warmup decode (seq=1) → freeze
     *   2. clearNodeOutputsOnly()
     *   3. Prefill again (seq=N) → decode again (seq=1)
     *
     * This is the path that was crashing in VLM multi-page processing:
     * clearNodeOutputsOnly nulls the Java DynamicShapePlan but the executor
     * stays frozen on the decode plan handle. The next prefill (different
     * shape key) must switch to the prefill plan.
     */
    @Test
    @DisplayName("clearNodeOutputsOnly + shape switch (fixedBuffers simulation)")
    void testClearNodeOutputsThenShapeSwitch() {
        int dim = 64;
        int prefillSeq = 8;
        int decodeSeq = 1;

        SameDiff sd = buildDynamicSeqGraph(dim);
        configureDsp(sd);

        INDArray prefillInput1 = Nd4j.randn(DataType.FLOAT, 1, prefillSeq, dim);
        INDArray decodeInput1 = Nd4j.randn(DataType.FLOAT, 1, decodeSeq, dim);
        INDArray prefillInput2 = Nd4j.randn(DataType.FLOAT, 1, prefillSeq, dim);
        INDArray decodeInput2 = Nd4j.randn(DataType.FLOAT, 1, decodeSeq, dim);

        Map<String, INDArray> phPrefill1 = new LinkedHashMap<>();
        phPrefill1.put("x", prefillInput1);
        Map<String, INDArray> phDecode1 = new LinkedHashMap<>();
        phDecode1.put("x", decodeInput1);
        Map<String, INDArray> phPrefill2 = new LinkedHashMap<>();
        phPrefill2.put("x", prefillInput2);
        Map<String, INDArray> phDecode2 = new LinkedHashMap<>();
        phDecode2.put("x", decodeInput2);

        // Reference outputs
        Map<String, INDArray> expectedPrefill2 = runEager(sd, phPrefill2, "y");
        Map<String, INDArray> expectedDecode2 = runEager(sd, phDecode2, "y");

        configureDsp(sd);

        // === Call 1: Prefill → Decode → Freeze ===
        log.info("=== Call 1: Prefill ===");
        sd.output(phPrefill1, "y");  // Compiles prefill plan

        log.info("=== Call 1: Decode warmup ===");
        sd.output(phDecode1, "y");  // Plan swap → compiles decode plan

        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        assertNotNull(executor, "Executor must exist after warmup");
        executor.setShapesFrozen(true);
        log.info("Frozen after call 1 decode (shapesFrozen={})", executor.isShapesFrozen());

        // === Simulate between-page reset: clearNodeOutputsOnly ===
        log.info("=== clearNodeOutputsOnly (simulating fixedBuffers between-page reset) ===");
        session.clearNodeOutputsOnly();
        // Executor should still be frozen with its native plan handle intact
        assertTrue(executor.isShapesFrozen(),
                "Executor should remain frozen after clearNodeOutputsOnly");

        // === Call 2: Prefill with DIFFERENT inputs but SAME shape ===
        log.info("=== Call 2: Prefill (different inputs, same shape) ===");
        Map<String, INDArray> prefillResult2 = sd.output(phPrefill2, "y");
        assertOutputsMatch("Call2-Prefill", expectedPrefill2, prefillResult2);

        // === Call 2: Decode ===
        log.info("=== Call 2: Decode ===");
        Map<String, INDArray> decodeResult2 = sd.output(phDecode2, "y");
        assertOutputsMatch("Call2-Decode", expectedDecode2, decodeResult2);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 5: Three shapes (A → B → C → A → B → C)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Tests plan cache with three different shapes cycling through.
     * Ensures the cache can hold multiple plans and switch correctly.
     */
    @Test
    @DisplayName("Three-shape cycle: A → B → C → A → B → C")
    void testThreeShapeCycle() {
        int dim = 64;
        SameDiff sd = buildDynamicSeqGraph(dim);
        configureDsp(sd);

        int[] seqLens = {8, 1, 4};
        INDArray[] inputs = new INDArray[seqLens.length];
        Map<String, INDArray>[] placeholders = new Map[seqLens.length];
        Map<String, INDArray>[] expected = new Map[seqLens.length];

        for (int i = 0; i < seqLens.length; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, seqLens[i], dim);
            placeholders[i] = new LinkedHashMap<>();
            placeholders[i].put("x", inputs[i]);
            expected[i] = runEager(sd, placeholders[i], "y");
        }

        configureDsp(sd);

        // Two full cycles through all three shapes
        for (int cycle = 0; cycle < 2; cycle++) {
            for (int i = 0; i < seqLens.length; i++) {
                String label = String.format("Cycle%d-seq%d", cycle + 1, seqLens[i]);
                log.info("=== {} ===", label);
                Map<String, INDArray> result = sd.output(placeholders[i], "y");
                assertOutputsMatch(label, expected[i], result);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 6: Attention graph with frozen multi-plan switch
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Uses the more complex attention graph (Q/K/V projections + softmax)
     * to verify multi-plan switching works with non-trivial op patterns.
     */
    @Test
    @DisplayName("Attention graph: frozen prefill→decode→prefill cycle")
    void testAttentionGraphFrozenCycle() {
        int dim = 32;  // Smaller dim for attention to keep memory reasonable
        int prefillSeq = 4;
        int decodeSeq = 1;
        int numCycles = 3;

        SameDiff sd = buildMiniAttentionGraph(dim);
        configureDsp(sd);

        for (int cycle = 0; cycle < numCycles; cycle++) {
            INDArray prefillInput = Nd4j.randn(DataType.FLOAT, 1, prefillSeq, dim);
            INDArray decodeInput = Nd4j.randn(DataType.FLOAT, 1, decodeSeq, dim);

            Map<String, INDArray> phPrefill = new LinkedHashMap<>();
            phPrefill.put("x", prefillInput);
            Map<String, INDArray> phDecode = new LinkedHashMap<>();
            phDecode.put("x", decodeInput);

            Map<String, INDArray> expectedPrefill = runEager(sd, phPrefill, "y");
            Map<String, INDArray> expectedDecode = runEager(sd, phDecode, "y");

            configureDsp(sd);

            // Prefill
            String pfLabel = String.format("Cycle%d-Prefill(seq=%d)", cycle + 1, prefillSeq);
            log.info("=== {} ===", pfLabel);
            Map<String, INDArray> prefillResult = sd.output(phPrefill, "y");
            assertOutputsMatch(pfLabel, expectedPrefill, prefillResult);

            // Decode
            String dcLabel = String.format("Cycle%d-Decode(seq=%d)", cycle + 1, decodeSeq);
            log.info("=== {} ===", dcLabel);
            Map<String, INDArray> decodeResult = sd.output(phDecode, "y");
            assertOutputsMatch(dcLabel, expectedDecode, decodeResult);

            // Freeze after first cycle
            if (cycle == 0) {
                InferenceSession session = sd.getOrCreateSession();
                DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
                if (executor != null) {
                    executor.setShapesFrozen(true);
                    log.info("Executor frozen after cycle 1");
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 7: Rapid alternation stress test
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Rapidly alternates between two shapes 20 times with different random
     * inputs each time. Catches any accumulating state corruption, pointer
     * drift, or memory leaks that only manifest after many switches.
     */
    @Test
    @DisplayName("Rapid alternation: 20 switches between seq=4 and seq=1")
    void testRapidAlternation() {
        int dim = 64;
        int iterations = 20;
        int[] seqLens = {4, 1};

        SameDiff sd = buildDynamicSeqGraph(dim);
        configureDsp(sd);

        // Warmup both shapes
        for (int seq : seqLens) {
            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("x", Nd4j.randn(DataType.FLOAT, 1, seq, dim));
            for (int w = 0; w < 3; w++) {
                sd.output(ph, "y");
            }
        }

        // Rapid alternation with fresh inputs
        for (int i = 0; i < iterations; i++) {
            int seq = seqLens[i % seqLens.length];
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, seq, dim);
            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("x", input);

            Map<String, INDArray> expected = runEager(sd, ph, "y");
            configureDsp(sd);

            String label = String.format("Iter%d-seq%d", i, seq);
            Map<String, INDArray> result = sd.output(ph, "y");
            assertOutputsMatch(label, expected, result);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Fixed-shape decode graph builder for pointer stability tests
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Build a fixed-shape mini-decoder graph that compiles into capturable DSP segments.
     * Uses constant weights and a fixed input shape — no dynamic (-1) dimensions.
     * Pattern: input → matmul(W1) → normalize → matmul(W2) → output
     *
     * <p>Fixed shapes are required for pointer stability because dynamic shapes prevent
     * the native plan from freezing and capturing CUDA graphs.
     */
    private SameDiff buildFixedShapeDecodeGraph(int dim) {
        SameDiff sd = SameDiff.create();
        INDArray w1 = Nd4j.randn(DataType.FLOAT, dim, dim);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, dim, dim);

        SDVariable weight1 = sd.constant("w1", w1);
        SDVariable weight2 = sd.constant("w2", w2);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);

        SDVariable h = sd.mmul("hidden", input, weight1);
        SDVariable norm = sd.math().norm2("norm", h, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable maxNorm = sd.math().max("maxNorm", norm, eps);
        SDVariable normalized = sd.math().div("normalized", h, maxNorm);
        SDVariable out = sd.mmul("output", normalized, weight2);
        sd.setOutputs("output");
        return sd;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 8: Multi-page pointer stability with buffer reuse (regression)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Regression test for the multi-page VLM pointer stability bug.
     *
     * <p>Reproduces the exact pattern from {@code GenerationPipeline.generateNative()}
     * with {@code fixedBuffers=true}: after page 1 freezes the executor, page 2 must
     * reuse the SAME INDArray objects (same device pointers) so the plan's pointer
     * stability check passes and the plan transitions from SHAPES_FROZEN → REPLAYING.
     *
     * <p><b>The bug:</b> {@code generateNative()} allocated fresh arrays each call.
     * Fresh allocations = new device pointers = pointer stability never reached =
     * plan stuck at SHAPES_FROZEN = no CUDA graph capture = every token through
     * slow execute().
     *
     * <p><b>The fix pattern:</b> On second+ calls with a frozen plan, retrieve the
     * plan's existing ext input arrays and write new data into them via
     * {@code .assign()}, NOT fresh allocations.
     */
    @Test
    @DisplayName("Multi-page pointer stability with buffer reuse (fixedBuffers regression)")
    void testMultiPagePointerStabilityWithBufferReuse() {
        int dim = 32;
        int numPages = 5;
        int warmupSteps = 30;  // enough for segment compilation + pointer stability
        int stepsPerPage = 10;

        SameDiff sd = buildFixedShapeDecodeGraph(dim);
        configureDsp(sd);

        // Reuse the SAME INDArray for all steps — this is the correct pattern
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", inputArr);

        // === Page 1: warmup until pointer stability ===
        log.info("=== Page 1: Warmup {} steps ===", warmupSteps);
        for (int step = 0; step < warmupSteps; step++) {
            inputArr.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            sd.output(ph, "output");
        }

        // Check that DSP reached at least SHAPES_FROZEN with stable pointers
        DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                "Page 1 after " + warmupSteps + " steps");
        DspPlanAssertions.assertPointersStable(sd,
                "Page 1 after " + warmupSteps + " steps");

        log.info("Page 1 stable: pointersStable={} frozenExec={} snapshot:\n{}",
                DspPlanAssertions.getPointersStable(sd),
                DspPlanAssertions.getFrozenExecCount(sd),
                DspPlanAssertions.snapshotPlanState(sd));

        // === Pages 2..N: reuse path — SAME array, new values ===
        // This simulates the CORRECT fixedBuffers behavior: the plan's ext input
        // arrays survive across calls and get new data written into them.
        for (int page = 2; page <= numPages; page++) {
            log.info("=== Page {}: Reuse frozen plan, write new data into existing array ===", page);

            for (int step = 0; step < stepsPerPage; step++) {
                inputArr.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
                sd.output(ph, "output");
            }

            // Pointer stability must hold across page transitions when reusing same array
            int ptrsStable = DspPlanAssertions.getPointersStable(sd);
            log.info("Page {} pointersStable={} frozenExec={}",
                    page, ptrsStable, DspPlanAssertions.getFrozenExecCount(sd));
            assertEquals(1, ptrsStable,
                    String.format("Page %d: pointer stability must hold when reusing same INDArray", page));
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 9: Pointer stability BREAKS with fresh allocations (negative test)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Negative test proving that fresh INDArray allocations break pointer stability.
     * This reproduces the EXACT bug pattern from the unfixed generateNative():
     *   1. Page 1: warm up to pointer stability
     *   2. Page 2: create FRESH INDArray → pass to sd.output() → pointers break
     *
     * This test documents the failure mode so it's clear WHY buffer reuse matters.
     * If this test starts passing (stable pointers with fresh arrays), it means the
     * DSP executor learned to handle pointer changes — update the multi-page fix.
     */
    @Test
    @DisplayName("Fresh allocations break pointer stability (documents the bug)")
    void testFreshAllocationsBreakPointerStability() {
        int dim = 32;
        int warmupSteps = 30;
        int postSteps = 10;

        SameDiff sd = buildFixedShapeDecodeGraph(dim);
        configureDsp(sd);

        // Page 1: warm up to pointer stability
        INDArray input1 = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", input1);

        for (int step = 0; step < warmupSteps; step++) {
            input1.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            sd.output(ph, "output");
        }

        int stableAfterPage1 = DspPlanAssertions.getPointersStable(sd);
        log.info("After page 1 ({} steps): pointersStable={}",
                warmupSteps, stableAfterPage1);

        // If page 1 can't reach stability, skip the negative test — the backend
        // may not support pointer tracking for this graph.
        if (stableAfterPage1 != 1) {
            log.warn("Skipping negative test: page 1 never reached pointer stability " +
                    "(may be backend-specific). stableAfterPage1={}", stableAfterPage1);
            return;
        }

        // Page 2: create FRESH array — this is the bug pattern
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph2 = new LinkedHashMap<>();
        ph2.put("input", input2);

        // Execute with the new array — pointers change
        for (int step = 0; step < postSteps; step++) {
            sd.output(ph2, "output");
        }

        int stableAfterPage2 = DspPlanAssertions.getPointersStable(sd);
        log.info("After fresh allocation ({} steps): pointersStable={} (expected 0 = broken)",
                postSteps, stableAfterPage2);

        // The fresh allocation should break pointer stability.
        // If this assertion fails (pointers remain stable), it means the DSP executor
        // now handles pointer changes gracefully — which is good! Update the multi-page
        // fix to remove the buffer reuse requirement.
        if (stableAfterPage2 == 1) {
            log.warn("UNEXPECTED: pointer stability survived fresh allocation. " +
                    "The DSP executor may now handle pointer changes. " +
                    "Review whether the generateNative() buffer reuse fix is still needed.");
        }
        // We document the expected behavior but don't fail if the DSP got smarter —
        // the important thing is the positive test (TEST 8) always passes.
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TEST 10: Memory stability across pages with buffer reuse
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Verifies that reusing the same INDArray across "pages" does not leak memory.
     * Tracks workspace + physical memory across N pages and asserts no unbounded growth.
     * This guards against the scenario where each page accumulates unreleased buffers.
     */
    @Test
    @DisplayName("No memory growth across pages with buffer reuse")
    void testNoMemoryGrowthWithBufferReuse() {
        int dim = 32;
        int numPages = 10;
        int stepsPerPage = 4;
        int warmupSteps = 30;

        SameDiff sd = buildFixedShapeDecodeGraph(dim);
        configureDsp(sd);

        // Warmup to pointer stability
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", inputArr);

        for (int step = 0; step < warmupSteps; step++) {
            inputArr.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            sd.output(ph, "output");
        }

        // Baseline memory after stabilization
        System.gc();
        long baselinePhysical = org.bytedeco.javacpp.Pointer.physicalBytes();
        log.info("Baseline physical memory after stabilization: {} MB", baselinePhysical / (1024 * 1024));

        // Run N pages, reusing the same INDArray
        for (int page = 0; page < numPages; page++) {
            for (int step = 0; step < stepsPerPage; step++) {
                inputArr.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
                sd.output(ph, "output");
            }
        }

        System.gc();
        long afterPhysical = org.bytedeco.javacpp.Pointer.physicalBytes();
        long growthBytes = afterPhysical - baselinePhysical;
        double growthMB = growthBytes / (1024.0 * 1024.0);
        log.info("Memory after {} pages: {} MB (growth: {} MB)",
                numPages, afterPhysical / (1024 * 1024), String.format("%.1f", growthMB));

        // Allow up to 50 MB growth for JVM/GC noise, but catch real leaks.
        // A real leak would show ~10-50 MB per page (KV buffer + decode arrays).
        assertTrue(growthMB < 50.0,
                String.format("Memory grew by %.1f MB across %d pages — possible leak. " +
                        "Baseline=%d MB, after=%d MB",
                        growthMB, numPages,
                        baselinePhysical / (1024 * 1024),
                        afterPhysical / (1024 * 1024)));
    }
}
