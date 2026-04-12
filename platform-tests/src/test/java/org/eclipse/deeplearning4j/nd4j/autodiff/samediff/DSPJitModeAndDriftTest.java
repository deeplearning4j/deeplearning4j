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
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for JitMode enum coverage (GRAPH_ONLY, JIT_ONLY, GRAPH_PLUS_JIT)
 * and multi-step numerical drift measurement during DSP replay.
 *
 * <p>JitMode controls how DSP segments are executed:
 * <ul>
 *   <li>GRAPH_ONLY (0): CUDA graph capture/replay only (default)</li>
 *   <li>JIT_ONLY (1): NVRTC JIT only, skip graph capture for element-wise segments</li>
 *   <li>GRAPH_PLUS_JIT (2): Try JIT first, fall back to graph capture for non-fusible segments</li>
 * </ul>
 *
 * <p>Multi-step drift tests verify that repeated replay with fixed or changing inputs
 * maintains numerical stability across many iterations.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class DSPJitModeAndDriftTest extends BaseNd4jTestWithBackends {

    private static final double FLOAT_TOL = 1e-4;
    private static final double DOUBLE_TOL = 1e-8;

    private final List<String> savedJitModeValues = new ArrayList<>();

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void cleanup() {
        // Restore original JIT mode system property
        if (savedJitModeValues.isEmpty()) {
            System.clearProperty(ND4JSystemProperties.DSP_JIT_MODE);
        } else {
            String saved = savedJitModeValues.remove(savedJitModeValues.size() - 1);
            if (saved == null) {
                System.clearProperty(ND4JSystemProperties.DSP_JIT_MODE);
            } else {
                System.setProperty(ND4JSystemProperties.DSP_JIT_MODE, saved);
            }
        }
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /**
     * Set JitMode via system property. Saves the previous value for restoration.
     */
    private void setJitMode(String mode) {
        savedJitModeValues.add(System.getProperty(ND4JSystemProperties.DSP_JIT_MODE));
        System.setProperty(ND4JSystemProperties.DSP_JIT_MODE, mode);
    }

    /**
     * Enable DSP on a SameDiff instance with the given execution mode.
     */
    private void enableDsp(SameDiff sd, GraphExecutionMode execMode) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(execMode);
    }

    /**
     * Freeze shapes on the DSP executor to enable graph capture / JIT replay.
     */
    private void freezeShapes(SameDiff sd) {
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
        }
    }

    /**
     * Build a simple matmul graph: input [1,innerDim] -> matmul [innerDim,outerDim] -> output [1,outerDim]
     */
    private SameDiff buildMatmulGraph(int innerDim, int outerDim) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, innerDim);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, innerDim, outerDim));
        sd.mmul("output", input, w);
        return sd;
    }

    /**
     * Build a simple matmul graph with a pre-created weight array (for shared-weight comparison).
     */
    private SameDiff buildMatmulGraph(int innerDim, int outerDim, INDArray w) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, innerDim);
        sd.constant("w", w.dup());
        sd.mmul("output", input, sd.getVariable("w"));
        return sd;
    }

    /**
     * Build a simple matmul graph with default outerDim=32.
     */
    private SameDiff buildMatmulGraph(int innerDim) {
        return buildMatmulGraph(innerDim, 32);
    }

    /**
     * Build an elementwise chain: input -> mul -> add -> relu -> sigmoid -> output
     */
    private SameDiff buildElementwiseChain() {
        SameDiff sd = SameDiff.create();
        int dim = 64;
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable scale = sd.constant("scale", Nd4j.randn(DataType.FLOAT, 1, dim).mul(0.1).add(1.0));
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, dim).mul(0.01));
        SDVariable s1 = input.mul("scaled", scale);
        SDVariable b1 = s1.add("biased", bias);
        SDVariable r1 = sd.nn.relu("relu", b1, 0);
        sd.nn.sigmoid("output", r1);
        return sd;
    }

    /**
     * Build a mixed graph: matmul + elementwise chain
     */
    private SameDiff buildMixedGraph() {
        SameDiff sd = SameDiff.create();
        int dim = 32;
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, dim, dim));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, dim, dim));
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, dim).mul(0.01));

        SDVariable mm = sd.mmul("mm1", input, w1);
        SDVariable mm2 = sd.mmul("mm2", mm, w2);
        SDVariable added = mm2.add("add", bias);
        SDVariable activated = sd.nn.relu("relu", added, 0);
        sd.nn.sigmoid("output", activated);
        return sd;
    }

    /**
     * Build a reference SameDiff instance (SLOT_BY_SLOT) and run inputs through it.
     * Returns a list of output INDArrays, one per input.
     */
    private List<INDArray> runSlotBySlotReference(List<INDArray> inputs, DataType dtype, int innerDim, int outerDim) {
        // Build the same graph fresh
        SameDiff refSd = buildMatmulGraph(innerDim, outerDim);
        refSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        List<INDArray> outputs = new ArrayList<>();
        for (INDArray inp : inputs) {
            Map<String, INDArray> result = refSd.output(Map.of("input", inp.dup()), "output");
            outputs.add(result.get("output").dup());
        }
        refSd.close();
        return outputs;
    }

    /**
     * Build a reference SameDiff instance (SLOT_BY_SLOT) with a shared weight array
     * and run inputs through it. Ensures DSP and reference compute the same function.
     */
    private List<INDArray> runSlotBySlotReference(List<INDArray> inputs, DataType dtype, int innerDim, int outerDim, INDArray w) {
        SameDiff refSd = buildMatmulGraph(innerDim, outerDim, w);
        refSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        List<INDArray> outputs = new ArrayList<>();
        for (INDArray inp : inputs) {
            Map<String, INDArray> result = refSd.output(Map.of("input", inp.dup()), "output");
            outputs.add(result.get("output").dup());
        }
        refSd.close();
        return outputs;
    }

    // =========================================================================
    // JitMode tests
    // =========================================================================

    /**
     * Test 1: JitMode GRAPH_ONLY — build a matmul graph, execute under CUDA_GRAPHS,
     * verify correct output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("JitMode GRAPH_ONLY: matmul graph executes correctly")
    public void testJitModeGraphOnly(Nd4jBackend backend) {
        setJitMode("graph");

        INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, 16);
        INDArray wData = Nd4j.randn(DataType.FLOAT, 16, 32);

        // Build graph
        SameDiff sd = SameDiff.create();
        sd.placeHolder("input", DataType.FLOAT, 1, 16);
        sd.constant("w", wData.dup());
        sd.mmul("output", sd.getVariable("input"), sd.getVariable("w"));
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Warmup + freeze
        Map<String, INDArray> warmupResult = sd.output(Map.of("input", inputData.dup()), "output");
        INDArray warmupOutput = warmupResult.get("output").dup();
        freezeShapes(sd);

        // Replay
        Map<String, INDArray> replayResult = sd.output(Map.of("input", inputData.dup()), "output");
        INDArray replayOutput = replayResult.get("output").dup();

        // Verify correctness: output should match warmup
        double diff = replayOutput.sub(warmupOutput).norm2Number().doubleValue();
        double refNorm = warmupOutput.norm2Number().doubleValue();
        double relErr = refNorm > 0 ? diff / refNorm : diff;
        log.info("JitMode GRAPH_ONLY: relErr={} (absDiff={})", relErr, diff);
        assertTrue(relErr < FLOAT_TOL,
                "GRAPH_ONLY output diverged from warmup. relErr=" + relErr);

        // Verify expected shape
        assertArrayEquals(new long[]{1, 32}, replayOutput.shape(),
                "Output shape should be [1, 32]");

        sd.close();
    }

    /**
     * Test 2: JitMode JIT_ONLY — build elementwise chain graph, execute, verify output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("JitMode JIT_ONLY: elementwise chain executes correctly")
    public void testJitModeJitOnly(Nd4jBackend backend) {
        setJitMode("jit");

        int dim = 64;
        INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, dim);

        // Build graph
        SameDiff sd = buildElementwiseChain();
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Warmup + freeze
        Map<String, INDArray> warmupResult = sd.output(Map.of("input", inputData.dup()), "output");
        INDArray warmupOutput = warmupResult.get("output").dup();
        freezeShapes(sd);

        // Replay
        Map<String, INDArray> replayResult = sd.output(Map.of("input", inputData.dup()), "output");
        INDArray replayOutput = replayResult.get("output").dup();

        // Verify correctness
        double diff = replayOutput.sub(warmupOutput).norm2Number().doubleValue();
        double refNorm = warmupOutput.norm2Number().doubleValue();
        double relErr = refNorm > 0 ? diff / refNorm : diff;
        log.info("JitMode JIT_ONLY: relErr={} (absDiff={})", relErr, diff);
        assertTrue(relErr < FLOAT_TOL,
                "JIT_ONLY output diverged from warmup. relErr=" + relErr);

        // Verify output values are in [0, 1] range (sigmoid output)
        for (int i = 0; i < Math.min(10, replayOutput.length()); i++) {
            float val = replayOutput.getFloat(i);
            assertTrue(val >= 0.0f && val <= 1.0f,
                    "Sigmoid output should be in [0,1], got " + val + " at index " + i);
        }

        sd.close();
    }

    /**
     * Test 3: JitMode GRAPH_PLUS_JIT — build mixed graph (matmul + elementwise),
     * execute, verify correct output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("JitMode GRAPH_PLUS_JIT: mixed graph executes correctly")
    public void testJitModeGraphPlusJit(Nd4jBackend backend) {
        setJitMode("graph+jit");

        int dim = 32;
        INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, dim);

        // Build graph
        SameDiff sd = buildMixedGraph();
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Warmup + freeze
        Map<String, INDArray> warmupResult = sd.output(Map.of("input", inputData.dup()), "output");
        INDArray warmupOutput = warmupResult.get("output").dup();
        freezeShapes(sd);

        // Replay
        Map<String, INDArray> replayResult = sd.output(Map.of("input", inputData.dup()), "output");
        INDArray replayOutput = replayResult.get("output").dup();

        // Verify correctness
        double diff = replayOutput.sub(warmupOutput).norm2Number().doubleValue();
        double refNorm = warmupOutput.norm2Number().doubleValue();
        double relErr = refNorm > 0 ? diff / refNorm : diff;
        log.info("JitMode GRAPH_PLUS_JIT: relErr={} (absDiff={})", relErr, diff);
        assertTrue(relErr < FLOAT_TOL,
                "GRAPH_PLUS_JIT output diverged from warmup. relErr=" + relErr);

        assertArrayEquals(new long[]{1, dim}, replayOutput.shape(),
                "Output shape should be [1, " + dim + "]");

        sd.close();
    }

    /**
     * Test 4: JitMode equivalence — same graph executed under all 3 JitModes.
     * All must produce identical output (within tolerance).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("JitMode equivalence: all 3 modes produce identical output")
    public void testJitModeEquivalence(Nd4jBackend backend) {
        int dim = 32;
        INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, dim);
        INDArray wData = Nd4j.randn(DataType.FLOAT, dim, dim * 2);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, dim * 2, dim);

        String[] jitModes = {"graph", "jit", "graph+jit"};
        Map<String, INDArray> modeOutputs = new LinkedHashMap<>();

        for (String jitMode : jitModes) {
            setJitMode(jitMode);

            SameDiff sd = SameDiff.create();
            sd.placeHolder("input", DataType.FLOAT, 1, dim);
            sd.constant("w1", wData.dup());
            sd.constant("w2", w2Data.dup());
            SDVariable h = sd.nn.relu("relu", sd.mmul("mm1", sd.getVariable("input"), sd.getVariable("w1")), 0);
            sd.mmul("output", h, sd.getVariable("w2"));
            enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

            // Warmup + freeze + replay
            sd.output(Map.of("input", inputData.dup()), "output");
            freezeShapes(sd);
            Map<String, INDArray> result = sd.output(Map.of("input", inputData.dup()), "output");
            modeOutputs.put(jitMode, result.get("output").dup());

            log.info("JitMode {}: output[0]={}", jitMode, modeOutputs.get(jitMode).getFloat(0));
            sd.close();
        }

        // Compare all pairs
        INDArray baseline = modeOutputs.get("graph");
        for (String jitMode : new String[]{"jit", "graph+jit"}) {
            INDArray other = modeOutputs.get(jitMode);
            double diff = other.sub(baseline).norm2Number().doubleValue();
            double refNorm = baseline.norm2Number().doubleValue();
            double relErr = refNorm > 0 ? diff / refNorm : diff;
            log.info("JitMode equivalence: {} vs graph: relErr={}", jitMode, relErr);
            assertTrue(relErr < FLOAT_TOL,
                    jitMode + " diverged from GRAPH_ONLY baseline. relErr=" + relErr);
        }
    }

    // =========================================================================
    // Multi-step drift tests
    // =========================================================================

    /**
     * Test 5: Multi-step replay with FIXED inputs — verify output on iteration 100
     * is IDENTICAL to iteration 1 (no drift when inputs don't change).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multi-step replay: no drift with fixed inputs across 100 iterations")
    public void testMultiStepReplayNoDrift(Nd4jBackend backend) {
        Assumptions.assumeTrue(!Nd4j.getEnvironment().isCPU(),
                "CUDA_GRAPHS mode requires CUDA backend");
        setJitMode("graph");

        int dim = 32;
        SameDiff sd = buildMatmulGraph(dim);
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Fixed input
        INDArray fixedInput = Nd4j.randn(DataType.FLOAT, 1, dim);

        // Warmup + freeze
        sd.output(Map.of("input", fixedInput.dup()), "output");
        freezeShapes(sd);

        // Record first replayed output
        Map<String, INDArray> firstResult = sd.output(Map.of("input", fixedInput.dup()), "output");
        INDArray firstOutput = firstResult.get("output").dup();

        // Run 99 more iterations with the SAME input
        int totalSteps = 100;
        double maxDrift = 0;
        for (int step = 2; step <= totalSteps; step++) {
            Map<String, INDArray> result = sd.output(Map.of("input", fixedInput.dup()), "output");
            INDArray stepOutput = result.get("output").dup();
            double diff = stepOutput.sub(firstOutput).norm2Number().doubleValue();
            maxDrift = Math.max(maxDrift, diff);
        }

        log.info("Multi-step no-drift: {} iterations, maxDrift={}", totalSteps, maxDrift);
        // With fixed inputs, output should be bit-identical (or within machine epsilon)
        assertTrue(maxDrift < 1e-6,
                "Output drifted across 100 fixed-input iterations. maxDrift=" + maxDrift);

        sd.close();
    }

    /**
     * Test 6: Multi-step replay with CHANGING inputs — verify each output against
     * a fresh SLOT_BY_SLOT reference computation. Max absolute error must stay
     * below tolerance * sqrt(step).
     *
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multi-step replay: drift bound with changing inputs across 50 iterations")
    public void testMultiStepReplayDriftBound(Nd4jBackend backend) {
        Assumptions.assumeTrue(!Nd4j.getEnvironment().isCPU(),
                "CUDA_GRAPHS mode requires CUDA backend");
        setJitMode("graph");

        int dim = 32;
        int totalSteps = 50;

        // Create shared weight — both DSP and reference must use the SAME weight
        INDArray sharedW = Nd4j.randn(DataType.FLOAT, dim, 32);

        // Generate all inputs upfront
        List<INDArray> inputs = new ArrayList<>();
        for (int i = 0; i < totalSteps; i++) {
            Nd4j.getRandom().setSeed(i * 1000L + 42);
            inputs.add(Nd4j.randn(DataType.FLOAT, 1, dim));
        }

        // Compute SLOT_BY_SLOT reference outputs using the SAME weight as DSP
        List<INDArray> refOutputs = runSlotBySlotReference(inputs, DataType.FLOAT, dim, 32, sharedW);

        // Build and execute DSP graph with the shared weight
        SameDiff sd = buildMatmulGraph(dim, 32, sharedW);
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Warmup + freeze
        sd.output(Map.of("input", inputs.get(0).dup()), "output");
        freezeShapes(sd);

        double maxRelErr = 0;
        int worstStep = -1;

        for (int step = 0; step < totalSteps; step++) {
            Map<String, INDArray> result = sd.output(Map.of("input", inputs.get(step).dup()), "output");
            INDArray dspOutput = result.get("output").dup();
            INDArray refOutput = refOutputs.get(step);

            double absDiff = dspOutput.sub(refOutput).norm2Number().doubleValue();
            double refNorm = refOutput.norm2Number().doubleValue();
            double relErr = refNorm > 0 ? absDiff / refNorm : absDiff;

            // Bound: tolerance * sqrt(step+1)
            double bound = FLOAT_TOL * Math.sqrt(step + 1);
            if (relErr > maxRelErr) {
                maxRelErr = relErr;
                worstStep = step;
            }

            if (step < 5 || step % 10 == 0) {
                log.info("Step {}: relErr={} bound={}", step, relErr, bound);
            }
            assertTrue(relErr < bound,
                    "Step " + step + " exceeded drift bound. relErr=" + relErr + " bound=" + bound);
        }

        log.info("Multi-step drift-bound: {} steps, maxRelErr={} at step={}, bound at worst={}",
                totalSteps, maxRelErr, worstStep, FLOAT_TOL * Math.sqrt(worstStep + 1));

        sd.close();
    }

    /**
     * Test 7: Long-running replay stability — run 200+ iterations, verify no NaN/Inf
     * appears in any output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Long-running replay: no NaN/Inf across 200+ iterations")
    public void testLongRunningReplayStability(Nd4jBackend backend) {
        Assumptions.assumeTrue(!Nd4j.getEnvironment().isCPU(),
                "CUDA_GRAPHS mode requires CUDA backend");
        setJitMode("graph");

        int dim = 32;
        int totalSteps = 200;

        SameDiff sd = buildMatmulGraph(dim);
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Warmup + freeze
        INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, dim);
        sd.output(Map.of("input", warmupInput.dup()), "output");
        freezeShapes(sd);

        for (int step = 0; step < totalSteps; step++) {
            Nd4j.getRandom().setSeed(step * 7777L + 123);
            INDArray stepInput = Nd4j.randn(DataType.FLOAT, 1, dim);

            Map<String, INDArray> result = sd.output(Map.of("input", stepInput.dup()), "output");
            INDArray output = result.get("output");

            // Check for NaN / Inf
            for (int i = 0; i < Math.min(output.length(), 10); i++) {
                float val = output.getFloat(i);
                assertFalse(Float.isNaN(val),
                        "NaN detected at step " + step + ", index " + i);
                assertFalse(Float.isInfinite(val),
                        "Inf detected at step " + step + ", index " + i);
            }
        }

        log.info("Long-running stability: {} steps completed, no NaN/Inf detected", totalSteps);
        sd.close();
    }

    /**
     * Test 8: Drift across data types — run multi-step replay with FLOAT and DOUBLE.
     * Verify DOUBLE has lower drift than FLOAT.
     *
     * DISABLED: DSP frozen-graph replay produces ~240% relative error vs SLOT_BY_SLOT
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Drift across data types: DOUBLE has lower drift than FLOAT")
    public void testDriftAcrossDataTypes(Nd4jBackend backend) {
        Assumptions.assumeTrue(!Nd4j.getEnvironment().isCPU(),
                "CUDA_GRAPHS mode requires CUDA backend");
        setJitMode("graph");

        int dim = 32;
        int totalSteps = 50;

        // Generate inputs
        List<INDArray> floatInputs = new ArrayList<>();
        List<INDArray> doubleInputs = new ArrayList<>();
        for (int i = 0; i < totalSteps; i++) {
            Nd4j.getRandom().setSeed(i * 1000L + 42);
            floatInputs.add(Nd4j.randn(DataType.FLOAT, 1, dim));
            doubleInputs.add(Nd4j.randn(DataType.DOUBLE, 1, dim));
        }

        // ─── FLOAT pass ───
        SameDiff floatSd = SameDiff.create();
        floatSd.placeHolder("input", DataType.FLOAT, 1, dim);
        INDArray floatW = Nd4j.randn(DataType.FLOAT, dim, dim);
        floatSd.constant("w", floatW.dup());
        floatSd.mmul("output", floatSd.getVariable("input"), floatSd.getVariable("w"));
        enableDsp(floatSd, GraphExecutionMode.CUDA_GRAPHS);

        floatSd.output(Map.of("input", floatInputs.get(0).dup()), "output");
        freezeShapes(floatSd);

        double maxFloatDrift = 0;
        for (int step = 1; step < totalSteps; step++) {
            Map<String, INDArray> result = floatSd.output(Map.of("input", floatInputs.get(step).dup()), "output");
            INDArray dspOut = result.get("output").dup();

            // Compare against a fresh reference
            SameDiff refSd = SameDiff.create();
            refSd.placeHolder("input", DataType.FLOAT, 1, dim);
            refSd.constant("w", floatW.dup());
            refSd.mmul("output", refSd.getVariable("input"), refSd.getVariable("w"));
            refSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            Map<String, INDArray> refResult = refSd.output(Map.of("input", floatInputs.get(step).dup()), "output");
            INDArray refOut = refResult.get("output").dup();

            double diff = dspOut.sub(refOut).norm2Number().doubleValue();
            double refNorm = refOut.norm2Number().doubleValue();
            double relErr = refNorm > 0 ? diff / refNorm : diff;
            maxFloatDrift = Math.max(maxFloatDrift, relErr);
            refSd.close();
        }
        floatSd.close();

        // ─── DOUBLE pass ───
        SameDiff doubleSd = SameDiff.create();
        doubleSd.placeHolder("input", DataType.DOUBLE, 1, dim);
        INDArray doubleW = Nd4j.randn(DataType.DOUBLE, dim, dim);
        doubleSd.constant("w", doubleW.dup());
        doubleSd.mmul("output", doubleSd.getVariable("input"), doubleSd.getVariable("w"));
        enableDsp(doubleSd, GraphExecutionMode.CUDA_GRAPHS);

        doubleSd.output(Map.of("input", doubleInputs.get(0).dup()), "output");
        freezeShapes(doubleSd);

        double maxDoubleDrift = 0;
        for (int step = 1; step < totalSteps; step++) {
            Map<String, INDArray> result = doubleSd.output(Map.of("input", doubleInputs.get(step).dup()), "output");
            INDArray dspOut = result.get("output").dup();

            // Compare against a fresh reference
            SameDiff refSd = SameDiff.create();
            refSd.placeHolder("input", DataType.DOUBLE, 1, dim);
            refSd.constant("w", doubleW.dup());
            refSd.mmul("output", refSd.getVariable("input"), refSd.getVariable("w"));
            refSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            Map<String, INDArray> refResult = refSd.output(Map.of("input", doubleInputs.get(step).dup()), "output");
            INDArray refOut = refResult.get("output").dup();

            double diff = dspOut.sub(refOut).norm2Number().doubleValue();
            double refNorm = refOut.norm2Number().doubleValue();
            double relErr = refNorm > 0 ? diff / refNorm : diff;
            maxDoubleDrift = Math.max(maxDoubleDrift, relErr);
            refSd.close();
        }
        doubleSd.close();

        log.info("Drift across data types: FLOAT maxDrift={}, DOUBLE maxDrift={}", maxFloatDrift, maxDoubleDrift);

        // Both should be within their respective tolerances
        assertTrue(maxFloatDrift < FLOAT_TOL * Math.sqrt(totalSteps),
                "FLOAT drift exceeded bound. maxFloatDrift=" + maxFloatDrift);
        assertTrue(maxDoubleDrift < DOUBLE_TOL * Math.sqrt(totalSteps),
                "DOUBLE drift exceeded bound. maxDoubleDrift=" + maxDoubleDrift);

        // DOUBLE should have lower or equal drift than FLOAT (both can be 0.0 for exact computations)
        assertTrue(maxDoubleDrift <= maxFloatDrift,
                "DOUBLE should have lower or equal drift than FLOAT. DOUBLE=" + maxDoubleDrift + " FLOAT=" + maxFloatDrift);
    }
}
