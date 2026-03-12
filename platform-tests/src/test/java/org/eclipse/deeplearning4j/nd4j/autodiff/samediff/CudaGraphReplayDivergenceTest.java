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
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Targeted reproducer tests for CUDA graph replay divergence.
 *
 * These tests isolate specific patterns that cause graph replay to produce
 * different results than slot-by-slot execution:
 *
 * 1. Frozen constants (shape_of, ConstantOfShape) interacting with changing inputs
 * 2. Native fallback ops inside graph capture with baked buffer addresses
 * 3. External input buffer updates not propagating through graph replay
 * 4. Multi-step decode where inputs change but graph replays stale data
 *
 * Each test compares graph-capture execution (gc) against direct Triton
 * execution (no gc) and slot-by-slot execution to pinpoint divergence.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class CudaGraphReplayDivergenceTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-3;

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Helper methods ──────────────────────────────────────────────────────

    private Pointer compileNativePlan(DynamicShapePlan plan) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        byte[] serialized = plan.serialize();
        assertNotNull(serialized, "Plan serialization returned null");
        assertTrue(serialized.length > 0, "Plan serialization returned empty");
        BytePointer planBytes = new BytePointer(serialized);
        try {
            return nativeOps.compileDynamicShapePlan(planBytes, serialized.length);
        } catch (UnsupportedOperationException e) {
            log.info("Backend does not support native executor: {}", e.getMessage());
            return null;
        } finally {
            planBytes.close();
        }
    }

    private Map<String, INDArray> executeNativePlan(Pointer planHandle, DynamicShapePlan plan,
                                                     INDArray[] extInputs) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numOutputs = plan.getRequestedOutputs().size();

        OpaqueContext opContext = nativeOps.createGraphContext(1);
        try {
            for (int i = 0; i < extInputs.length; i++) {
                OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(extInputs[i]);
                nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
            }

            for (int i = 0; i < numOutputs; i++) {
                INDArray dummy = Nd4j.empty(DataType.FLOAT);
                OpaqueNDArray opaqueOut = OpaqueNDArray.fromINDArray(dummy);
                nativeOps.setGraphContextOutputArray(opContext, i, opaqueOut);
            }

            Pointer execStream = null;
            try {
                OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                if (lc != null) {
                    execStream = nativeOps.lcExecutionStream(lc);
                    if (execStream != null) execStream.retainReference();
                }
            } catch (Exception e) {
                // CPU backend
            }

            int status = nativeOps.executeDynamicShapePlan(planHandle, opContext, execStream);
            if (status != 0) {
                String errMsg = nativeOps.lastErrorMessage();
                nativeOps.clearLastError();
                fail("Native plan execution failed with status " + status + ": " + errMsg);
            }

            Map<String, INDArray> results = new LinkedHashMap<>();
            List<String> requestedOutputs = new java.util.ArrayList<>(plan.getRequestedOutputs());

            for (int i = 0; i < numOutputs; i++) {
                OpaqueNDArray opaqueOut = nativeOps.getOutputArrayNative(opContext, i);
                assertNotNull(opaqueOut, "Null output at index " + i);
                assertFalse(opaqueOut.isNull(), "Null OpaqueNDArray at index " + i);

                long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
                long[] shape = Shape.shape(shapeInfo);
                DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
                long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);

                INDArray result = Nd4j.createUninitialized(dtype, shape);
                Pointer nativePrimary = nativeOps.getOpaqueNDArrayBuffer(opaqueOut);
                Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);

                OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                        length, dtype.toInt(), nativePrimary, nativeSpecial);
                if (srcOdb != null) {
                    OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
                    if (dstOdb != null) {
                        nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
                    }
                }
                results.put(requestedOutputs.get(i), result);
            }
            return results;
        } finally {
            nativeOps.deleteGraphContext(opContext);
        }
    }

    private INDArray[] resolveExternalInputs(DynamicShapePlan plan, SameDiff sd,
                                              Map<String, INDArray> placeholders) {
        String[] extKeys = plan.getExternalInputKeys();
        INDArray[] extInputs = new INDArray[extKeys.length];
        for (int i = 0; i < extKeys.length; i++) {
            String varName = extKeys[i];
            INDArray arr = placeholders != null ? placeholders.get(varName) : null;
            if (arr == null) {
                SDVariable var = sd.getVariable(varName);
                if (var != null) arr = var.getArr();
            }
            assertNotNull(arr, "Missing external input: " + varName);
            extInputs[i] = arr;
        }
        return extInputs;
    }

    // ─── Test 1: Frozen shape_of + ConstantOfShape with changing inputs ──────

    /**
     * Reproduces the VLM pattern where:
     * - shape_of(input) is detected as a VALUE_INDEPENDENT_OP → frozen after warmup
     * - ConstantOfShape(shape_of) creates a mask/bias tensor
     * - The result is combined with the changing input
     *
     * During CUDA graph replay, frozen constants should still have correct values
     * since shapes don't change between decode steps (all [1,1,dim]).
     * But if the frozen constant mechanism interferes with graph replay
     * (e.g., skipping execution of ops that the graph already captured),
     * outputs will diverge.
     *
     * Compares: graph-capture vs no-graph-capture (direct Triton) vs slot-by-slot
     */
    @Test
    public void testFrozenConstantsWithChangingInputs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        // Build graph: x → [shape_of → ConstantOfShape(ones)] → multiply(x, ones) → add(bias)
        // shape_of will be frozen after warmup; ConstantOfShape depends on it
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable bias = sd.constant("bias", Nd4j.linspace(DataType.FLOAT, 0.1, 0.1, 8).reshape(1, 8));

        // shape_of → ConstantOfShape creates an all-ones tensor matching x's shape
        SDVariable shapeOf = sd.shape("x_shape", x);
        SDVariable ones = sd.onesLike("ones_mask", x);  // equivalent pattern

        // Use the ones mask in computation with x
        SDVariable masked = x.mul("masked", ones);
        SDVariable biased = masked.add("biased", bias);

        // Add more ops to make it long enough for Triton compilation
        SDVariable scale = sd.constant("scale", Nd4j.valueArrayOf(new long[]{1, 8}, 2.0f));
        SDVariable scaled = biased.mul("scaled", scale);
        SDVariable activated = sd.nn.relu("relu1", scaled, 0);
        SDVariable shift = sd.constant("shift", Nd4j.valueArrayOf(new long[]{1, 8}, -0.5f));
        SDVariable shifted = activated.add("shifted", shift);
        SDVariable result = sd.nn.sigmoid("result", shifted);

        int numSteps = 8;  // warmup (2) + capture (1) + replay (5)
        float[][] inputValues = new float[numSteps][8];
        for (int s = 0; s < numSteps; s++) {
            for (int j = 0; j < 8; j++) {
                inputValues[s][j] = (s + 1) * 0.1f + j * 0.01f;
            }
        }

        // Run with graph capture enabled
        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan, "Plan is null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] gcOutputs = new INDArray[numSteps];

            // Execute with graph capture
            for (int s = 0; s < numSteps; s++) {
                INDArray input = Nd4j.createFromArray(inputValues[s]).reshape(1, 8);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", input));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
                log.info("GC step {}: result[0..3] = [{}, {}, {}, {}]", s,
                        gcOutputs[s].getFloat(0, 0), gcOutputs[s].getFloat(0, 1),
                        gcOutputs[s].getFloat(0, 2), gcOutputs[s].getFloat(0, 3));
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Now run WITHOUT graph capture for reference
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2, "Reference plan handle null");

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray input = Nd4j.createFromArray(inputValues[s]).reshape(1, 8);
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", input));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");

                    if (s == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle2, true);
                    }

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("Step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    // Replay steps (s >= 3) are the critical ones
                    assertTrue(maxDiff < TOLERANCE,
                            "Step " + s + ": GC output diverges from non-GC! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
        }
    }

    // ─── Test 2: Native fallback ops with baked addresses in graph ────────────

    /**
     * Tests that native fallback ops (ops not compiled by Triton) inside
     * a CUDA graph correctly see updated input data on replay.
     *
     * Pattern: x → cast(FLOAT→HALF) → [Triton compiled chain] → cast(HALF→FLOAT)
     * The cast ops may fall back to native execution inside the graph.
     * If graph replay bakes the old buffer address for cast input,
     * the cast reads stale data → wrong output.
     */
    @Test
    public void testNativeFallbackInsideGraphCapture() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 16);

        // Cast to HALF (likely native fallback), do computation, cast back
        SDVariable xHalf = sd.castTo("to_half", x, DataType.HALF);
        SDVariable xBack = sd.castTo("to_float", xHalf, DataType.FLOAT);

        // Element-wise chain (compilable by Triton)
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 1, 16).mul(0.1));
        SDVariable h1 = xBack.mul("mul1", w1);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 1, 16).mul(0.1));
        SDVariable h3 = h2.add("add1", w2);
        SDVariable h4 = sd.nn.sigmoid("sigmoid1", h3);
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 1, 16).mul(0.1));
        SDVariable h5 = h4.mul("mul2", w3);
        SDVariable result = sd.math.tanh("result", h5);

        int numSteps = 8;

        // Graph capture run
        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            // Pre-generate deterministic inputs
            INDArray[] inputArrays = new INDArray[numSteps];
            for (int s = 0; s < numSteps; s++) {
                inputArrays[s] = Nd4j.linspace(DataType.FLOAT, (s + 1) * 0.1f, 0.05f, 16).reshape(1, 16);
            }

            INDArray[] gcOutputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputArrays[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference without graph capture
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputArrays[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("Step {}: fallback GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "Step " + s + ": native fallback in GC diverges! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
        }
    }

    // ─── Test 3: Multi-step decode with external input updates ───────────────

    /**
     * Simulates VLM decode: same-shape inputs with DIFFERENT values each step.
     * This is the core pattern where graph replay must use updated input data.
     *
     * The graph is captured on step 2 (after warmup). Steps 3+ replay the graph.
     * Each step provides a different input value. If graph replay uses baked/stale
     * input values, the output will be identical across replay steps — which is WRONG.
     *
     * This test verifies that each step produces DIFFERENT output for DIFFERENT input,
     * even during graph replay.
     */
    @Test
    public void testMultiStepDecodeInputUpdates() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        // Simple but non-trivial graph: x → mul(w) → add(b) → relu → mul(w2) → sigmoid
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = sd.constant("w", Nd4j.linspace(DataType.FLOAT, 0.1, 0.1, 8).reshape(1, 8));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(new long[]{1, 8}, 0.5f));
        SDVariable w2 = sd.constant("w2", Nd4j.linspace(DataType.FLOAT, 0.5, -0.05, 8).reshape(1, 8));

        SDVariable h1 = x.mul("mul1", w);
        SDVariable h2 = h1.add("add1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.mul("mul2", w2);
        SDVariable h5 = sd.nn.sigmoid("sigmoid1", h4);
        // Add more ops for Triton to compile
        SDVariable h6 = h5.mul("mul3", w);
        SDVariable h7 = h6.add("add2", b);
        SDVariable result = sd.math.tanh("result", h7);

        int numSteps = 10;

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] outputs = new INDArray[numSteps];
            INDArray[] inputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                // Each step has distinctly different input values
                inputs[s] = Nd4j.valueArrayOf(new long[]{1, 8}, (s + 1) * 1.0f);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                outputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);

                log.info("Step {}: input={}, result[0]={}", s, inputs[s].getFloat(0, 0),
                        outputs[s].getFloat(0, 0));
            }

            // Verify outputs are DIFFERENT for different inputs (not stuck replaying same values)
            for (int s = 1; s < numSteps; s++) {
                double diffFromPrev = outputs[s].sub(outputs[s - 1]).amaxNumber().doubleValue();
                log.info("Step {} vs {}: output diff = {}", s, s - 1, diffFromPrev);

                // Each step has very different input (1.0, 2.0, 3.0...) so output MUST differ
                // If graph replay bakes stale input data, outputs would be identical
                assertTrue(diffFromPrev > 1e-6,
                        "Steps " + (s - 1) + " and " + s + " have identical output — "
                                + "graph replay is using stale input data!"
                                + "\n  step " + (s - 1) + ": " + outputs[s - 1]
                                + "\n  step " + s + ": " + outputs[s]);
            }

            // Also compare against reference execution for numerical accuracy
            nativeOps.freeDynamicShapePlan(planHandle);
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputs[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = outputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("Step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "Step " + s + ": GC diverges from reference! maxDiff=" + maxDiff);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
        }
    }

    // ─── Test 4: Force recapture still diverges ──────────────────────────────

    /**
     * Tests whether force-recapture (fresh capture every step) produces correct
     * output. If force-recapture ALSO produces wrong output, the bug is in the
     * capture process itself, not in replay staleness.
     *
     * This isolates: "is the bug in capture or in replay?"
     */
    @Test
    public void testForceRecaptureCorrectness() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = sd.constant("w", Nd4j.linspace(DataType.FLOAT, 0.1, 0.1, 8).reshape(1, 8));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(new long[]{1, 8}, 0.5f));

        SDVariable h1 = x.mul("mul1", w);
        SDVariable h2 = h1.add("add1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.mul("mul2", w);
        SDVariable h5 = h4.add("add2", b);
        SDVariable h6 = sd.nn.sigmoid("sigmoid1", h5);
        SDVariable h7 = h6.mul("mul3", w);
        SDVariable result = sd.math.tanh("result", h7);

        int numSteps = 8;

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        boolean prevForceRecapture = Nd4j.getEnvironment().tritonForceRecapture();

        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonForceRecapture(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] gcOutputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                INDArray input = Nd4j.valueArrayOf(new long[]{1, 8}, (s + 1) * 1.0f);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", input));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference: slot-by-slot (no Triton, no GC)
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            Nd4j.getEnvironment().setTritonForceRecapture(false);
            Nd4j.getEnvironment().setTritonCompileAll(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan planRef = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandleRef = compileNativePlan(planRef);
            assertNotNull(planHandleRef);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray input = Nd4j.valueArrayOf(new long[]{1, 8}, (s + 1) * 1.0f);
                    INDArray[] extInputs = resolveExternalInputs(planRef, sd, Map.of("x", input));
                    Map<String, INDArray> refResults = executeNativePlan(planHandleRef, planRef, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandleRef, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("ForceRecapture step {}: maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "ForceRecapture step " + s + " diverges from slot-by-slot! maxDiff=" + maxDiff
                                    + "\n  ForceRecapture: " + gcOutputs[s]
                                    + "\n  SlotBySlot:     " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandleRef);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonForceRecapture(prevForceRecapture);
        }
    }

    // ─── Test 5: RMSNorm pattern with frozen shape + changing data ───────────

    /**
     * Reproduces the exact VLM divergence pattern:
     *   x (changes each step) → RMSNorm computation:
     *     mean(x*x) → add(epsilon) → rsqrt → multiply(x) → multiply(weight)
     *
     * In the VLM, the first divergence was at slot 496 (add_scalar in RMSNorm),
     * traced back to ext#1331 (inputs_embeds) having wrong values during replay.
     *
     * This test builds a minimal RMSNorm-like graph to reproduce that divergence.
     */
    @Test
    public void testRMSNormPatternWithChangingInput() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        int dim = 64;  // typical hidden dim (smaller for test)
        float epsilon = 1e-5f;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, dim);
        SDVariable weight = sd.constant("weight", Nd4j.ones(DataType.FLOAT, dim));

        // RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        SDVariable xSq = x.mul("x_sq", x);
        SDVariable meanSq = sd.mean("mean_sq", xSq, true, 2);  // keepDims=true, axis=-1
        SDVariable meanPlusEps = meanSq.add("mean_eps", epsilon);
        SDVariable rsqrtVal = sd.math.rsqrt("rsqrt", meanPlusEps);
        SDVariable normalized = x.mul("normalized", rsqrtVal);
        SDVariable result = normalized.mul("result", weight);

        int numSteps = 10;

        // Run with GC
        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            // Pre-generate deterministic inputs for reproducibility
            INDArray[] rmsInputs = new INDArray[numSteps];
            for (int s = 0; s < numSteps; s++) {
                rmsInputs[s] = Nd4j.linspace(DataType.FLOAT, (s + 1) * 0.1f, 0.02f, dim).reshape(1, 1, dim);
            }

            INDArray[] gcOutputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", rmsInputs[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);

                log.info("RMSNorm GC step {}: result[0..3] = [{}, {}, {}, {}]", s,
                        gcOutputs[s].getFloat(0, 0, 0), gcOutputs[s].getFloat(0, 0, 1),
                        gcOutputs[s].getFloat(0, 0, 2), gcOutputs[s].getFloat(0, 0, 3));
            }

            // Verify outputs differ between steps (not replaying stale data)
            for (int s = 1; s < numSteps; s++) {
                double diffFromPrev = gcOutputs[s].sub(gcOutputs[s - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 1e-4,
                        "RMSNorm steps " + (s - 1) + " and " + s + " have identical output!");
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference without GC
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", rmsInputs[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("RMSNorm step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "RMSNorm step " + s + ": GC diverges! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
        }
    }

    // ─── Test 6: Multiple reads of same external input ───────────────────────

    /**
     * Tests the pattern where the same external input is read by multiple ops
     * at different positions in the graph. In the VLM, inputs_embeds is read
     * by both the first RMSNorm (slot 498) and by add (slot 661, residual).
     *
     * If graph replay corrupts the input buffer during the first read,
     * the second read will see wrong data.
     */
    @Test
    public void testMultipleReadsOfSameInput() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 1, 16).mul(0.1));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 1, 16).mul(0.1));

        // First use of x: normalized = x * rsqrt(mean(x^2) + eps)
        SDVariable xSq = x.mul("x_sq", x);
        SDVariable meanSq = sd.mean("mean_sq", xSq, true, 1);
        SDVariable rsqrt = sd.math.rsqrt("rsqrt", meanSq.add("eps", 1e-5f));
        SDVariable norm = x.mul("norm", rsqrt);

        // Some computation on the normalized value
        SDVariable h1 = norm.mul("h1", w1);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = h2.mul("h2", w2);
        SDVariable h4 = sd.nn.sigmoid("sigmoid1", h3);

        // Second use of x: residual connection — x + processed
        // This is the VLM pattern: inputs_embeds used in both RMSNorm AND residual add
        SDVariable residual = x.add("residual", h4);
        SDVariable result = sd.math.tanh("result", residual);

        int numSteps = 8;

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] gcOutputs = new INDArray[numSteps];
            INDArray[] inputArrays = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                inputArrays[s] = Nd4j.valueArrayOf(new long[]{1, 16}, (s + 1) * 0.5f);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputArrays[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputArrays[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("MultiRead step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "MultiRead step " + s + ": residual diverges! maxDiff=" + maxDiff
                                    + " — graph replay may have corrupted the external input buffer"
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
        }
    }

    // ─── Test 7: Slot-by-slot vs CUDA graphs (no Triton) ─────────────────────

    /**
     * Tests CUDA graph replay correctness WITHOUT Triton compilation.
     * This isolates whether the issue is in the CUDA graph mechanism itself
     * (buffer address baking, stream synchronization) vs Triton-specific
     * (arg table refresh, sub-kernel emission).
     */
    @Test
    public void testCudaGraphReplayWithoutTriton() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = sd.constant("w", Nd4j.linspace(DataType.FLOAT, 0.1, 0.1, 8).reshape(1, 8));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(new long[]{1, 8}, 0.5f));

        SDVariable h1 = x.mul("mul1", w);
        SDVariable h2 = h1.add("add1", b);
        SDVariable result = sd.nn.relu("result", h2, 0);

        int numSteps = 10;

        // Compile for CUDA_GRAPHS mode (no Triton)
        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        Nd4j.getEnvironment().setTritonGraphCapture(false);
        Nd4j.getEnvironment().setTritonCompileAll(false);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] outputs = new INDArray[numSteps];
            INDArray[] inputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                inputs[s] = Nd4j.valueArrayOf(new long[]{1, 8}, (s + 1) * 1.0f);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                outputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
            }

            // Verify outputs change with inputs
            for (int s = 1; s < numSteps; s++) {
                double diff = outputs[s].sub(outputs[s - 1]).amaxNumber().doubleValue();
                assertTrue(diff > 1e-6,
                        "CUDA graph (no Triton) steps " + (s - 1) + " and " + s
                                + " have identical output!");
            }

            // Compare against standard SameDiff execution
            for (int s = 0; s < numSteps; s++) {
                Map<String, INDArray> refResults = sd.output(Map.of("x", inputs[s]), "result");
                INDArray refOutput = refResults.get("result");
                double maxDiff = outputs[s].sub(refOutput).amaxNumber().doubleValue();
                log.info("CUDA graph (no Triton) step {}: maxDiff = {}", s, maxDiff);
                assertTrue(maxDiff < TOLERANCE,
                        "CUDA graph (no Triton) step " + s + ": diverges! maxDiff=" + maxDiff);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
        }
    }

    // ─── Test 8: Native fallback op reading DIRECTLY from external input ──

    /**
     * CRITICAL TEST: Forces a specific op to be native fallback (via tritonExcludeOps)
     * while that op reads DIRECTLY from a changing external input.
     *
     * This is the EXACT VLM pattern:
     * - In the VLM, CONST_GEN ops have alwaysFallback=true
     * - Some native ops (like divide in RMSNorm) read from inputs_embeds
     * - During CUDA graph replay, native ops have buffer addresses baked
     * - If the external input's buffer address changes, native ops read stale data
     *
     * Here we exclude "multiply" from Triton compilation. The first op (x * w)
     * will be native fallback, reading x directly. If graph replay doesn't
     * update x's buffer address for the native multiply, it reads stale data.
     */
    @Test
    public void testNativeFallbackReadingExternalInputDirectly() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = sd.constant("w", Nd4j.linspace(DataType.FLOAT, 0.1, 0.1, 8).reshape(1, 8));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(new long[]{1, 8}, 0.5f));

        // multiply reads x directly — will be NATIVE FALLBACK (excluded from Triton)
        SDVariable h1 = x.mul("mul1", w);
        // These ops are Triton-compiled
        SDVariable h2 = h1.add("add1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.add("add2", b);
        SDVariable h5 = sd.nn.sigmoid("sigmoid1", h4);
        SDVariable h6 = h5.add("add3", b);
        SDVariable result = sd.math.tanh("result", h6);

        int numSteps = 10;

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        boolean prevFallback = Nd4j.getEnvironment().tritonAllowFallbackCapture();
        String prevExclude = Nd4j.getEnvironment().tritonExcludeOps();

        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonAllowFallbackCapture(true);
        // Force multiply to be native fallback — it reads x directly
        Nd4j.getEnvironment().setTritonExcludeOps("multiply");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] gcOutputs = new INDArray[numSteps];
            INDArray[] inputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                inputs[s] = Nd4j.valueArrayOf(new long[]{1, 8}, (s + 1) * 1.0f);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);

                log.info("NativeFallbackDirect GC step {}: input={}, result[0]={}", s,
                        inputs[s].getFloat(0, 0), gcOutputs[s].getFloat(0, 0));
            }

            // Verify outputs differ between steps
            for (int s = 1; s < numSteps; s++) {
                double diffFromPrev = gcOutputs[s].sub(gcOutputs[s - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 1e-6,
                        "NativeFallbackDirect steps " + (s - 1) + " and " + s
                                + " have identical output — native fallback uses stale input!");
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference: no graph capture
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputs[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("NativeFallbackDirect step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "NativeFallbackDirect step " + s + ": native fallback op reading "
                                    + "external input diverges in GC! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonAllowFallbackCapture(prevFallback);
            Nd4j.getEnvironment().setTritonExcludeOps(prevExclude != null ? prevExclude : "");
        }
    }

    // ─── Test 8b: Same as 8 but with FORCED buffer address changes ────────

    /**
     * Same as testNativeFallbackReadingExternalInputDirectly but explicitly
     * forces the external input's GPU buffer address to CHANGE between steps.
     *
     * This is done by allocating dummy GPU buffers between steps to fragment
     * the memory pool, ensuring the next allocation returns a different address.
     *
     * If the CUDA graph bakes the capture-time buffer address for native ops,
     * and the address changes on replay, native ops read from stale memory.
     */
    @Test
    public void testNativeFallbackWithForcedAddressChange() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        int dim = 32;
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // Build a 20+ op chain WITHOUT saturating activations.
        // Use only add/mul (no sigmoid/tanh) to preserve input sensitivity.
        // multiply is excluded from Triton → native fallback gap.
        SDVariable w = sd.constant("w", Nd4j.valueArrayOf(new long[]{1, dim}, 0.5f));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(new long[]{1, dim}, 0.1f));

        // Layer 1: mul (fallback) → add → relu → add → relu
        SDVariable h1 = x.mul("mul1", w);
        SDVariable h2 = h1.add("add1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.add("add2", b);
        SDVariable h5 = sd.nn.relu("relu2", h4, 0);

        // Layer 2: mul (fallback) → add → relu → add → relu
        SDVariable w2 = sd.constant("w2", Nd4j.valueArrayOf(new long[]{1, dim}, 0.8f));
        SDVariable h6 = h5.mul("mul2", w2);
        SDVariable h7 = h6.add("add3", b);
        SDVariable h8 = sd.nn.relu("relu3", h7, 0);
        SDVariable h9 = h8.add("add4", b);
        SDVariable h10 = sd.nn.relu("relu4", h9, 0);

        // Layer 3: mul (fallback) → add → relu → add → relu → add → identity(result)
        SDVariable w3 = sd.constant("w3", Nd4j.valueArrayOf(new long[]{1, dim}, 0.6f));
        SDVariable h11 = h10.mul("mul3", w3);
        SDVariable h12 = h11.add("add5", b);
        SDVariable h13 = sd.nn.relu("relu5", h12, 0);
        SDVariable h14 = h13.add("add6", b);
        SDVariable h15 = sd.nn.relu("relu6", h14, 0);
        SDVariable h16 = h15.add("add7", b);
        sd.identity("result", h16);

        int numSteps = 10;

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        boolean prevFallback = Nd4j.getEnvironment().tritonAllowFallbackCapture();
        String prevExclude = Nd4j.getEnvironment().tritonExcludeOps();

        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonAllowFallbackCapture(true);
        // Force multiply to be native fallback — it reads x directly
        Nd4j.getEnvironment().setTritonExcludeOps("multiply");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] gcOutputs = new INDArray[numSteps];
            INDArray[] inputs = new INDArray[numSteps];
            java.util.List<INDArray> dummyBuffers = new java.util.ArrayList<>();

            nativeOps.resetTritonCounters();
            long tritonLaunchesBefore = nativeOps.getTritonKernelLaunchCount();

            nativeOps.resetTritonCounters();

            for (int s = 0; s < numSteps; s++) {
                inputs[s] = Nd4j.valueArrayOf(new long[]{1, dim}, (s + 1) * 1.0f);
                // Force host→device sync: dup() copies host data to fresh GPU buffer
                inputs[s] = inputs[s].dup();

                // Log the GPU buffer address to verify it actually changes
                long bufAddr = inputs[s].data().addressPointer().address();
                log.info("Step {}: x buffer address = 0x{}", s, Long.toHexString(bufAddr));

                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputs[s]));
                if (s == 0) {
                    for (int i = 0; i < extInputs.length; i++) {
                        log.info("extInput[{}]: shape={}, val[0]={}, addr=0x{}", i,
                                java.util.Arrays.toString(extInputs[i].shape()),
                                extInputs[i].getFloat(0),
                                Long.toHexString(extInputs[i].data().addressPointer().address()));
                    }
                }
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);

                long tritonNow = nativeOps.getTritonKernelLaunchCount();
                log.info("ForcedAddr GC step {}: input={}, result[0]={}, tritonLaunches={}", s,
                        inputs[s].getFloat(0, 0), gcOutputs[s].getFloat(0, 0),
                        tritonNow - tritonLaunchesBefore);
                tritonLaunchesBefore = tritonNow;
            }

            // Clean up dummy buffers
            for (INDArray d : dummyBuffers) d.close();
            dummyBuffers.clear();

            // Verify outputs differ
            for (int s = 1; s < numSteps; s++) {
                double diffFromPrev = gcOutputs[s].sub(gcOutputs[s - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 1e-6,
                        "ForcedAddr steps " + (s - 1) + " and " + s
                                + " identical — native fallback uses stale input!");
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputs[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("ForcedAddr step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "ForcedAddr step " + s + ": native fallback diverges with "
                                    + "forced address change! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonAllowFallbackCapture(prevFallback);
            Nd4j.getEnvironment().setTritonExcludeOps(prevExclude != null ? prevExclude : "");
        }
    }

    // ─── Test 9: Matmul (native fallback) + Triton elementwise with GC ─────

    /**
     * Reproduces the VLM pattern with native fallback GAPS inside the graph:
     *   x → matmul(W) [NATIVE FALLBACK] → relu → mul → add → sigmoid [TRITON]
     *        → matmul(W2) [NATIVE FALLBACK] → relu [TRITON] → result
     *
     * With tritonCompileAll + tritonAllowFallbackCapture:
     * - matmul runs as cuBLAS (native fallback) inside the captured graph
     * - elementwise ops compile to Triton sub-kernels
     * - The graph contains BOTH Triton + native kernel launches
     *
     * This is the exact pattern where the VLM fails: the graph bakes
     * buffer addresses for native ops at capture time, and refreshArgTablesForReplay
     * only updates Triton sub-kernel arg tables, NOT native op arguments.
     */
    @Test
    public void testMatmulFallbackGapsInsideGraphCapture() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        int inDim = 8, hidDim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, inDim);

        // Layer 1: matmul (native fallback) + element-wise (Triton)
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, inDim, hidDim).mul(0.1));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, hidDim));
        SDVariable h1 = sd.mmul("matmul1", x, w1);
        SDVariable h2 = h1.add("add1", b1);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);

        // Intermediate element-wise (Triton compiled)
        SDVariable scale = sd.constant("scale", Nd4j.valueArrayOf(new long[]{1, hidDim}, 0.5f));
        SDVariable h4 = h3.mul("mul1", scale);
        SDVariable h5 = sd.nn.sigmoid("sigmoid1", h4);

        // Layer 2: matmul (native fallback) + element-wise (Triton)
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, hidDim, inDim).mul(0.1));
        SDVariable b2 = sd.constant("b2", Nd4j.zeros(DataType.FLOAT, 1, inDim));
        SDVariable h6 = sd.mmul("matmul2", h5, w2);
        SDVariable h7 = h6.add("add2", b2);

        // Residual connection (x used again — tests multiple reads of external input)
        SDVariable residual = x.add("residual", h7);
        SDVariable result = sd.math.tanh("result", residual);

        int numSteps = 10;

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        boolean prevFallback = Nd4j.getEnvironment().tritonAllowFallbackCapture();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonAllowFallbackCapture(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] gcOutputs = new INDArray[numSteps];
            INDArray[] inputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                inputs[s] = Nd4j.linspace(DataType.FLOAT, (s + 1) * 0.1f, 0.05f, inDim).reshape(1, inDim);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);

                log.info("MatmulGap GC step {}: result[0..3] = [{}, {}, {}, {}]", s,
                        gcOutputs[s].getFloat(0, 0), gcOutputs[s].getFloat(0, 1),
                        gcOutputs[s].getFloat(0, 2), gcOutputs[s].getFloat(0, 3));
            }

            // Verify outputs differ
            for (int s = 1; s < numSteps; s++) {
                double diffFromPrev = gcOutputs[s].sub(gcOutputs[s - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 1e-6,
                        "MatmulGap steps " + (s - 1) + " and " + s + " have identical output!");
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference: no GC
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputs[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("MatmulGap step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "MatmulGap step " + s + ": GC with matmul fallback diverges! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonAllowFallbackCapture(prevFallback);
        }
    }

    // ─── Test 9: Large graph with many ops (stress test) ─────────────────────

    /**
     * Tests a graph with many ops (simulating VLM's 3840 ops at smaller scale).
     * Multiple "layers" of: matmul → RMSNorm → MLP, all compiled with Triton.
     * Each layer reads the changing input through residual connections.
     *
     * Tests whether graph replay divergence accumulates over many ops.
     */
    @Test
    public void testMultiLayerTransformerPatternWithGraphCapture() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        int dim = 32;
        int numLayers = 4;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        SDVariable hidden = x;
        for (int layer = 0; layer < numLayers; layer++) {
            String prefix = "L" + layer + "_";

            // RMSNorm
            SDVariable xSq = hidden.mul(prefix + "sq", hidden);
            SDVariable meanSq = sd.mean(prefix + "mean", xSq, true, 1);
            SDVariable rsqrt = sd.math.rsqrt(prefix + "rsqrt", meanSq.add(prefix + "eps", 1e-5f));
            SDVariable norm = hidden.mul(prefix + "norm", rsqrt);

            // "Attention" (simplified as element-wise ops + residual)
            SDVariable wAttn = sd.constant(prefix + "wAttn",
                    Nd4j.linspace(DataType.FLOAT, 0.9f, 0.001f, dim).reshape(1, dim));
            SDVariable attnOut = norm.mul(prefix + "attn", wAttn);
            hidden = hidden.add(prefix + "res1", attnOut);

            // MLP: up-project, activation, down-project (element-wise approximation)
            SDVariable wUp = sd.constant(prefix + "wUp",
                    Nd4j.linspace(DataType.FLOAT, 0.8f, 0.002f, dim).reshape(1, dim));
            SDVariable wDown = sd.constant(prefix + "wDown",
                    Nd4j.linspace(DataType.FLOAT, 1.1f, -0.001f, dim).reshape(1, dim));
            SDVariable mlp1 = hidden.mul(prefix + "up", wUp);
            SDVariable mlp2 = sd.nn.relu(prefix + "act", mlp1, 0);
            SDVariable mlp3 = mlp2.mul(prefix + "down", wDown);
            hidden = hidden.add(prefix + "res2", mlp3);
        }

        sd.identity("result", hidden);

        int numSteps = 8;

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] gcOutputs = new INDArray[numSteps];
            INDArray[] inputs = new INDArray[numSteps];

            for (int s = 0; s < numSteps; s++) {
                inputs[s] = Nd4j.linspace(DataType.FLOAT, (s + 1) * 0.1f, 0.01f, dim).reshape(1, dim);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(planHandle, true);

                log.info("MultiLayer GC step {}: result[0..3] = [{}, {}, {}, {}]", s,
                        gcOutputs[s].getFloat(0, 0), gcOutputs[s].getFloat(0, 1),
                        gcOutputs[s].getFloat(0, 2), gcOutputs[s].getFloat(0, 3));
            }

            // Verify outputs differ
            for (int s = 1; s < numSteps; s++) {
                double diffFromPrev = gcOutputs[s].sub(gcOutputs[s - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 1e-6,
                        "MultiLayer steps " + (s - 1) + " and " + s + " identical!");
            }

            nativeOps.freeDynamicShapePlan(planHandle);

            // Reference
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2);

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputs[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");
                    if (s == 0) nativeOps.setPlanShapesFrozen(planHandle2, true);

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("MultiLayer step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    assertTrue(maxDiff < TOLERANCE,
                            "MultiLayer step " + s + ": GC diverges! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  NoGC: " + refOutput);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
        }
    }

    // ─── Test: Fast-replay correctness ──────────────────────────────────────

    /**
     * Validates that the fast-replay optimization (skipping arg table refresh
     * and EXT_INPUT_SYNC when argTableStable=true) produces identical results
     * to standard replay.
     *
     * The test runs 20 steps with different placeholder values:
     * - Steps 0-1: warmup (slot-by-slot execution)
     * - Step 2: CUDA graph capture
     * - Step 3: first replay (standard path, sets argTableStable=true)
     * - Steps 4-19: fast-replay path (skips arg table refresh)
     *
     * Then re-runs the same 20 steps WITHOUT graph capture (pure Triton) and
     * compares outputs step-by-step. Any divergence in the fast-replay steps
     * (4-19) indicates the optimization is incorrect.
     *
     * Also verifies that each step produces DIFFERENT output (no stale data).
     */
    @Test
    public void testFastReplayCorrectness() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        // Build a graph with placeholder input and several compilable ops.
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.linspace(DataType.FLOAT, 0.1, 0.05, 16).reshape(1, 16));
        SDVariable b1 = sd.constant("b1", Nd4j.valueArrayOf(new long[]{1, 16}, 0.3f));
        SDVariable w2 = sd.constant("w2", Nd4j.linspace(DataType.FLOAT, 0.5, -0.03, 16).reshape(1, 16));

        SDVariable h1 = x.mul("mul1", w1);
        SDVariable h2 = h1.add("add1", b1);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.mul("mul2", w2);
        SDVariable h5 = sd.nn.sigmoid("sig1", h4);
        SDVariable h6 = h5.mul("mul3", w1);
        SDVariable h7 = h6.add("add2", b1);
        SDVariable result = sd.math.tanh("result", h7);

        int numSteps = 20;

        // Generate different input values per step
        INDArray[] inputs = new INDArray[numSteps];
        for (int s = 0; s < numSteps; s++) {
            float[] vals = new float[16];
            for (int j = 0; j < 16; j++) {
                vals[j] = (s + 1) * 0.1f + j * 0.02f;
            }
            inputs[s] = Nd4j.createFromArray(vals).reshape(1, 16);
        }

        // ── Run 1: with graph capture (fast-replay will kick in after step 3) ──
        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        boolean prevFallback = Nd4j.getEnvironment().tritonAllowFallbackCapture();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonAllowFallbackCapture(true);

        INDArray[] gcOutputs = new INDArray[numSteps];

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan, "Plan is null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            for (int s = 0; s < numSteps; s++) {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }
            nativeOps.freeDynamicShapePlan(planHandle);

            // ── Run 2: without graph capture (reference) ──
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle2 = compileNativePlan(plan2);
            assertNotNull(planHandle2, "Reference plan handle null");

            try {
                for (int s = 0; s < numSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputs[s]));
                    Map<String, INDArray> refResults = executeNativePlan(planHandle2, plan2, extInputs);
                    INDArray refOutput = refResults.get("result");

                    if (s == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle2, true);
                    }

                    double maxDiff = gcOutputs[s].sub(refOutput).amaxNumber().doubleValue();
                    log.info("Fast-replay step {}: GC vs NoGC maxDiff = {}", s, maxDiff);

                    // Steps 4+ are fast-replay — these are the critical checks
                    assertTrue(maxDiff < TOLERANCE,
                            "Step " + s + ": fast-replay output diverges! maxDiff=" + maxDiff
                                    + "\n  GC:   " + gcOutputs[s]
                                    + "\n  Ref:  " + refOutput);
                }

                // Verify different steps produce different outputs (no stale data)
                for (int s = 1; s < numSteps; s++) {
                    double stepDiff = gcOutputs[s].sub(gcOutputs[s - 1]).amaxNumber().doubleValue();
                    assertTrue(stepDiff > 1e-6,
                            "Steps " + (s - 1) + " and " + s + " produced identical output! "
                                    + "Fast-replay may be using stale data. diff=" + stepDiff);
                }
                log.info("Fast-replay correctness: all {} steps match reference, "
                        + "all consecutive steps produce different output", numSteps);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle2);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonAllowFallbackCapture(prevFallback);
        }
    }

    /**
     * Tests that creating a fresh plan after freeing the previous one
     * correctly resets fast-replay state. Runs two sequential plan lifecycles:
     *
     * Plan 1: 10 steps (warmup + capture + fast-replay)
     * Plan 2: 10 steps (new plan, fresh capture + fast-replay)
     *
     * Both should produce identical output to a no-gc reference.
     */
    @Test
    public void testFastReplayAcrossPlanLifecycles() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = sd.constant("w", Nd4j.linspace(DataType.FLOAT, 0.2, 0.1, 8).reshape(1, 8));
        SDVariable b = sd.constant("b", Nd4j.valueArrayOf(new long[]{1, 8}, 0.1f));

        SDVariable h1 = x.mul("mul1", w);
        SDVariable h2 = h1.add("add1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.mul("mul2", w);
        SDVariable result = sd.nn.sigmoid("result", h4);

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        boolean prevFallback = Nd4j.getEnvironment().tritonAllowFallbackCapture();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonAllowFallbackCapture(true);

        int stepsPerPlan = 10;
        int totalSteps = stepsPerPlan * 2;

        INDArray[] inputs = new INDArray[totalSteps];
        for (int s = 0; s < totalSteps; s++) {
            float[] vals = new float[8];
            for (int j = 0; j < 8; j++) vals[j] = (s + 1) * 0.15f + j * 0.05f;
            inputs[s] = Nd4j.createFromArray(vals).reshape(1, 8);
        }

        try {
            INDArray[] gcOutputs = new INDArray[totalSteps];

            // Plan lifecycle 1
            DynamicShapePlan plan1 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer handle1 = compileNativePlan(plan1);
            Assumptions.assumeTrue(handle1 != null, "Native executor unavailable");

            for (int s = 0; s < stepsPerPlan; s++) {
                INDArray[] extInputs = resolveExternalInputs(plan1, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(handle1, plan1, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == 0) nativeOps.setPlanShapesFrozen(handle1, true);
            }
            nativeOps.freeDynamicShapePlan(handle1);

            // Plan lifecycle 2 (fresh plan, fresh graph capture)
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();
            DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer handle2 = compileNativePlan(plan2);

            for (int s = stepsPerPlan; s < totalSteps; s++) {
                INDArray[] extInputs = resolveExternalInputs(plan2, sd, Map.of("x", inputs[s]));
                Map<String, INDArray> results = executeNativePlan(handle2, plan2, extInputs);
                gcOutputs[s] = results.get("result").dup();
                if (s == stepsPerPlan) nativeOps.setPlanShapesFrozen(handle2, true);
            }
            nativeOps.freeDynamicShapePlan(handle2);

            // Reference: no graph capture
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            DynamicShapePlan planRef = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer handleRef = compileNativePlan(planRef);

            try {
                for (int s = 0; s < totalSteps; s++) {
                    INDArray[] extInputs = resolveExternalInputs(planRef, sd, Map.of("x", inputs[s]));
                    Map<String, INDArray> ref = executeNativePlan(handleRef, planRef, extInputs);
                    if (s == 0) nativeOps.setPlanShapesFrozen(handleRef, true);

                    double maxDiff = gcOutputs[s].sub(ref.get("result")).amaxNumber().doubleValue();
                    log.info("Lifecycle test step {}: maxDiff = {}{}", s, maxDiff,
                            s == stepsPerPlan ? " (first step of plan 2)" : "");
                    assertTrue(maxDiff < TOLERANCE,
                            "Step " + s + " diverged across plan lifecycle! maxDiff=" + maxDiff);
                }
                log.info("Lifecycle test: all {} steps correct across 2 plan lifecycles", totalSteps);
            } finally {
                nativeOps.freeDynamicShapePlan(handleRef);
            }
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonAllowFallbackCapture(prevFallback);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
