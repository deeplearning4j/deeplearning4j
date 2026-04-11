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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Triton fusion and replay accuracy tests.
 *
 * These tests target the two bug classes found in TritonGraphBackendTest:
 * 1. REPLAY BUGS: correct on first execution, wrong on iteration 2+ (stale data in replay)
 * 2. COMPILATION BUGS: wrong even on first execution (incorrect Triton IR for certain op patterns)
 *
 * Each test runs multiple iterations and verifies output matches native reference on EVERY iteration.
 * Tests cover: scalar constants, broadcast patterns, fused chains, changing inputs across replays,
 * residual connections, in-place-like patterns, and multi-output graphs.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonFusionReplayAccuracyTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-4;
    private static final int REPLAY_ITERS = 6; // enough to trigger compile + replay

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Infrastructure ─────────────────────────────────────────────────────

    private Pointer compileNativePlan(DynamicShapePlan plan) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        byte[] serialized = plan.serialize();
        assertNotNull(serialized);
        assertTrue(serialized.length > 0);
        BytePointer planBytes = new BytePointer(serialized);
        try {
            return nativeOps.compileDynamicShapePlan(planBytes, serialized.length);
        } catch (UnsupportedOperationException e) {
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
                // CPU backend — stream stays null
            }

            int status = nativeOps.executeDynamicShapePlan(planHandle, opContext, execStream);
            if (status != 0) {
                String errMsg = nativeOps.lastErrorMessage();
                nativeOps.clearLastError();
                fail("Native plan execution failed with status " + status + ": " + errMsg);
            }

            Map<String, INDArray> results = new LinkedHashMap<>();
            List<String> requestedOutputs = new ArrayList<>(plan.getRequestedOutputs());
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
                                              Map<String, INDArray> ph) {
        String[] extKeys = plan.getExternalInputKeys();
        INDArray[] extInputs = new INDArray[extKeys.length];
        for (int i = 0; i < extKeys.length; i++) {
            String varName = extKeys[i];
            INDArray arr = ph != null ? ph.get(varName) : null;
            if (arr == null) {
                SDVariable var = sd.getVariable(varName);
                if (var != null) {
                    arr = var.getArr();
                }
            }
            assertNotNull(arr, "Missing external input: " + varName);
            extInputs[i] = arr;
        }
        return extInputs;
    }

    /**
     * Run a graph through native DSP plan for multiple iterations, comparing every
     * iteration against the Java reference output. Freezes shapes after iter 0.
     * This is the core replay accuracy test pattern.
     */
    private void runReplayAccuracyTest(String testName, SameDiff sd, Map<String, INDArray> ph,
                                        String outputName, double tolerance) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), testName + ": Triton unavailable");

        // Use SLOT-BY-SLOT native execution as reference (not sd.output() which has bugs
        // with certain graph patterns returning zeros for multi-op chains with constants).
        // This tests Triton compilation + replay vs native slot-by-slot execution.
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, testName + ": plan is null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, testName + ": native executor unavailable");

        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // First execution (iter 0) is slot-by-slot — use it as reference
            Map<String, INDArray> refResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray refOutput = refResults.get(outputName).dup();
            assertNotNull(refOutput, testName + ": reference output is null");
            log.info("{}: ref shape={} sum={}", testName, refOutput.shape(), refOutput.sumNumber());

            // Freeze shapes to enable Triton compilation + replay
            nativeOps.setPlanShapesFrozen(planHandle, true);

            for (int iter = 1; iter < REPLAY_ITERS; iter++) {
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                INDArray actual = results.get(outputName);
                assertNotNull(actual, testName + ": null output at iter " + iter);

                double maxDiff = refOutput.sub(actual).amaxNumber().doubleValue();
                log.info("{}: iter={} maxDiff={} actSum={}", testName, iter, maxDiff,
                        actual.sumNumber());

                if (maxDiff > tolerance) {
                    INDArray flatRef = refOutput.reshape(refOutput.length());
                    INDArray flatAct = actual.reshape(actual.length());
                    int printed = 0;
                    for (long j = 0; j < flatRef.length() && printed < 10; j++) {
                        double d = Math.abs(flatRef.getDouble(j) - flatAct.getDouble(j));
                        if (d > tolerance) {
                            log.error("  MISMATCH idx={}: ref={} actual={} diff={}", j,
                                    flatRef.getDouble(j), flatAct.getDouble(j), d);
                            printed++;
                        }
                    }
                }

                assertTrue(maxDiff < tolerance,
                        testName + ": maxDiff=" + maxDiff + " at iter " + iter +
                                " (tolerance=" + tolerance + ")");
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Same as runReplayAccuracyTest but changes placeholder values between iterations.
     * This catches stale-input bugs where the graph reads from capture-time buffers.
     * Uses TWO native plan executions per iteration: one slot-by-slot (no freeze) as ref,
     * one with the frozen/compiled plan as test.
     */
    private void runReplayWithChangingInputs(String testName, SameDiff sd,
                                              java.util.function.Supplier<Map<String, INDArray>> inputFactory,
                                              String outputName, double tolerance) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), testName + ": Triton unavailable");

        // Compile two plans: one for reference (slot-by-slot), one for test (frozen/compiled)
        // Use a mutable map so we can update values in-place across iterations
        Map<String, INDArray> ph = new java.util.HashMap<>(inputFactory.get());
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, testName + ": plan is null");

        Pointer refPlanHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(refPlanHandle != null, testName + ": native executor unavailable");

        Pointer testPlanHandle = compileNativePlan(plan);
        assertNotNull(testPlanHandle, testName + ": test plan is null");

        try {
            // First iter on both plans — warmup
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            executeNativePlan(refPlanHandle, plan, extInputs);
            executeNativePlan(testPlanHandle, plan, extInputs);

            // Freeze only the test plan
            nativeOps.setPlanShapesFrozen(testPlanHandle, true);

            for (int iter = 0; iter < REPLAY_ITERS; iter++) {
                // Generate new values but copy INTO existing buffers to keep GPU addresses stable.
                // In real VLM decode, inputs change value but stay at the same GPU address.
                Map<String, INDArray> newPh = inputFactory.get();
                for (Map.Entry<String, INDArray> entry : newPh.entrySet()) {
                    INDArray existing = ph.get(entry.getKey());
                    if (existing != null && java.util.Arrays.equals(existing.shape(), entry.getValue().shape())) {
                        existing.assign(entry.getValue());
                    } else {
                        ph.put(entry.getKey(), entry.getValue());
                    }
                }
                extInputs = resolveExternalInputs(plan, sd, ph);

                // Reference: unfrozen slot-by-slot execution
                Map<String, INDArray> refResults = executeNativePlan(refPlanHandle, plan, extInputs);
                INDArray refOutput = refResults.get(outputName).dup();

                // Test: frozen/compiled execution
                Map<String, INDArray> testResults = executeNativePlan(testPlanHandle, plan, extInputs);
                INDArray actual = testResults.get(outputName);
                assertNotNull(actual, testName + ": null at iter " + iter);

                double maxDiff = refOutput.sub(actual).amaxNumber().doubleValue();
                log.info("{}: iter={} maxDiff={} refSum={} actSum={}", testName, iter, maxDiff,
                        refOutput.sumNumber(), actual.sumNumber());

                assertTrue(maxDiff < tolerance,
                        testName + ": maxDiff=" + maxDiff + " at iter " + iter);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(refPlanHandle);
            nativeOps.freeDynamicShapePlan(testPlanHandle);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // REPLAY BUG TESTS: correct on first exec, verify still correct on replay
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Simple add+mul chain — the most basic fused graph.
     * If this fails on replay, the fundamental replay mechanism is broken.
     */
    @Test
    public void testReplayAddMulChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable h1 = x.add("add1", sd.constant("b1", Nd4j.ones(DataType.FLOAT, 1, 32)));
        SDVariable h2 = h1.mul("mul1", sd.constant("w1", Nd4j.ones(DataType.FLOAT, 1, 32).mul(2)));
        sd.identity("result", h2);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("addMulChain", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Add+mul chain with changing inputs each replay iteration.
     * Catches stale-input bugs (graph reads from capture-time buffer instead of fresh data).
     */
    @Test
    public void testReplayAddMulChainChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable h1 = x.add("add1", sd.constant("b1", Nd4j.ones(DataType.FLOAT, 1, 32)));
        SDVariable h2 = h1.mul("mul1", sd.constant("w1", Nd4j.ones(DataType.FLOAT, 1, 32).mul(2)));
        sd.identity("result", h2);

        runReplayWithChangingInputs("addMulChainChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 32)),
                "result", TOLERANCE);
    }

    /**
     * Broadcast multiply with different rank inputs — the pattern that fails in VLM decode.
     * Tests [B, H, S, D] * [B, 1, S, D] broadcast.
     */
    @Test
    public void testReplayBroadcastMul4D() {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 4, 8, 16);
        SDVariable kv = sd.placeHolder("kv", DataType.FLOAT, 1, 1, 8, 16);
        SDVariable product = q.mul("product", kv);
        sd.identity("result", product);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16);
        INDArray kvArr = Nd4j.randn(DataType.FLOAT, 1, 1, 8, 16);
        runReplayAccuracyTest("broadcastMul4D", sd, Map.of("q", qArr, "kv", kvArr),
                "result", TOLERANCE);
    }

    /**
     * Broadcast multiply with changing inputs on every iteration.
     */
    @Test
    public void testReplayBroadcastMulChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 4, 8, 16);
        SDVariable kv = sd.placeHolder("kv", DataType.FLOAT, 1, 1, 8, 16);
        SDVariable product = q.mul("product", kv);
        sd.identity("result", product);

        runReplayWithChangingInputs("broadcastMulChanging", sd,
                () -> Map.of("q", Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16),
                             "kv", Nd4j.randn(DataType.FLOAT, 1, 1, 8, 16)),
                "result", TOLERANCE);
    }

    /**
     * Residual connection: h = relu(x @ W + b) + x.
     * Residuals are common in transformers and test that the input is read correctly
     * both at the start AND end of the fused segment.
     */
    @Test
    public void testReplayResidualConnection() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 16).mul(0.1));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable h = sd.nn.relu("relu1", x.mmul("mm1", w).add("add1", b), 0);
        h.add("result", x); // residual

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("residualConn", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Two-layer residual chain — deeper fusion with multiple residual add-backs.
     */
    @Test
    public void testReplayTwoLayerResidual() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);

        // Layer 1
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 16).mul(0.1));
        SDVariable h1 = sd.nn.relu("relu1", x.mmul("mm1", w1), 0);
        SDVariable r1 = h1.add("res1", x);

        // Layer 2
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 16).mul(0.1));
        SDVariable h2 = sd.nn.relu("relu2", r1.mmul("mm2", w2), 0);
        h2.add("result", r1);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("twoLayerResidual", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Softmax along last axis — tests reduction+elementwise fusion.
     * Softmax is critical for attention and commonly fused.
     */
    @Test
    public void testReplaySoftmax() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        sd.nn.softmax("result", x, -1);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("softmax", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Softmax with changing inputs — verifies the reduction state is fresh each replay.
     */
    @Test
    public void testReplaySoftmaxChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        sd.nn.softmax("result", x, -1);

        runReplayWithChangingInputs("softmaxChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 32)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SCALAR CONSTANT TESTS: ops with scalar constants fused into the kernel
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Scalar add + scalar mul — tests that scalar constants are baked into the kernel correctly.
     * This is the pattern that testTritonSetScalar fails on.
     */
    @Test
    public void testReplayScalarAddMul() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable h1 = x.add("add1", sd.constant("c1", Nd4j.scalar(DataType.FLOAT, 1.0f)));
        SDVariable h2 = h1.mul("mul1", sd.constant("c2", Nd4j.scalar(DataType.FLOAT, 2.0f)));
        sd.identity("result", h2);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("scalarAddMul", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Scalar sub + scalar div — another scalar constant pattern.
     */
    @Test
    public void testReplayScalarSubDiv() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable h1 = x.sub("sub1", sd.constant("c1", Nd4j.scalar(DataType.FLOAT, 0.5f)));
        SDVariable h2 = h1.div("div1", sd.constant("c2", Nd4j.scalar(DataType.FLOAT, 3.0f)));
        sd.identity("result", h2);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16).add(1.0); // avoid div-by-zero issues
        runReplayAccuracyTest("scalarSubDiv", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Multiple scalar constants in a chain: (x + 1) * 2 - 0.5.
     */
    @Test
    public void testReplayScalarTripleChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable h1 = x.add("add1", sd.constant("c1", Nd4j.scalar(DataType.FLOAT, 1.0f)));
        SDVariable h2 = h1.mul("mul1", sd.constant("c2", Nd4j.scalar(DataType.FLOAT, 2.0f)));
        SDVariable h3 = h2.sub("sub1", sd.constant("c3", Nd4j.scalar(DataType.FLOAT, 0.5f)));
        sd.identity("result", h3);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("scalarTriple", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Scalar constant with changing inputs — catches stale scalar buffer reads.
     */
    @Test
    public void testReplayScalarChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable h1 = x.add("add1", sd.constant("c1", Nd4j.scalar(DataType.FLOAT, 1.0f)));
        SDVariable h2 = h1.mul("mul1", sd.constant("c2", Nd4j.scalar(DataType.FLOAT, 2.0f)));
        sd.identity("result", h2);

        runReplayWithChangingInputs("scalarChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 16)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // BROADCAST PATTERN TESTS: various broadcast shapes that arise in models
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Row broadcast add: [B, N] + [1, N].
     */
    @Test
    public void testReplayRowBroadcastAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable h = x.add("add1", bias);
        sd.nn.relu("result", h, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32);
        runReplayAccuracyTest("rowBroadcastAdd", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Column broadcast mul: [B, N] * [B, 1].
     */
    @Test
    public void testReplayColumnBroadcastMul() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable scale = sd.placeHolder("scale", DataType.FLOAT, -1, 1);
        SDVariable h = x.mul("mul1", scale);
        sd.identity("result", h);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32);
        INDArray scaleArr = Nd4j.randn(DataType.FLOAT, 8, 1).add(1.0);
        runReplayAccuracyTest("colBroadcastMul", sd,
                Map.of("x", xArr, "scale", scaleArr), "result", TOLERANCE);
    }

    /**
     * 3D broadcast: [B, S, D] + [1, 1, D] — typical bias add in transformers.
     */
    @Test
    public void testReplayBroadcast3DBiasAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8, 64);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 1, 64));
        SDVariable h = x.add("add1", bias);
        sd.nn.relu("result", h, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, 8, 64);
        runReplayAccuracyTest("broadcast3DBias", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // MULTI-OP FUSION CHAIN TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * RMS norm pattern: x * rsqrt(mean(x^2) + eps) * gamma.
     * This is the core normalization pattern in LLMs.
     */
    @Test
    public void testReplayRmsNormPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, 64));
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-6f));

        SDVariable xSq = x.mul("xsq", x);
        SDVariable mean = xSq.mean("mean1", true, 1);
        SDVariable meanEps = mean.add("meaneps", eps);
        SDVariable rsqrt = sd.math.rsqrt("rsqrt1", meanEps);
        SDVariable norm = x.mul("norm", rsqrt);
        norm.mul("result", gamma);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 64);
        runReplayAccuracyTest("rmsNormPattern", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * GELU activation pattern: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
     * Tests complex elementwise fusion.
     */
    @Test
    public void testReplayGeluPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        // Simplified: just use the built-in GELU
        sd.nn.gelu("result", x);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32);
        runReplayAccuracyTest("geluPattern", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Gather + elementwise + reduce chain — tests data movement fused with compute.
     */
    @Test
    public void testReplayGatherAddReduce() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 16, 32);
        SDVariable gathered = sd.gather("gather1", x, new int[]{0, 3, 7, 15}, 0);
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 32));
        SDVariable added = gathered.add("add1", bias);
        added.sum("result", 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 16, 32);
        runReplayAccuracyTest("gatherAddReduce", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Two-branch diverge-then-merge: tests that both branches of a fork produce
     * correct results and the merge (add) is correct.
     *   h1 = relu(x)
     *   h2 = sigmoid(x)
     *   result = h1 + h2
     */
    @Test
    public void testReplayForkMerge() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable h1 = sd.nn.relu("relu1", x, 0);
        SDVariable h2 = sd.nn.sigmoid("sig1", x);
        h1.add("result", h2);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("forkMerge", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Fork-merge with changing inputs — ensures both branches read fresh data.
     */
    @Test
    public void testReplayForkMergeChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable h1 = sd.nn.relu("relu1", x, 0);
        SDVariable h2 = sd.nn.sigmoid("sig1", x);
        h1.add("result", h2);

        runReplayWithChangingInputs("forkMergeChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 32)),
                "result", TOLERANCE);
    }

    /**
     * Concat followed by elementwise — tests that concat output addressing is correct
     * when consumed by a fused downstream op.
     */
    @Test
    public void testReplayConcatThenRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 16);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 16);
        SDVariable cat = sd.concat("concat1", 1, a, b);
        sd.nn.relu("result", cat, 0);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        INDArray bArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("concatRelu", sd, Map.of("a", aArr, "b", bArr), "result", TOLERANCE);
    }

    /**
     * Split then process each half — tests that split output slicing is correct in fusion.
     */
    @Test
    public void testReplaySplitThenProcess() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable[] splits = sd.split(new String[]{"split0", "split1"}, x, 2, 1);
        SDVariable h1 = sd.nn.relu("relu1", splits[0], 0);
        SDVariable h2 = sd.nn.sigmoid("sig1", splits[1]);
        h1.add("result", h2);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("splitProcess", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // MATMUL FUSION TESTS — matmul combined with other ops
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Linear layer: matmul + bias + relu. The most common fused pattern.
     */
    @Test
    public void testReplayLinearBiasRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 1, 32));
        SDVariable mm = x.mmul("mm1", w);
        SDVariable added = mm.add("add1", b);
        sd.nn.relu("result", added, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("linearBiasRelu", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Linear with changing inputs — the decode pattern.
     */
    @Test
    public void testReplayLinearChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 1, 32));
        SDVariable mm = x.mmul("mm1", w);
        SDVariable added = mm.add("add1", b);
        sd.nn.relu("result", added, 0);

        runReplayWithChangingInputs("linearChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 16)),
                "result", 1e-3);
    }

    /**
     * Two sequential linear layers: x @ W1 + b1 -> relu -> @ W2 + b2.
     * Tests multi-matmul fusion.
     */
    @Test
    public void testReplayTwoLinearLayers() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 32));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16).mul(0.1));
        SDVariable b2 = sd.constant("b2", Nd4j.zeros(DataType.FLOAT, 1, 16));

        SDVariable h1 = sd.nn.relu("relu1", x.mmul("mm1", w1).add("add1", b1), 0);
        h1.mmul("mm2", w2).add("result", b2);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("twoLinear", sd, Map.of("x", xArr), "result", 1e-3);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // EDGE CASE TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Identity-only graph — tests the minimal possible fused segment.
     */
    @Test
    public void testReplayIdentityOnly() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        sd.identity("result", x);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("identityOnly", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Large tensor — tests that indexing doesn't overflow for big arrays.
     */
    @Test
    public void testReplayLargeTensor() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 2048);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 2048));
        SDVariable h = x.add("add1", bias);
        sd.nn.relu("result", h, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, 2048);
        runReplayAccuracyTest("largeTensor", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Zeros input — tests that zero buffers don't get confused with uninitialized memory.
     */
    @Test
    public void testReplayZerosInput() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable h = x.add("add1", sd.constant("b", Nd4j.ones(DataType.FLOAT, 1, 16)));
        sd.nn.relu("result", h, 0);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("zerosInput", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Negative values through tanh — tests saturation region handling.
     */
    @Test
    public void testReplayTanhSaturation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        sd.math.tanh("result", x);

        // Large values that saturate tanh to +/-1
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32).mul(10.0);
        runReplayAccuracyTest("tanhSaturation", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Multiple placeholders consumed by a single op — tests multi-input arg table handling.
     */
    @Test
    public void testReplayMultiPlaceholderAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 16);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 16);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, -1, 16);
        SDVariable h = a.add("add1", b);
        h.add("result", c);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        INDArray bArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        INDArray cArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runReplayAccuracyTest("multiPlaceholder", sd,
                Map.of("a", aArr, "b", bArr, "c", cArr), "result", TOLERANCE);
    }

    /**
     * Three placeholders with changing inputs each iteration.
     */
    @Test
    public void testReplayMultiPlaceholderChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 16);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 16);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, -1, 16);
        SDVariable h = a.add("add1", b);
        h.add("result", c);

        runReplayWithChangingInputs("multiPhChanging", sd,
                () -> Map.of("a", Nd4j.randn(DataType.FLOAT, 4, 16),
                             "b", Nd4j.randn(DataType.FLOAT, 4, 16),
                             "c", Nd4j.randn(DataType.FLOAT, 4, 16)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SHAPE MANIPULATION + COMPUTE TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Reshape then elementwise — reshape changes strides and memory layout.
     * Tests that Triton handles the reshaped (potentially non-contiguous) layout correctly.
     * Pattern: [4, 8] -> reshape [2, 16] -> add bias -> relu
     */
    @Test
    public void testReplayReshapeThenElementwise() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable reshaped = sd.reshape("reshape1", x, 2, 16);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable added = reshaped.add("add1", bias);
        sd.nn.relu("result", added, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8);
        runReplayAccuracyTest("reshapeElementwise", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Reshape with changing inputs — catches stale-pointer bugs after reshape.
     */
    @Test
    public void testReplayReshapeThenElementwiseChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable reshaped = sd.reshape("reshape1", x, 2, 16);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable added = reshaped.add("add1", bias);
        sd.nn.relu("result", added, 0);

        runReplayWithChangingInputs("reshapeEwChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)),
                "result", TOLERANCE);
    }

    /**
     * Permute/transpose then matmul — common attention pattern (transpose K before Q @ K^T).
     * Tests that permuted strides are handled correctly by the fused segment.
     * Pattern: K [1, 4, 8, 16] -> permute [1, 4, 16, 8] -> matmul with Q [1, 4, 8, 16]
     * producing attention scores [1, 4, 8, 8].
     */
    @Test
    public void testReplayPermuteThenMatmul() {
        SameDiff sd = SameDiff.create();
        // Q: [batch, heads, seqQ, dim]
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 4, 8, 16);
        // K: [batch, heads, seqK, dim] -> transpose last two dims -> [batch, heads, dim, seqK]
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, 4, 8, 16);
        SDVariable kT = sd.permute("kT", k, 0, 1, 3, 2);
        // attn = Q @ K^T -> [1, 4, 8, 8]
        SDVariable attn = sd.mmul("attn", q, kT);
        // scale by 1/sqrt(dim)
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(16)));
        attn.mul("result", scale);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16).mul(0.1);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16).mul(0.1);
        runReplayAccuracyTest("permuteMatmul", sd, Map.of("q", qArr, "k", kArr), "result", 1e-3);
    }

    /**
     * Permute+matmul with changing inputs — catches stale permuted buffer reads on replay.
     */
    @Test
    public void testReplayPermuteThenMatmulChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 4, 8, 16);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, 4, 8, 16);
        SDVariable kT = sd.permute("kT", k, 0, 1, 3, 2);
        SDVariable attn = sd.mmul("attn", q, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(16)));
        attn.mul("result", scale);

        runReplayWithChangingInputs("permuteMatmulChanging", sd,
                () -> Map.of("q", Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16).mul(0.1),
                             "k", Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16).mul(0.1)),
                "result", 1e-3);
    }

    /**
     * Constant generation fused with compute — used in attention masks.
     * Creates a constant tensor (ones) and uses it in a multiply with the input.
     * Tests that constant buffers are correctly handled in fused segments.
     */
    @Test
    public void testReplayConstantOfShapeWithCompute() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 32);
        // Simulate constant mask: all ones -> scale -> add to input
        SDVariable mask = sd.constant("mask", Nd4j.ones(DataType.FLOAT, 4, 32));
        SDVariable scaled = x.mul("masked", mask);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable added = scaled.add("add1", bias);
        sd.nn.relu("result", added, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("constOfShapeCompute", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Constant mask with negative values (attention mask pattern): mask = -3.4e38 where condition is false.
     * Tests constant generation with extreme float values in fusion.
     */
    @Test
    public void testReplayConstantMaskExtreme() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 32);
        // Create a mask with very negative values (attention mask pattern)
        INDArray maskData = Nd4j.zeros(DataType.FLOAT, 4, 32);
        // Set first 16 columns to 0 (attend), last 16 to -3.4e38 (mask out)
        for (int i = 0; i < 4; i++) {
            for (int j = 16; j < 32; j++) {
                maskData.putScalar(i, j, -3.4028235e+38f);
            }
        }
        SDVariable mask = sd.constant("mask", maskData);
        SDVariable masked = x.add("masked", mask);
        sd.nn.softmax("result", masked, -1);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("constMaskExtreme", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Where/Select — conditional ops in fused segments (used in masked attention).
     * Pattern: where(condition, x, fillValue) — selects x where condition is true, fillValue otherwise.
     */
    @Test
    public void testReplayWhereSelect() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 32);
        // Condition: x > 0
        SDVariable condition = sd.gt("cond", x, 0.0);
        // Fill value for false positions
        SDVariable fillVal = sd.constant("fill", Nd4j.zeros(DataType.FLOAT, 4, 32));
        // where(x, fill, cond) — select x where cond is true, fill otherwise
        SDVariable selected = sd.where("where1", x, fillVal, condition);
        // Add bias after selection
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 32));
        selected.add("result", bias);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("whereSelect", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Where/Select with changing inputs — condition changes each iteration.
     */
    @Test
    public void testReplayWhereSelectChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 32);
        SDVariable condition = sd.gt("cond", x, 0.0);
        SDVariable fillVal = sd.constant("fill", Nd4j.zeros(DataType.FLOAT, 4, 32));
        SDVariable selected = sd.where("where1", x, fillVal, condition);
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 32));
        selected.add("result", bias);

        runReplayWithChangingInputs("whereSelectChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 32)),
                "result", TOLERANCE);
    }

    /**
     * Batch normalization pattern: (x - mean) / sqrt(var + eps) * gamma + beta.
     * This is the standard BN inference pattern that should be fusible.
     */
    @Test
    public void testReplayBatchNormPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, 64));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 1, 64));
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));

        // mean and var across batch dimension
        SDVariable mean = x.mean("mean1", true, 0);
        SDVariable centered = x.sub("centered", mean);
        SDVariable var = centered.mul("sq", centered).mean("var1", true, 0);
        SDVariable varEps = var.add("vareps", eps);
        SDVariable std = sd.math.sqrt("std", varEps);
        SDVariable normed = centered.div("normed", std);
        SDVariable scaled = normed.mul("scaled", gamma);
        scaled.add("result", beta);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 64);
        runReplayAccuracyTest("batchNormPattern", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Batch norm with changing inputs — catches stale running-stats bugs.
     */
    @Test
    public void testReplayBatchNormChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, 64));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 1, 64));
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));

        SDVariable mean = x.mean("mean1", true, 0);
        SDVariable centered = x.sub("centered", mean);
        SDVariable var = centered.mul("sq", centered).mean("var1", true, 0);
        SDVariable varEps = var.add("vareps", eps);
        SDVariable std = sd.math.sqrt("std", varEps);
        SDVariable normed = centered.div("normed", std);
        SDVariable scaled = normed.mul("scaled", gamma);
        scaled.add("result", beta);

        runReplayWithChangingInputs("batchNormChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 64)),
                "result", 1e-3);
    }

    /**
     * Chained reductions — sum then mean. Tests multiple reduction passes in a fused segment.
     * Pattern: x [4, 8, 16] -> sum(axis=2) -> [4, 8] -> mean(axis=1) -> [4, 1]
     */
    @Test
    public void testReplayChainedReductions() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8, 16);
        SDVariable summed = x.sum("sum1", 2);   // [4, 8]
        SDVariable meaned = summed.mean("mean1", true, 1);  // [4, 1]
        // Add a post-reduction elementwise so there's more to fuse
        SDVariable bias = sd.constant("bias", Nd4j.scalar(DataType.FLOAT, 1.0f));
        meaned.add("result", bias);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8, 16);
        runReplayAccuracyTest("chainedReductions", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Chained reductions with changing inputs.
     */
    @Test
    public void testReplayChainedReductionsChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8, 16);
        SDVariable summed = x.sum("sum1", 2);
        SDVariable meaned = summed.mean("mean1", true, 1);
        SDVariable bias = sd.constant("bias", Nd4j.scalar(DataType.FLOAT, 1.0f));
        meaned.add("result", bias);

        runReplayWithChangingInputs("chainedRedsChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8, 16)),
                "result", 1e-3);
    }

    /**
     * Mixed dtype cast + compute — cast from INT64 to FLOAT then compute.
     * Common in position_ids / token_type_ids processing in transformers.
     * Pattern: int64 input -> cast to float -> add bias -> relu
     */
    @Test
    public void testReplayMixedDtypeCastCompute() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.INT64, 4, 16);
        SDVariable xFloat = sd.castTo("cast1", x, DataType.FLOAT);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable added = xFloat.add("add1", bias);
        sd.nn.relu("result", added, 0);

        INDArray xArr = Nd4j.create(DataType.INT64, 4, 16);
        // Fill with position ids: 0, 1, 2, ...
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                xArr.putScalar(i, j, j);
            }
        }
        runReplayAccuracyTest("mixedDtypeCast", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Cast with changing inputs — new position ids each iteration.
     */
    @Test
    public void testReplayMixedDtypeCastChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.INT64, 4, 16);
        SDVariable xFloat = sd.castTo("cast1", x, DataType.FLOAT);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable added = xFloat.add("add1", bias);
        sd.nn.relu("result", added, 0);

        runReplayWithChangingInputs("castChanging", sd,
                () -> {
                    INDArray posIds = Nd4j.create(DataType.INT64, 4, 16);
                    int offset = (int) (Math.random() * 100);
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 16; j++) {
                            posIds.putScalar(i, j, offset + j);
                        }
                    }
                    return Map.of("x", posIds);
                },
                "result", TOLERANCE);
    }

    /**
     * Squeeze + unsqueeze around compute — rank manipulation before/after ops.
     * Pattern: [4, 1, 32] -> squeeze(axis=1) -> [4, 32] -> add+relu -> expandDims(axis=1) -> [4, 1, 32]
     */
    @Test
    public void testReplaySqueezeUnsqueezeCompute() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 1, 32);
        SDVariable squeezed = sd.squeeze("squeeze1", x, 1);  // [4, 32]
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable added = squeezed.add("add1", bias);
        SDVariable activated = sd.nn.relu("relu1", added, 0);
        sd.expandDims("result", activated, 1);  // [4, 1, 32]

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 1, 32);
        runReplayAccuracyTest("squeezeUnsqueeze", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Squeeze/unsqueeze with changing inputs.
     */
    @Test
    public void testReplaySqueezeUnsqueezeChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 1, 32);
        SDVariable squeezed = sd.squeeze("squeeze1", x, 1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable added = squeezed.add("add1", bias);
        SDVariable activated = sd.nn.relu("relu1", added, 0);
        sd.expandDims("result", activated, 1);

        runReplayWithChangingInputs("sqUnsqChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 1, 32)),
                "result", TOLERANCE);
    }

    /**
     * Cumulative sum — scan operation in fusion.
     * Pattern: x [4, 16] -> cumsum(axis=1) -> add bias -> relu
     * cumsum is a scan, not a simple reduction, so it tests a different code path.
     */
    @Test
    public void testReplayCumulativeSum() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable cumulated = sd.cumsum("cumsum1", x, false, false, 1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable added = cumulated.add("add1", bias);
        sd.nn.relu("result", added, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16).mul(0.1); // small values to avoid overflow
        runReplayAccuracyTest("cumulativeSum", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Cumulative sum with changing inputs.
     */
    @Test
    public void testReplayCumulativeSumChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable cumulated = sd.cumsum("cumsum1", x, false, false, 1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable added = cumulated.add("add1", bias);
        sd.nn.relu("result", added, 0);

        runReplayWithChangingInputs("cumsumChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 16).mul(0.1)),
                "result", 1e-3);
    }

    /**
     * Multiple outputs from same graph — test that requesting 2 outputs works in replay.
     * Pattern: x -> two branches, each producing an output.
     *   out1 = relu(x + bias1)
     *   out2 = sigmoid(x + bias2)
     * Both outputs are requested and verified independently.
     */
    @Test
    public void testReplayMultipleOutputs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "multiOutput: Triton unavailable");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable bias1 = sd.constant("bias1", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable bias2 = sd.constant("bias2", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable branch1 = sd.nn.relu("out1", x.add("add1", bias1), 0);
        SDVariable branch2 = sd.nn.sigmoid("out2", x.add("add2", bias2));

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        Map<String, INDArray> ph = Map.of("x", xArr);

        // Compile plan with both outputs
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "out1", "out2");
        assertNotNull(plan, "multiOutput: plan is null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "multiOutput: native executor unavailable");

        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // Iter 0: slot-by-slot (unfrozen) — use as reference
            Map<String, INDArray> refResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray refOut1 = refResults.get("out1").dup();
            INDArray refOut2 = refResults.get("out2").dup();
            assertNotNull(refOut1, "multiOutput: ref out1 is null");
            assertNotNull(refOut2, "multiOutput: ref out2 is null");
            log.info("multiOutput: ref out1 sum={} out2 sum={}", refOut1.sumNumber(), refOut2.sumNumber());

            // Freeze shapes to enable compilation + replay
            nativeOps.setPlanShapesFrozen(planHandle, true);

            for (int iter = 1; iter < REPLAY_ITERS; iter++) {
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);

                INDArray actOut1 = results.get("out1");
                INDArray actOut2 = results.get("out2");
                assertNotNull(actOut1, "multiOutput: null out1 at iter " + iter);
                assertNotNull(actOut2, "multiOutput: null out2 at iter " + iter);

                double maxDiff1 = refOut1.sub(actOut1).amaxNumber().doubleValue();
                double maxDiff2 = refOut2.sub(actOut2).amaxNumber().doubleValue();
                log.info("multiOutput: iter={} maxDiff1={} maxDiff2={}", iter, maxDiff1, maxDiff2);

                assertTrue(maxDiff1 < TOLERANCE,
                        "multiOutput: out1 maxDiff=" + maxDiff1 + " at iter " + iter);
                assertTrue(maxDiff2 < TOLERANCE,
                        "multiOutput: out2 maxDiff=" + maxDiff2 + " at iter " + iter);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Multiple outputs with changing inputs — both outputs must update.
     * Uses two plan handles: one unfrozen (reference), one frozen (test).
     */
    @Test
    public void testReplayMultipleOutputsChangingInputs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "multiOutputChanging: Triton unavailable");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable bias1 = sd.constant("bias1", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable bias2 = sd.constant("bias2", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable branch1 = sd.nn.relu("out1", x.add("add1", bias1), 0);
        SDVariable branch2 = sd.nn.sigmoid("out2", x.add("add2", bias2));

        Map<String, INDArray> ph = Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 32));
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "out1", "out2");
        assertNotNull(plan, "multiOutputChanging: plan is null");

        Pointer refPlanHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(refPlanHandle != null, "multiOutputChanging: native executor unavailable");
        Pointer testPlanHandle = compileNativePlan(plan);
        assertNotNull(testPlanHandle, "multiOutputChanging: test plan is null");

        try {
            // Warmup both plans
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            executeNativePlan(refPlanHandle, plan, extInputs);
            executeNativePlan(testPlanHandle, plan, extInputs);

            // Freeze only the test plan
            nativeOps.setPlanShapesFrozen(testPlanHandle, true);

            for (int iter = 0; iter < REPLAY_ITERS; iter++) {
                ph = Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 32));
                extInputs = resolveExternalInputs(plan, sd, ph);

                // Reference: unfrozen slot-by-slot
                Map<String, INDArray> refResults = executeNativePlan(refPlanHandle, plan, extInputs);
                INDArray refOut1 = refResults.get("out1").dup();
                INDArray refOut2 = refResults.get("out2").dup();

                // Test: frozen/compiled
                Map<String, INDArray> testResults = executeNativePlan(testPlanHandle, plan, extInputs);

                double maxDiff1 = refOut1.sub(testResults.get("out1")).amaxNumber().doubleValue();
                double maxDiff2 = refOut2.sub(testResults.get("out2")).amaxNumber().doubleValue();
                log.info("multiOutputChanging: iter={} maxDiff1={} maxDiff2={}", iter, maxDiff1, maxDiff2);

                assertTrue(maxDiff1 < TOLERANCE,
                        "multiOutputChanging: out1 maxDiff=" + maxDiff1 + " at iter " + iter);
                assertTrue(maxDiff2 < TOLERANCE,
                        "multiOutputChanging: out2 maxDiff=" + maxDiff2 + " at iter " + iter);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(refPlanHandle);
            nativeOps.freeDynamicShapePlan(testPlanHandle);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TRANSFORMER ATTENTION BLOCK TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Full transformer self-attention: Q*K^T / sqrt(d) + mask -> softmax -> * V.
     * 4D tensors [batch, heads, seq, dim]. This is the core attention pattern
     * in every transformer model. Tests fusion of matmul, scale, add, softmax, matmul.
     */
    @Test
    public void testReplayTransformerSelfAttention() {
        SameDiff sd = SameDiff.create();
        int B = 1, H = 4, S = 8, D = 16;
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, H, S, D);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, B, H, S, D);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, B, H, S, D);

        // K^T: [B, H, S, D] -> [B, H, D, S]
        SDVariable kT = sd.permute("kT", k, 0, 1, 3, 2);
        // scores = Q @ K^T -> [B, H, S, S]
        SDVariable scores = sd.mmul("scores", q, kT);
        // scale by 1/sqrt(d)
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(D)));
        SDVariable scaled = scores.mul("scaled", scale);
        // add causal mask (upper triangular = -inf)
        INDArray maskArr = Nd4j.zeros(DataType.FLOAT, 1, 1, S, S);
        for (int i = 0; i < S; i++) {
            for (int j = i + 1; j < S; j++) {
                maskArr.putScalar(new int[]{0, 0, i, j}, -3.4028235e+38f);
            }
        }
        SDVariable mask = sd.constant("mask", maskArr);
        SDVariable masked = scaled.add("masked", mask);
        // softmax along last axis
        SDVariable attnWeights = sd.nn.softmax("attnWeights", masked, -1);
        // output = attnWeights @ V -> [B, H, S, D]
        sd.mmul("result", attnWeights, v);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1);
        runReplayAccuracyTest("selfAttention", sd,
                Map.of("q", qArr, "k", kArr, "v", vArr), "result", 1e-3);
    }

    /**
     * Transformer self-attention with changing inputs.
     */
    @Test
    public void testReplayTransformerSelfAttentionChangingInputs() {
        SameDiff sd = SameDiff.create();
        int B = 1, H = 4, S = 8, D = 16;
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, H, S, D);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, B, H, S, D);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, B, H, S, D);

        SDVariable kT = sd.permute("kT", k, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("scores", q, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(D)));
        SDVariable scaled = scores.mul("scaled", scale);
        INDArray maskArr = Nd4j.zeros(DataType.FLOAT, 1, 1, S, S);
        for (int i = 0; i < S; i++) {
            for (int j = i + 1; j < S; j++) {
                maskArr.putScalar(new int[]{0, 0, i, j}, -3.4028235e+38f);
            }
        }
        SDVariable mask = sd.constant("mask", maskArr);
        SDVariable masked = scaled.add("masked", mask);
        SDVariable attnWeights = sd.nn.softmax("attnWeights", masked, -1);
        sd.mmul("result", attnWeights, v);

        runReplayWithChangingInputs("selfAttnChanging", sd,
                () -> Map.of("q", Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1),
                             "k", Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1),
                             "v", Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1)),
                "result", 1e-3);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SiLU / SWISH ACTIVATION TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * SiLU/Swish activation: x * sigmoid(x). Used in LLaMA/Mistral FFN blocks.
     * Tests elementwise multiply fused with sigmoid — a non-trivial elementwise chain
     * because the same input feeds both branches.
     */
    @Test
    public void testReplaySiLUActivation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable sigX = sd.nn.sigmoid("sigX", x);
        x.mul("result", sigX);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 64);
        runReplayAccuracyTest("siluActivation", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * SiLU with changing inputs.
     */
    @Test
    public void testReplaySiLUActivationChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable sigX = sd.nn.sigmoid("sigX", x);
        x.mul("result", sigX);

        runReplayWithChangingInputs("siluChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 64)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // GLU (GATED LINEAR UNIT) TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * GLU: split input in half along feature dim, sigmoid(first) * second.
     * Used in PaLM, GLU-variant FFN blocks. Tests split + sigmoid + elementwise mul fusion.
     */
    @Test
    public void testReplayGLU() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 128);
        SDVariable[] halves = sd.split(new String[]{"gate_half", "value_half"}, x, 2, 1);
        SDVariable gate = sd.nn.sigmoid("gate", halves[0]);
        gate.mul("result", halves[1]);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 128);
        runReplayAccuracyTest("glu", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * GLU with changing inputs.
     */
    @Test
    public void testReplayGLUChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 128);
        SDVariable[] halves = sd.split(new String[]{"gate_half", "value_half"}, x, 2, 1);
        SDVariable gate = sd.nn.sigmoid("gate", halves[0]);
        gate.mul("result", halves[1]);

        runReplayWithChangingInputs("gluChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 128)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // LAYER NORM TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.
     * Unlike batch norm, reduces over the feature dimension (last axis), not batch.
     * This is the standard transformer layer norm pattern.
     */
    @Test
    public void testReplayLayerNorm() {
        SameDiff sd = SameDiff.create();
        int D = 64;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, D);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, D));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 1, D));
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));

        // mean and var across feature dimension (axis=1)
        SDVariable mean = x.mean("mean1", true, 1);     // [B, 1]
        SDVariable centered = x.sub("centered", mean);
        SDVariable var = centered.mul("sq", centered).mean("var1", true, 1);  // [B, 1]
        SDVariable varEps = var.add("vareps", eps);
        SDVariable invStd = sd.math.rsqrt("rsqrt1", varEps);
        SDVariable normed = centered.mul("normed", invStd);
        SDVariable scaled = normed.mul("scaled", gamma);
        scaled.add("result", beta);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, D);
        runReplayAccuracyTest("layerNorm", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Layer norm with changing inputs.
     */
    @Test
    public void testReplayLayerNormChangingInputs() {
        SameDiff sd = SameDiff.create();
        int D = 64;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, D);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, D));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 1, D));
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));

        SDVariable mean = x.mean("mean1", true, 1);
        SDVariable centered = x.sub("centered", mean);
        SDVariable var = centered.mul("sq", centered).mean("var1", true, 1);
        SDVariable varEps = var.add("vareps", eps);
        SDVariable invStd = sd.math.rsqrt("rsqrt1", varEps);
        SDVariable normed = centered.mul("normed", invStd);
        SDVariable scaled = normed.mul("scaled", gamma);
        scaled.add("result", beta);

        runReplayWithChangingInputs("layerNormChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 8, D)),
                "result", 1e-3);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // ROTARY POSITION EMBEDDINGS (RoPE) TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Simplified RoPE: apply sin/cos position encoding to Q/K via elementwise mul+add.
     * Pattern: q_rot = q * cos_pos - rotate_half(q) * sin_pos
     * Simplified here as q * cos + (-q) * sin to test multi-branch elementwise fusion
     * with trig-based constants.
     */
    @Test
    public void testReplayRoPE() {
        SameDiff sd = SameDiff.create();
        int B = 1, H = 4, S = 8, D = 16;

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, H, S, D);
        // Pre-computed sin/cos position tables [1, 1, S, D]
        SDVariable cosPos = sd.placeHolder("cosPos", DataType.FLOAT, 1, 1, S, D);
        SDVariable sinPos = sd.placeHolder("sinPos", DataType.FLOAT, 1, 1, S, D);

        // q_rotated = q * cos + (-q) * sin
        SDVariable qMulCos = q.mul("qcos", cosPos);
        SDVariable negQ = q.mul("negq", sd.constant("negOne", Nd4j.scalar(DataType.FLOAT, -1.0f)));
        SDVariable qMulSin = negQ.mul("qsin", sinPos);
        qMulCos.add("result", qMulSin);

        // Generate sin/cos tables
        INDArray cosArr = Nd4j.zeros(DataType.FLOAT, 1, 1, S, D);
        INDArray sinArr = Nd4j.zeros(DataType.FLOAT, 1, 1, S, D);
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                double freq = 1.0 / Math.pow(10000.0, (2.0 * (d / 2)) / D);
                cosArr.putScalar(new int[]{0, 0, s, d}, (float) Math.cos(s * freq));
                sinArr.putScalar(new int[]{0, 0, s, d}, (float) Math.sin(s * freq));
            }
        }

        INDArray qArr = Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1);
        runReplayAccuracyTest("rope", sd,
                Map.of("q", qArr, "cosPos", cosArr, "sinPos", sinArr), "result", TOLERANCE);
    }

    /**
     * RoPE with changing inputs (Q changes each step, positions stay fixed).
     */
    @Test
    public void testReplayRoPEChangingInputs() {
        SameDiff sd = SameDiff.create();
        int B = 1, H = 4, S = 8, D = 16;

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, H, S, D);
        SDVariable cosPos = sd.placeHolder("cosPos", DataType.FLOAT, 1, 1, S, D);
        SDVariable sinPos = sd.placeHolder("sinPos", DataType.FLOAT, 1, 1, S, D);

        SDVariable qMulCos = q.mul("qcos", cosPos);
        SDVariable negQ = q.mul("negq", sd.constant("negOne", Nd4j.scalar(DataType.FLOAT, -1.0f)));
        SDVariable qMulSin = negQ.mul("qsin", sinPos);
        qMulCos.add("result", qMulSin);

        // Fixed position tables
        INDArray cosArr = Nd4j.zeros(DataType.FLOAT, 1, 1, S, D);
        INDArray sinArr = Nd4j.zeros(DataType.FLOAT, 1, 1, S, D);
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                double freq = 1.0 / Math.pow(10000.0, (2.0 * (d / 2)) / D);
                cosArr.putScalar(new int[]{0, 0, s, d}, (float) Math.cos(s * freq));
                sinArr.putScalar(new int[]{0, 0, s, d}, (float) Math.sin(s * freq));
            }
        }
        final INDArray cosFixed = cosArr;
        final INDArray sinFixed = sinArr;

        runReplayWithChangingInputs("ropeChanging", sd,
                () -> Map.of("q", Nd4j.randn(DataType.FLOAT, B, H, S, D).mul(0.1),
                             "cosPos", cosFixed,
                             "sinPos", sinFixed),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // EMBEDDING LOOKUP + SCALE TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Embedding lookup + scale by sqrt(d_model).
     * Pattern: gather from embedding table -> multiply by sqrt(d_model).
     * Common in transformer encoder input (original "Attention is All You Need" scaling).
     */
    @Test
    public void testReplayEmbeddingLookupScale() {
        SameDiff sd = SameDiff.create();
        int vocabSize = 32, dModel = 64;
        SDVariable embedTable = sd.constant("embedTable", Nd4j.randn(DataType.FLOAT, vocabSize, dModel).mul(0.1));
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, -1);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, (float) Math.sqrt(dModel)));

        SDVariable embedded = sd.gather("gathered", embedTable, indices, 0);
        embedded.mul("result", scale);

        INDArray idxArr = Nd4j.createFromArray(new long[]{0, 5, 12, 31, 7, 22, 3, 15});
        runReplayAccuracyTest("embeddingScale", sd, Map.of("indices", idxArr), "result", TOLERANCE);
    }

    /**
     * Embedding lookup + scale with changing indices.
     */
    @Test
    public void testReplayEmbeddingLookupScaleChangingInputs() {
        SameDiff sd = SameDiff.create();
        int vocabSize = 32, dModel = 64;
        SDVariable embedTable = sd.constant("embedTable", Nd4j.randn(DataType.FLOAT, vocabSize, dModel).mul(0.1));
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, -1);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, (float) Math.sqrt(dModel)));

        SDVariable embedded = sd.gather("gathered", embedTable, indices, 0);
        embedded.mul("result", scale);

        runReplayWithChangingInputs("embeddingScaleChanging", sd,
                () -> {
                    long[] ids = new long[8];
                    for (int i = 0; i < 8; i++) ids[i] = (long) (Math.random() * 32);
                    return Map.of("indices", Nd4j.createFromArray(ids));
                },
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // CROSS-ATTENTION TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Cross-attention: Q from decoder, K/V from encoder.
     * Tests multi-source fusion where Q and K/V come from different inputs
     * with potentially different sequence lengths.
     */
    @Test
    public void testReplayCrossAttention() {
        SameDiff sd = SameDiff.create();
        int B = 1, H = 4, Sq = 4, Sk = 12, D = 16;
        // Q from decoder: [B, H, Sq, D]
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, H, Sq, D);
        // K from encoder: [B, H, Sk, D]
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, B, H, Sk, D);
        // V from encoder: [B, H, Sk, D]
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, B, H, Sk, D);

        // K^T: [B, H, D, Sk]
        SDVariable kT = sd.permute("kT", k, 0, 1, 3, 2);
        // scores: [B, H, Sq, Sk]
        SDVariable scores = sd.mmul("scores", q, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(D)));
        SDVariable scaled = scores.mul("scaled", scale);
        SDVariable attnWeights = sd.nn.softmax("attnWeights", scaled, -1);
        // output: [B, H, Sq, D]
        sd.mmul("result", attnWeights, v);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, B, H, Sq, D).mul(0.1);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, B, H, Sk, D).mul(0.1);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, B, H, Sk, D).mul(0.1);
        runReplayAccuracyTest("crossAttention", sd,
                Map.of("q", qArr, "k", kArr, "v", vArr), "result", 1e-3);
    }

    /**
     * Cross-attention with changing inputs.
     */
    @Test
    public void testReplayCrossAttentionChangingInputs() {
        SameDiff sd = SameDiff.create();
        int B = 1, H = 4, Sq = 4, Sk = 12, D = 16;
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, H, Sq, D);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, B, H, Sk, D);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, B, H, Sk, D);

        SDVariable kT = sd.permute("kT", k, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("scores", q, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(D)));
        SDVariable scaled = scores.mul("scaled", scale);
        SDVariable attnWeights = sd.nn.softmax("attnWeights", scaled, -1);
        sd.mmul("result", attnWeights, v);

        runReplayWithChangingInputs("crossAttnChanging", sd,
                () -> Map.of("q", Nd4j.randn(DataType.FLOAT, B, H, Sq, D).mul(0.1),
                             "k", Nd4j.randn(DataType.FLOAT, B, H, Sk, D).mul(0.1),
                             "v", Nd4j.randn(DataType.FLOAT, B, H, Sk, D).mul(0.1)),
                "result", 1e-3);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // CLAMP / CLIP VALUES TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Clamp/clip values: clipByValue(x, -1, 1).
     * Used for gradient clipping, activation bounds, and clamped attention logits.
     * Tests that clipByValue is correctly compiled and replayed.
     */
    @Test
    public void testReplayClampValues() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable clamped = sd.math.clipByValue("clamped", x, -1.0, 1.0);
        // Chain with another op to ensure fusion
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.1));
        clamped.add("result", bias);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32).mul(3.0); // large values to trigger clipping
        runReplayAccuracyTest("clampValues", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Clamp with changing inputs.
     */
    @Test
    public void testReplayClampValuesChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable clamped = sd.math.clipByValue("clamped", x, -1.0, 1.0);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.1));
        clamped.add("result", bias);

        runReplayWithChangingInputs("clampChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 32).mul(3.0)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // POWER + SCALE TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * x^2 pattern (used in variance computation and RMS norm).
     * Tests square + scale fusion.
     */
    @Test
    public void testReplaySquareScale() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable squared = sd.math.square("sq", x);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 0.5f));
        squared.mul("result", scale);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32);
        runReplayAccuracyTest("squareScale", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * x^3 pattern (used in GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))).
     * Tests pow + mul chain.
     */
    @Test
    public void testReplayGeluApproxCubic() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        // GELU approximation components
        SDVariable xCubed = x.pow("xcube", 3.0);
        SDVariable coeff = sd.constant("coeff", Nd4j.scalar(DataType.FLOAT, 0.044715f));
        SDVariable inner = x.add("inner", xCubed.mul("cubescale", coeff));
        SDVariable sqrtTwoPi = sd.constant("sqrtTwoPi", Nd4j.scalar(DataType.FLOAT, (float) Math.sqrt(2.0 / Math.PI)));
        SDVariable tanhArg = inner.mul("tanharg", sqrtTwoPi);
        SDVariable tanhVal = sd.math.tanh("tanhval", tanhArg);
        SDVariable onePlus = tanhVal.add("oneplus", sd.constant("one", Nd4j.scalar(DataType.FLOAT, 1.0f)));
        SDVariable half = sd.constant("half", Nd4j.scalar(DataType.FLOAT, 0.5f));
        SDVariable halfX = x.mul("halfx", half);
        halfX.mul("result", onePlus);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32);
        runReplayAccuracyTest("geluApproxCubic", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Power + scale with changing inputs.
     */
    @Test
    public void testReplaySquareScaleChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable squared = sd.math.square("sq", x);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 0.5f));
        squared.mul("result", scale);

        runReplayWithChangingInputs("squareScaleChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 32)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // NEGATIVE INDEXING IN REDUCTION TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Reduce sum along axis=-1 on 3D tensor. Tests that negative axis indices
     * are correctly resolved during compilation and replay.
     */
    @Test
    public void testReplayReduceNegativeAxis() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8, 16);
        // sum along axis=-1 (last dim)
        SDVariable summed = x.sum("sum1", -1);  // [4, 8]
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 8));
        summed.add("result", bias);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8, 16);
        runReplayAccuracyTest("reduceNegAxis1", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Reduce mean along axis=-2 on 3D tensor (reduces the middle dimension).
     */
    @Test
    public void testReplayReduceNegativeAxis2() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8, 16);
        // mean along axis=-2 (middle dim)
        SDVariable meaned = x.mean("mean1", true, -2);  // [4, 1, 16]
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 1, 16));
        meaned.add("result", bias);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8, 16);
        runReplayAccuracyTest("reduceNegAxis2", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Reduce with negative axis and changing inputs.
     */
    @Test
    public void testReplayReduceNegativeAxisChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8, 16);
        SDVariable summed = x.sum("sum1", -1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 8));
        summed.add("result", bias);

        runReplayWithChangingInputs("reduceNegAxisChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8, 16)),
                "result", 1e-3);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // VERY DEEP CHAIN (20+ OPS) STRESS TEST
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Very deep chain of 24 alternating add/mul/relu ops.
     * Stress tests long fusion segments — potential issues with register pressure,
     * intermediate buffer management, and segment boundary placement.
     */
    @Test
    public void testReplayVeryDeepChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable current = x;

        for (int i = 0; i < 8; i++) {
            // add -> mul -> relu (3 ops per block, 8 blocks = 24 ops)
            SDVariable addBias = sd.constant("bias_" + i, Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.01));
            current = current.add("add_" + i, addBias);
            SDVariable mulScale = sd.constant("scale_" + i, Nd4j.ones(DataType.FLOAT, 1, 32).add(0.01 * i));
            current = current.mul("mul_" + i, mulScale);
            current = sd.nn.relu("relu_" + i, current, 0);
        }
        sd.identity("result", current);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runReplayAccuracyTest("veryDeepChain", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Very deep chain with changing inputs.
     */
    @Test
    public void testReplayVeryDeepChainChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable current = x;

        for (int i = 0; i < 8; i++) {
            SDVariable addBias = sd.constant("bias_" + i, Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.01));
            current = current.add("add_" + i, addBias);
            SDVariable mulScale = sd.constant("scale_" + i, Nd4j.ones(DataType.FLOAT, 1, 32).add(0.01 * i));
            current = current.mul("mul_" + i, mulScale);
            current = sd.nn.relu("relu_" + i, current, 0);
        }
        sd.identity("result", current);

        runReplayWithChangingInputs("veryDeepChainChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 32)),
                "result", 1e-3);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // MIXED BROADCAST RANKS TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Three-way broadcast: [B,1,D] * [1,S,D] + [B,S,1].
     * Tests complex broadcast resolution across three operands with different
     * dimensions being broadcast in each. This pattern appears in attention
     * bias computation.
     */
    @Test
    public void testReplayMixedBroadcastRanks() {
        SameDiff sd = SameDiff.create();
        int B = 4, S = 8, D = 16;
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, B, 1, D);   // broadcast over S
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 1, S, D);   // broadcast over B
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, B, S, 1);   // broadcast over D

        SDVariable product = a.mul("ab", b);       // [B, S, D]
        product.add("result", c);                   // [B, S, D]

        INDArray aArr = Nd4j.randn(DataType.FLOAT, B, 1, D);
        INDArray bArr = Nd4j.randn(DataType.FLOAT, 1, S, D);
        INDArray cArr = Nd4j.randn(DataType.FLOAT, B, S, 1);
        runReplayAccuracyTest("mixedBroadcast", sd,
                Map.of("a", aArr, "b", bArr, "c", cArr), "result", TOLERANCE);
    }

    /**
     * Mixed broadcast with changing inputs.
     */
    @Test
    public void testReplayMixedBroadcastRanksChangingInputs() {
        SameDiff sd = SameDiff.create();
        int B = 4, S = 8, D = 16;
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, B, 1, D);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 1, S, D);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, B, S, 1);

        SDVariable product = a.mul("ab", b);
        product.add("result", c);

        runReplayWithChangingInputs("mixedBroadcastChanging", sd,
                () -> Map.of("a", Nd4j.randn(DataType.FLOAT, B, 1, D),
                             "b", Nd4j.randn(DataType.FLOAT, 1, S, D),
                             "c", Nd4j.randn(DataType.FLOAT, B, S, 1)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TRANSPOSE + ADD TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Transpose then add bias. Tests that non-contiguous (transposed) memory layout
     * is handled correctly when fused with broadcast add.
     * Pattern: [M, N] -> transpose -> [N, M] -> add bias [1, M]
     */
    @Test
    public void testReplayTransposeAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 8, 32);
        SDVariable transposed = sd.permute("transposed", x, 1, 0);  // [32, 8]
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable added = transposed.add("add1", bias);
        sd.nn.relu("result", added, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32);
        runReplayAccuracyTest("transposeAdd", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Transpose + add with changing inputs.
     */
    @Test
    public void testReplayTransposeAddChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 8, 32);
        SDVariable transposed = sd.permute("transposed", x, 1, 0);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable added = transposed.add("add1", bias);
        sd.nn.relu("result", added, 0);

        runReplayWithChangingInputs("transposeAddChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 32)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // TILE / REPEAT + COMPUTE TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Tile a tensor then process it. Tests that tile (repeat) creates correct
     * memory layout for subsequent fused elementwise ops.
     * Pattern: [1, D] -> tile [B, 1] -> [B, D] -> add bias -> relu
     */
    @Test
    public void testReplayTileCompute() {
        SameDiff sd = SameDiff.create();
        int B = 4, D = 32;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, D);
        // Tile: repeat along batch dimension
        SDVariable tiled = sd.tile("tiled", x, new int[]{B, 1});  // [B, D]
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, D));
        SDVariable added = tiled.add("add1", bias);
        sd.nn.relu("result", added, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, D);
        runReplayAccuracyTest("tileCompute", sd, Map.of("x", xArr), "result", TOLERANCE);
    }

    /**
     * Tile + compute with changing inputs.
     */
    @Test
    public void testReplayTileComputeChangingInputs() {
        SameDiff sd = SameDiff.create();
        int B = 4, D = 32;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, D);
        SDVariable tiled = sd.tile("tiled", x, new int[]{B, 1});
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, D));
        SDVariable added = tiled.add("add1", bias);
        sd.nn.relu("result", added, 0);

        runReplayWithChangingInputs("tileComputeChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 1, D)),
                "result", TOLERANCE);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // RECIPROCAL CHAIN TESTS
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Reciprocal chain: 1/x then multiply. Used in normalization denominators
     * (RMS norm computes 1/sqrt(mean(x^2)+eps) then multiplies).
     * Tests reciprocal + mul fusion.
     */
    @Test
    public void testReplayReciprocalChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        // Compute mean of squares (RMS-like)
        SDVariable sq = sd.math.square("sq", x);
        SDVariable meanSq = sq.mean("meansq", true, 1);  // [B, 1]
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable meanSqEps = meanSq.add("addeps", eps);
        SDVariable sqrtVal = sd.math.sqrt("sqrt1", meanSqEps);
        // reciprocal: 1/sqrt(mean(x^2)+eps)
        SDVariable invNorm = sd.math.reciprocal("recip", sqrtVal);
        // x * (1/norm)
        x.mul("result", invNorm);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 8, 32);
        runReplayAccuracyTest("reciprocalChain", sd, Map.of("x", xArr), "result", 1e-3);
    }

    /**
     * Reciprocal chain with changing inputs.
     */
    @Test
    public void testReplayReciprocalChainChangingInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable sq = sd.math.square("sq", x);
        SDVariable meanSq = sq.mean("meansq", true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable meanSqEps = meanSq.add("addeps", eps);
        SDVariable sqrtVal = sd.math.sqrt("sqrt1", meanSqEps);
        SDVariable invNorm = sd.math.reciprocal("recip", sqrtVal);
        x.mul("result", invNorm);

        runReplayWithChangingInputs("reciprocalChanging", sd,
                () -> Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 32)),
                "result", 1e-3);
    }
}
