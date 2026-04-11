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
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.LinkedHashSet;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Utility methods for Triton GPU backend tests.
 * Provides common helper methods for compiling and executing native plans.
 */
@Slf4j
public class TritonTestUtils {

    private static final double TOLERANCE = 1e-4;

    /**
     * Compile a DynamicShapePlan to native executable.
     */
    public static Pointer compileNativePlan(DynamicShapePlan plan) {
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

    /**
     * Execute a compiled native plan and return output maps.
     */
    public static Map<String, INDArray> executeNativePlan(Pointer planHandle, DynamicShapePlan plan,
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

    /**
     * Resolve external inputs from placeholders.
     */
    public static INDArray[] resolveExternalInputs(DynamicShapePlan plan, SameDiff sd,
                                                    Map<String, INDArray> placeholders) {
        String[] extKeys = plan.getExternalInputKeys();
        INDArray[] extInputs = new INDArray[extKeys.length];
        for (int i = 0; i < extKeys.length; i++) {
            String varName = extKeys[i];
            INDArray arr = placeholders != null ? placeholders.get(varName) : null;
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
     * Run an op test: compile plan, execute, compare to reference.
     */
    public static void runOpTest(String testName, SameDiff sd, Map<String, INDArray> ph, String outputName) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, testName + ": plan is null");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) {
            log.info("Skipping {} (native executor not supported)", testName);
            return;
        }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // First execution (slot-by-slot, unfrozen) — use as reference.
            // Do NOT use sd.output() as reference — it returns zeros for certain
            // multi-op graph patterns (e.g., chains with constants + identity).
            Map<String, INDArray> refResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray refOutput = refResults.get(outputName).dup();
            assertNotNull(refOutput, testName + ": reference output is null");
            log.info("{}: ref shape={} sum={}", testName, refOutput.shape(), refOutput.sumNumber());

            // Second execution — verify reproducibility
            Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray nativeOutput = nativeResults.get(outputName);
            assertNotNull(nativeOutput, testName + ": native output is null");

            // Verify output shape matches reference
            assertArrayEquals(refOutput.shape(), nativeOutput.shape(),
                testName + ": shape mismatch");

            // Verify output values match reference
            double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOLERANCE,
                String.format("%s: max diff %.6f exceeds tolerance %.6f",
                    testName, maxDiff, TOLERANCE));

            log.info("{}: PASSED (maxDiff={:.6f})", testName, maxDiff);

        } catch (Exception e) {
            fail(testName + ": execution failed - " + e.getMessage(), e);
        } finally {
            if (planHandle != null) {
                planHandle.close();
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1/2: Replay validation helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Holds a compiled native plan handle and executes the SAME handle across
     * multiple iterations, exposing replay diagnostics (phase, unit count,
     * signature, addresses) for Phase 1/2 validation.
     *
     * Usage:
     *   ReplayValidationHelper helper = new ReplayValidationHelper(sd, outputName);
     *   helper.warmup(placeholders);
     *   for (int i = 0; i < N; i++) helper.iterate(placeholders);
     *   helper.assertReplayUnitsDecreased();
     *   helper.close();
     */
    public static class ReplayValidationHelper implements AutoCloseable {
        private final String testName;
        private final SameDiff sd;
        private final String outputName;
        private final DynamicShapePlan plan;
        private final Pointer planHandle;
        private final INDArray[] extInputs;
        // Per-segment, per-iteration tracking: Map<segIdx, List<signature>>
        private final Map<Integer, List<Long>> perSegmentSignatures = new LinkedHashMap<>();
        // Per-segment, per-iteration tracking: Map<segIdx, List<unitCount>>
        private final Map<Integer, List<Integer>> perSegmentUnitCounts = new LinkedHashMap<>();
        // Per-segment, per-iteration tracking: Map<segIdx, List<execCount>>
        private final Map<Integer, List<Integer>> perSegmentExecCounts = new LinkedHashMap<>();
        private INDArray referenceOutput;
        private boolean warmedUp = false;

        public ReplayValidationHelper(String testName, SameDiff sd,
                                      Map<String, INDArray> placeholders,
                                      String outputName) {
            this.testName = testName;
            this.sd = sd;
            this.outputName = outputName;

            // Compile reference output
            Map<String, INDArray> ref = sd.output(placeholders, outputName);
            this.referenceOutput = ref.get(outputName);
            assertNotNull(referenceOutput, testName + ": reference output is null");

            // Compile plan once
            this.plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
            assertNotNull(plan, testName + ": plan is null");
            this.planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                throw new SkipException(testName + " (native executor not supported)");
            }

            // Resolve external inputs
            this.extInputs = resolveExternalInputs(plan, sd, placeholders);
        }

        /** Warmup execution — establishes baseline for replay comparison. */
        public Map<String, INDArray> warmup() {
            Map<String, INDArray> result = executeOnce("warmup");
            warmedUp = true;
            recordReplayState();
            return result;
        }

        /** Execute one iteration, recording replay diagnostics. */
        public Map<String, INDArray> iterate(int iteration) {
            if (!warmedUp) {
                throw new IllegalStateException("Must call warmup() before iterate()");
            }
            Map<String, INDArray> result = executeOnce("iter" + iteration);
            recordReplayState();
            return result;
        }

        private Map<String, INDArray> executeOnce(String label) {
            return executeNativePlan(planHandle, plan, extInputs);
        }

        /** Record current replay state per segment from native plan. */
        private void recordReplayState() {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            int segCount = nativeOps.getPlanSegmentCount(planHandle);
            for (int i = 0; i < segCount; i++) {
                long sig = nativeOps.getPlanReplaySignatureHash(planHandle, i);
                int units = nativeOps.getPlanReplayUnitCount(planHandle, i);
                int execCount = nativeOps.getSegmentExecutionCount(planHandle, i);

                perSegmentSignatures.computeIfAbsent(i, k -> new ArrayList<>()).add(sig);
                perSegmentUnitCounts.computeIfAbsent(i, k -> new ArrayList<>()).add(units);
                perSegmentExecCounts.computeIfAbsent(i, k -> new ArrayList<>()).add(execCount);
            }
        }

        /** Verify output matches reference (shape + values). */
        public void verifyOutput(Map<String, INDArray> results) {
            INDArray nativeOutput = results.get(outputName);
            assertNotNull(nativeOutput, testName + ": native output is null");
            assertArrayEquals(referenceOutput.shape(), nativeOutput.shape(),
                testName + ": shape mismatch");
            double maxDiff = referenceOutput.sub(nativeOutput).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOLERANCE,
                String.format("%s: max diff %.6f exceeds tolerance %.6f",
                    testName, maxDiff, TOLERANCE));
        }

        /** Assert replay signature is stable per segment across iterations. */
        public void assertSignatureStable() {
            for (Map.Entry<Integer, List<Long>> entry : perSegmentSignatures.entrySet()) {
                int segIdx = entry.getKey();
                List<Long> sigs = entry.getValue();
                if (sigs.size() < 2) continue;
                long first = sigs.get(0);
                for (int i = 1; i < sigs.size(); i++) {
                    assertEquals(first, sigs.get(i),
                        testName + ": seg[" + segIdx + "] replay signature changed at iteration " + i);
                }
                log.info("{}: seg[{}] replay signature stable across {} iterations (hash={})",
                    testName, segIdx, sigs.size(), Long.toHexString(first));
            }
        }

        /** Assert replay unit count is within expected range per segment. */
        public void assertReplayUnitCount(int expectedMax) {
            for (Map.Entry<Integer, List<Integer>> entry : perSegmentUnitCounts.entrySet()) {
                int segIdx = entry.getKey();
                for (int units : entry.getValue()) {
                    assertTrue(units <= expectedMax,
                        testName + ": seg[" + segIdx + "] replay unit count " + units + " exceeds max " + expectedMax);
                }
            }
            log.info("{}: replay unit counts per segment: {}", testName, perSegmentUnitCounts);
        }

        /** Assert replay state was recorded for every iteration on every observed segment. */
        public void assertRecordedStateSamples(int expectedSamples) {
            assertFalse(perSegmentSignatures.isEmpty(),
                    testName + ": no per-segment replay state was recorded");
            for (Map.Entry<Integer, List<Long>> entry : perSegmentSignatures.entrySet()) {
                int segIdx = entry.getKey();
                assertEquals(expectedSamples, entry.getValue().size(),
                        testName + ": seg[" + segIdx + "] signature sample count mismatch");
                List<Integer> unitCounts = perSegmentUnitCounts.get(segIdx);
                assertNotNull(unitCounts,
                        testName + ": seg[" + segIdx + "] missing replay unit samples");
                assertEquals(expectedSamples, unitCounts.size(),
                        testName + ": seg[" + segIdx + "] unit-count sample count mismatch");
                List<Integer> execCounts = perSegmentExecCounts.get(segIdx);
                assertNotNull(execCounts,
                        testName + ": seg[" + segIdx + "] missing execution count samples");
                assertEquals(expectedSamples, execCounts.size(),
                        testName + ": seg[" + segIdx + "] execution-count sample count mismatch");
            }
            log.info("{}: recorded {} replay-state samples across segments {}",
                    testName, expectedSamples, perSegmentSignatures.keySet());
        }

        /** Assert at least one segment replayed as a multi-unit schedule. */
        public void assertAnySegmentHasReplayUnitsAtLeast(int expectedMin) {
            for (Map.Entry<Integer, List<Integer>> entry : perSegmentUnitCounts.entrySet()) {
                int segIdx = entry.getKey();
                for (int units : entry.getValue()) {
                    if (units >= expectedMin) {
                        log.info("{}: seg[{}] reached replay unit count {} (expected >= {})",
                                testName, segIdx, units, expectedMin);
                        return;
                    }
                }
            }
            fail(testName + ": expected at least one segment to reach replay unit count >= "
                    + expectedMin + ", counts=" + perSegmentUnitCounts);
        }

        /** Assert at least one segment exposed non-zero replay metadata. */
        public void assertAnySegmentHasReplayMetadata() {
            for (Map.Entry<Integer, List<Long>> entry : perSegmentSignatures.entrySet()) {
                int segIdx = entry.getKey();
                List<Long> sigs = entry.getValue();
                List<Integer> counts = perSegmentExecCounts.get(segIdx);
                for (int i = 0; i < sigs.size(); i++) {
                    long sig = sigs.get(i);
                    int execCount = counts != null && i < counts.size() ? counts.get(i) : -1;
                    if (sig != 0L && execCount >= 0) {
                        log.info("{}: seg[{}] exposed replay metadata sig={} execCount={}",
                                testName, segIdx, Long.toHexString(sig), execCount);
                        return;
                    }
                }
            }
            fail(testName + ": expected at least one segment to expose non-zero replay metadata; signatures="
                    + perSegmentSignatures + " execCounts=" + perSegmentExecCounts);
        }

        /** Assert segment execution counts are monotonically increasing. */
        public void assertExecutionCountsMonotonic() {
            for (Map.Entry<Integer, List<Integer>> entry : perSegmentExecCounts.entrySet()) {
                int segIdx = entry.getKey();
                List<Integer> counts = entry.getValue();
                for (int i = 1; i < counts.size(); i++) {
                    assertTrue(counts.get(i) >= counts.get(i - 1),
                        testName + ": seg[" + segIdx + "] execution count decreased: " +
                        counts.get(i - 1) + " -> " + counts.get(i));
                }
                log.info("{}: seg[{}] execution counts monotonically increasing: {}",
                    testName, segIdx, counts);
            }
        }

        @Override
        public void close() {
            if (planHandle != null) {
                planHandle.close();
            }
        }
    }

    /**
     * Exception to skip tests that require native executor support.
     */
    public static class SkipException extends RuntimeException {
        public SkipException(String message) {
            super(message);
        }
    }
}
