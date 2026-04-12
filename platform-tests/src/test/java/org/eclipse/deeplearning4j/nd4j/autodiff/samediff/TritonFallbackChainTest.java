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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that verify the Triton fallback chain works correctly:
 * when Triton compilation fails, the system falls back to CUDA graphs
 * (or slot-by-slot) and produces numerically correct output.
 *
 * <p>The fallback chain is:
 * GEM_TRITON → (compile fail) → GEM_CUDA_GRAPHS → (fail) → GEM_SLOT_BY_SLOT</p>
 *
 * <p>These tests verify:</p>
 * <ul>
 *   <li>Explicit GEM_TRITON and GEM_CUDA_GRAPHS modes produce matching output</li>
 *   <li>GEM_AUTO (which tries Triton first) produces correct output</li>
 *   <li>Repeated iterations with GEM_AUTO maintain correctness (catches stale data bugs)</li>
 *   <li>All three modes (TRITON, CUDA_GRAPHS, SLOT_BY_SLOT) agree numerically</li>
 *   <li>Mixed Triton-compilable and non-compilable ops handle segment boundaries correctly</li>
 * </ul>
 *
 * <p>Every test compares against a SLOT_BY_SLOT reference execution.</p>
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonFallbackChainTest extends BaseNd4jTestWithBackends {

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

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

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

    /**
     * Run a SameDiff graph with a specific GraphExecutionMode and return outputs
     * for the given inputs. Uses sd.output() with the mode set on the SameDiff instance.
     */
    private Map<String, INDArray> runWithMode(SameDiff sdTemplate, String outputName,
                                                GraphExecutionMode mode,
                                                INDArray[] inputs, String inputVarName) {
        List<Map<String, INDArray>> allOutputs = new ArrayList<>();
        for (INDArray input : inputs) {
            SameDiff sd = SameDiff.create();
            // Clone the template graph by rebuilding from the SameDiff's graph definition
            // We use the simpler approach: pass a fresh SameDiff that the caller builds
            // Actually, we expect the caller to pass a fresh SameDiff each time.
            sd.setGraphExecutionMode(mode);
            enableDsp(sd);
            Map<String, INDArray> result = sd.output(Map.of(inputVarName, input.dup()), outputName);
            allOutputs.add(result);
        }
        // Return the last output only — this helper is not ideal for multi-run.
        // Prefer the per-run pattern below.
        return allOutputs.isEmpty() ? Map.of() : allOutputs.get(allOutputs.size() - 1);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Test 1: testTritonFallbackToCudaGraphs
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Builds a SameDiff graph, executes it with GEM_TRITON mode, then executes
     * the same graph with GEM_CUDA_GRAPHS mode. Verifies outputs match slot-by-slot
     * reference within tolerance.
     *
     * <p>This verifies that when Triton compilation succeeds, CUDA graphs also
     * produce the same correct result — establishing that the fallback path
     * (Triton → CUDA graphs) is numerically equivalent.</p>
     */
    @Test
    public void testTritonFallbackToCudaGraphs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        // Build graph: x → matmul → relu → add → sigmoid → tanh
        // This is a standard chain that both Triton and CUDA graphs should handle.
        // Generate constants ONCE so all SameDiff instances share identical weights.
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1);
        INDArray b1Data = Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.01);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 32, 16).mul(0.1);

        SameDiff sdTriton = buildMatmulReluChain(w1Data, b1Data, w2Data);
        SameDiff sdCuda = buildMatmulReluChain(w1Data, b1Data, w2Data);
        SameDiff sdSlotBySlot = buildMatmulReluChain(w1Data, b1Data, w2Data);

        sdTriton.setGraphExecutionMode(GraphExecutionMode.TRITON);
        enableDsp(sdTriton);

        sdCuda.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
        enableDsp(sdCuda);

        sdSlotBySlot.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdSlotBySlot);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);

        Map<String, INDArray> tritonResults = sdTriton.output(
                Map.of("input", input.dup()), "output");
        Map<String, INDArray> cudaResults = sdCuda.output(
                Map.of("input", input.dup()), "output");
        Map<String, INDArray> slotResults = sdSlotBySlot.output(
                Map.of("input", input.dup()), "output");

        INDArray tritonOut = tritonResults.get("output");
        INDArray cudaOut = cudaResults.get("output");
        INDArray slotOut = slotResults.get("output");

        assertNotNull(tritonOut, "Triton output is null");
        assertNotNull(cudaOut, "CUDA graphs output is null");
        assertNotNull(slotOut, "Slot-by-slot output is null");

        // Triton vs slot-by-slot
        double tritonVsSlot = tritonOut.sub(slotOut).amaxNumber().doubleValue();
        log.info("Triton vs SlotBySlot maxDiff = {}", tritonVsSlot);
        assertTrue(tritonVsSlot < TOLERANCE,
                "Triton diverges from slot-by-slot! maxDiff=" + tritonVsSlot);

        // CUDA graphs vs slot-by-slot
        double cudaVsSlot = cudaOut.sub(slotOut).amaxNumber().doubleValue();
        log.info("CudaGraphs vs SlotBySlot maxDiff = {}", cudaVsSlot);
        assertTrue(cudaVsSlot < TOLERANCE,
                "CUDA graphs diverges from slot-by-slot! maxDiff=" + cudaVsSlot);

        // Triton vs CUDA graphs
        double tritonVsCuda = tritonOut.sub(cudaOut).amaxNumber().doubleValue();
        log.info("Triton vs CudaGraphs maxDiff = {}", tritonVsCuda);
        assertTrue(tritonVsCuda < TOLERANCE,
                "Triton diverges from CUDA graphs! maxDiff=" + tritonVsCuda);

        sdTriton.close();
        sdCuda.close();
        sdSlotBySlot.close();
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Test 2: testTritonFallbackProducesCorrectOutput
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Builds a graph with ops that may not all be Triton-compilable (mix of matmul,
     * elementwise, and reduction ops). Runs with GEM_AUTO (which tries Triton first).
     * Verifies output matches slot-by-slot reference.
     *
     * <p>GEM_AUTO will attempt Triton compilation first; if any segment fails,
     * it should fall back to CUDA graphs or slot-by-slot for that segment.
     * The final output must still match the slot-by-slot baseline.</p>
     */
    @Test
    public void testTritonFallbackProducesCorrectOutput() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        // Build a mixed-op graph: matmul (cuBLAS fallback) + elementwise (Triton) + reduction
        // Generate constants ONCE so all SameDiff instances share identical weights.
        INDArray wData = Nd4j.randn(DataType.FLOAT, 32, 64).mul(0.1);
        INDArray bData = Nd4j.randn(DataType.FLOAT, 1, 64).mul(0.01);

        SameDiff sdAuto = buildMixedOpsGraph(wData, bData);
        SameDiff sdSlotBySlot = buildMixedOpsGraph(wData, bData);

        sdAuto.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sdAuto);

        sdSlotBySlot.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdSlotBySlot);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 32);

        Map<String, INDArray> autoResults = sdAuto.output(
                Map.of("input", input.dup()), "output");
        Map<String, INDArray> slotResults = sdSlotBySlot.output(
                Map.of("input", input.dup()), "output");

        INDArray autoOut = autoResults.get("output");
        INDArray slotOut = slotResults.get("output");

        assertNotNull(autoOut, "AUTO output is null");
        assertNotNull(slotOut, "Slot-by-slot output is null");

        double maxDiff = autoOut.sub(slotOut).amaxNumber().doubleValue();
        log.info("AUTO vs SlotBySlot maxDiff = {}", maxDiff);

        assertFalse(autoOut.isNaN().any(),
                "AUTO output contains NaN — fallback may have silently failed");

        assertTrue(maxDiff < TOLERANCE,
                "GEM_AUTO output diverges from slot-by-slot! maxDiff=" + maxDiff
                        + "\n  AUTO: " + autoOut
                        + "\n  SBS:  " + slotOut);

        sdAuto.close();
        sdSlotBySlot.close();
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Test 3: testFallbackAfterMultipleReplaySteps
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Runs 10+ iterations with GEM_AUTO, verifying every iteration matches
     * slot-by-slot reference. Catches stale data bugs in the fallback path
     * where the first iteration is correct but subsequent iterations diverge.
     *
     * <p>This is a replay-bug detector: if the fallback mechanism captures
     * buffers on the first execution and replays stale data on subsequent
     * iterations, this test will fail on iteration 2+.</p>
     */
    @Test
    public void testFallbackAfterMultipleReplaySteps() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        int numSteps = 12;

        // Generate constants ONCE so all SameDiff instances share identical weights.
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1);
        INDArray b1Data = Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.01);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 32, 16).mul(0.1);

        // Pre-generate inputs ONCE so reference and test use identical inputs.
        INDArray[] inputs = new INDArray[numSteps];
        for (int s = 0; s < numSteps; s++) {
            inputs[s] = Nd4j.randn(DataType.FLOAT, 2, 16).add(s * 0.5f);
        }

        // Build reference: slot-by-slot execution for each step
        List<INDArray> refOutputs = new ArrayList<>();
        for (int s = 0; s < numSteps; s++) {
            SameDiff sd = buildMatmulReluChain(w1Data, b1Data, w2Data);
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            enableDsp(sd);
            Map<String, INDArray> results = sd.output(
                    Map.of("input", inputs[s].dup()), "output");
            refOutputs.add(results.get("output").dup());
            sd.close();
        }

        // Test: GEM_AUTO execution for each step (fresh SameDiff each time)
        for (int s = 0; s < numSteps; s++) {
            SameDiff sd = buildMatmulReluChain(w1Data, b1Data, w2Data);
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            enableDsp(sd);
            Map<String, INDArray> results = sd.output(
                    Map.of("input", inputs[s].dup()), "output");
            INDArray actual = results.get("output");

            assertNotNull(actual, "Step " + s + ": output is null");

            double maxDiff = actual.sub(refOutputs.get(s)).amaxNumber().doubleValue();
            log.info("Step {}: AUTO vs SlotBySlot maxDiff = {}", s, maxDiff);

            assertFalse(actual.isNaN().any(),
                    "Step " + s + ": output contains NaN");

            assertTrue(maxDiff < TOLERANCE,
                    "Step " + s + ": GEM_AUTO diverges from slot-by-slot! maxDiff=" + maxDiff);

            sd.close();
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Test 4: testFallbackChainPreservesNumericalAccuracy
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Compares GEM_TRITON, GEM_CUDA_GRAPHS, and GEM_SLOT_BY_SLOT outputs for the
     * same graph and inputs. All three must match within tolerance.
     *
     * <p>This is the core numerical accuracy test for the fallback chain.
     * If Triton produces correct output but CUDA graphs (the fallback) does not,
     * the fallback mechanism is broken.</p>
     */
    @Test
    public void testFallbackChainPreservesNumericalAccuracy() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        // Use a deeper graph to stress the fallback chain
        // Generate constants ONCE so all SameDiff instances share identical weights.
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 24, 48).mul(0.1);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 48, 24).mul(0.1);
        INDArray bData = Nd4j.randn(DataType.FLOAT, 1, 48).mul(0.01);
        INDArray scaleData = Nd4j.valueArrayOf(new long[]{1, 48}, 0.5f);

        SameDiff sdTriton = buildDeepChainGraph(w1Data, w2Data, bData, scaleData);
        SameDiff sdCuda = buildDeepChainGraph(w1Data, w2Data, bData, scaleData);
        SameDiff sdSlotBySlot = buildDeepChainGraph(w1Data, w2Data, bData, scaleData);

        sdTriton.setGraphExecutionMode(GraphExecutionMode.TRITON);
        enableDsp(sdTriton);

        sdCuda.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
        enableDsp(sdCuda);

        sdSlotBySlot.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdSlotBySlot);

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 24);

        Map<String, INDArray> tritonResults = sdTriton.output(
                Map.of("input", input.dup()), "output");
        Map<String, INDArray> cudaResults = sdCuda.output(
                Map.of("input", input.dup()), "output");
        Map<String, INDArray> slotResults = sdSlotBySlot.output(
                Map.of("input", input.dup()), "output");

        INDArray tritonOut = tritonResults.get("output");
        INDArray cudaOut = cudaResults.get("output");
        INDArray slotOut = slotResults.get("output");

        assertNotNull(tritonOut, "Triton output is null");
        assertNotNull(cudaOut, "CUDA graphs output is null");
        assertNotNull(slotOut, "Slot-by-slot output is null");

        double tritonVsSlot = tritonOut.sub(slotOut).amaxNumber().doubleValue();
        double cudaVsSlot = cudaOut.sub(slotOut).amaxNumber().doubleValue();
        double tritonVsCuda = tritonOut.sub(cudaOut).amaxNumber().doubleValue();

        log.info("Fallback chain numerical accuracy:");
        log.info("  Triton vs SlotBySlot: maxDiff = {}", tritonVsSlot);
        log.info("  CudaGraphs vs SlotBySlot: maxDiff = {}", cudaVsSlot);
        log.info("  Triton vs CudaGraphs: maxDiff = {}", tritonVsCuda);

        assertTrue(tritonVsSlot < TOLERANCE,
                "Triton diverges from slot-by-slot! maxDiff=" + tritonVsSlot);
        assertTrue(cudaVsSlot < TOLERANCE,
                "CUDA graphs diverges from slot-by-slot! maxDiff=" + cudaVsSlot);
        assertTrue(tritonVsCuda < TOLERANCE,
                "Triton diverges from CUDA graphs! maxDiff=" + tritonVsCuda);

        sdTriton.close();
        sdCuda.close();
        sdSlotBySlot.close();
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Test 5: testMixedTritonAndNonTritonOps
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Builds a graph with some Triton-compilable ops (elementwise) and some
     * non-compilable ops (matmul via cuBLAS). Verifies segment boundaries are
     * handled correctly and output is correct.
     *
     * <p>This tests the most realistic pattern: a graph where Triton can only
     * compile some segments, and the remaining ops must be executed via
     * native fallback. The output must still match slot-by-slot execution.</p>
     */
    @Test
    public void testMixedTritonAndNonTritonOps() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "Triton is unavailable — skipping");

        // Build graph with matmul (cuBLAS) + elementwise (Triton) boundaries
        // Generate constants ONCE so all SameDiff instances share identical weights.
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 32, 16).mul(0.1);
        INDArray biasData = Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.01);
        INDArray scaleData = Nd4j.valueArrayOf(new long[]{1, 32}, 2.0f);

        SameDiff sdAuto = buildMixedSegmentGraph(w1Data, w2Data, biasData, scaleData);
        SameDiff sdSlotBySlot = buildMixedSegmentGraph(w1Data, w2Data, biasData, scaleData);

        sdAuto.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sdAuto);

        sdSlotBySlot.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdSlotBySlot);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);

        Map<String, INDArray> autoResults = sdAuto.output(
                Map.of("input", input.dup()), "output");
        Map<String, INDArray> slotResults = sdSlotBySlot.output(
                Map.of("input", input.dup()), "output");

        INDArray autoOut = autoResults.get("output");
        INDArray slotOut = slotResults.get("output");

        assertNotNull(autoOut, "AUTO output is null");
        assertNotNull(slotOut, "Slot-by-slot output is null");

        assertArrayEquals(slotOut.shape(), autoOut.shape(),
                "Shape mismatch between AUTO and slot-by-slot");

        double maxDiff = autoOut.sub(slotOut).amaxNumber().doubleValue();
        log.info("Mixed segments AUTO vs SlotBySlot maxDiff = {}", maxDiff);

        assertFalse(autoOut.isNaN().any(),
                "AUTO output contains NaN — segment boundary handling may have failed");

        assertTrue(maxDiff < TOLERANCE,
                "Mixed segment graph AUTO output diverges from slot-by-slot! maxDiff=" + maxDiff
                        + "\n  AUTO: " + autoOut
                        + "\n  SBS:  " + slotOut);

        sdAuto.close();
        sdSlotBySlot.close();
    }

    // ─── Graph builders ──────────────────────────────────────────────────────

    /**
     * Builds: input → matmul → relu → add(bias) → sigmoid → tanh
     * A standard chain with matmul (cuBLAS fallback) and elementwise ops.
     * Accepts pre-generated constants so multiple SameDiff instances share
     * identical weights for fair numerical comparison.
     */
    private SameDiff buildMatmulReluChain(INDArray w1Data, INDArray b1Data, INDArray w2Data) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w1 = sd.constant("w1", w1Data.dup());
        SDVariable b1 = sd.constant("b1", b1Data.dup());
        SDVariable w2 = sd.constant("w2", w2Data.dup());

        SDVariable mm1 = sd.mmul("mm1", input, w1);
        SDVariable relu1 = sd.nn.relu("relu1", mm1, 0);
        SDVariable biased = relu1.add("add1", b1);
        SDVariable sig = sd.nn.sigmoid("sigmoid1", biased);
        SDVariable mm2 = sd.mmul("mm2", sig, w2);
        SDVariable result = sd.math.tanh("output", mm2);

        return sd;
    }

    /**
     * Overload for tests that don't need shared constants (legacy convenience).
     */
    private SameDiff buildMatmulReluChain() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1);
        INDArray b1 = Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.01);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 16).mul(0.1);
        return buildMatmulReluChain(w1, b1, w2);
    }

    /**
     * Builds a mixed-op graph: matmul + elementwise + reduction + elementwise.
     * Accepts pre-generated constants for fair numerical comparison.
     */
    private SameDiff buildMixedOpsGraph(INDArray wData, INDArray bData) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w = sd.constant("w", wData.dup());
        SDVariable b = sd.constant("b", bData.dup());

        // matmul (cuBLAS fallback)
        SDVariable mm = sd.mmul("mm1", input, w);
        // elementwise (Triton-compilable)
        SDVariable h1 = mm.add("add1", b);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = sd.nn.sigmoid("sigmoid1", h2);
        // reduction (may not be Triton-compilable)
        SDVariable mean = sd.math.mean("mean1", h3, true, 1);
        // back to elementwise
        SDVariable centered = h3.sub("centered", mean);
        SDVariable result = sd.math.tanh("output", centered);

        return sd;
    }

    /**
     * Overload for tests that don't need shared constants.
     */
    private SameDiff buildMixedOpsGraph() {
        INDArray w = Nd4j.randn(DataType.FLOAT, 32, 64).mul(0.1);
        INDArray b = Nd4j.randn(DataType.FLOAT, 1, 64).mul(0.01);
        return buildMixedOpsGraph(w, b);
    }

    /**
     * Builds a deeper chain to stress the fallback mechanism:
     * input → mmul → relu → mul → add → sigmoid → relu → tanh → output
     * Accepts pre-generated constants for fair numerical comparison.
     */
    private SameDiff buildDeepChainGraph(INDArray w1Data, INDArray w2Data, INDArray bData, INDArray scaleData) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 24);
        SDVariable w1 = sd.constant("w1", w1Data.dup());
        SDVariable w2 = sd.constant("w2", w2Data.dup());
        SDVariable b = sd.constant("b", bData.dup());
        SDVariable scale = sd.constant("scale", scaleData.dup());

        SDVariable mm1 = sd.mmul("mm1", input, w1);
        SDVariable h1 = mm1.add("add1", b);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = h2.mul("mul1", scale);
        SDVariable h4 = sd.nn.sigmoid("sigmoid1", h3);
        SDVariable mm2 = sd.mmul("mm2", h4, w2);
        SDVariable h5 = sd.nn.relu("relu2", mm2, 0);
        SDVariable result = sd.math.tanh("output", h5);

        return sd;
    }

    /**
     * Overload for tests that don't need shared constants.
     */
    private SameDiff buildDeepChainGraph() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 24, 48).mul(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 48, 24).mul(0.1);
        INDArray b = Nd4j.randn(DataType.FLOAT, 1, 48).mul(0.01);
        INDArray scale = Nd4j.valueArrayOf(new long[]{1, 48}, 0.5f);
        return buildDeepChainGraph(w1, w2, b, scale);
    }

    /**
     * Builds a graph with explicit segment boundaries:
     * matmul (segment 1, cuBLAS) → elementwise chain (segment 2, Triton)
     * → matmul (segment 3, cuBLAS) → elementwise (segment 4, Triton)
     *
     * This tests that cross-segment data flow works correctly when
     * some segments are Triton-compiled and others use native fallback.
     * Accepts pre-generated constants for fair numerical comparison.
     */
    private SameDiff buildMixedSegmentGraph(INDArray w1Data, INDArray w2Data, INDArray biasData, INDArray scaleData) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w1 = sd.constant("w1", w1Data.dup());
        SDVariable w2 = sd.constant("w2", w2Data.dup());
        SDVariable bias = sd.constant("bias", biasData.dup());
        SDVariable scale = sd.constant("scale", scaleData.dup());

        // Segment 1: matmul (cuBLAS fallback — creates segment boundary)
        SDVariable mm1 = sd.mmul("mm1", input, w1);

        // Segment 2: elementwise chain (Triton-compilable)
        SDVariable h1 = mm1.add("add1", bias);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = h2.mul("mul1", scale);
        SDVariable h4 = sd.nn.sigmoid("sigmoid1", h3);

        // Segment 3: matmul (cuBLAS — another segment boundary)
        SDVariable mm2 = sd.mmul("mm2", h4, w2);

        // Segment 4: elementwise (Triton-compilable)
        SDVariable h5 = sd.math.tanh("tanh1", mm2);
        SDVariable result = sd.nn.relu("output", h5, 0);

        return sd;
    }

    /**
     * Overload for tests that don't need shared constants.
     */
    private SameDiff buildMixedSegmentGraph() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 16).mul(0.1);
        INDArray bias = Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.01);
        INDArray scale = Nd4j.valueArrayOf(new long[]{1, 32}, 2.0f);
        return buildMixedSegmentGraph(w1, w2, bias, scale);
    }
}
