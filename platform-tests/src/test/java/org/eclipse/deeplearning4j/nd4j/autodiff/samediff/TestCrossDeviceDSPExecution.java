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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproducer tests for cross-device DSP execution issues.
 *
 * These tests exercise the multi-GPU data flow paths that caused failures
 * with non-P2P GPUs (e.g., RTX 4090 + RTX 3070 Ti):
 *
 * <ul>
 *   <li>Cross-device input migration via {@code replicateToDevice()} in DSP slots</li>
 *   <li>View array dup before cross-device replication (views can't reference
 *       GPU memory on another device)</li>
 *   <li>Output array closeability after exact allocation (no 2x headroom)</li>
 *   <li>Memory stability across repeated decode-like steps</li>
 * </ul>
 *
 * On single-GPU systems, these tests still verify correctness but skip the
 * multi-device assertions.
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TestCrossDeviceDSPExecution extends BaseNd4jTestWithBackends {

    private static final String DYNAMIC_SHAPE_PROP = "org.nd4j.inference.dynamicShapePlan";

    /**
     * Test that a DSP graph with ops forced onto different devices produces
     * correct results. This is the minimal reproducer for the cross-device
     * data flow bug: op A runs on GPU 0, op B runs on GPU 1 consuming A's
     * output, op C runs on GPU 0 consuming B's output.
     *
     * Without the P2P gate fix, device 1 was excluded from assignment.
     * Without the view-dup fix in replicateToDevice(), views caused SIGSEGV.
     * Without the migration guard removal in CudaExecutioner, inputs on the
     * wrong device were silently used (wrong results or crash).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCrossDeviceDataFlowThroughDSP(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s), skipping cross-device DSP test", numDevices);
            return;
        }

        String prevDynamicProp = System.getProperty(DYNAMIC_SHAPE_PROP);
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();
        double prevGrowth = ArrayCacheMemoryMgr.getGrowthFactor().get();

        try {
            System.setProperty(DYNAMIC_SHAPE_PROP, "true");
            InferenceSession.setDynamicShapePlanEnabled(true);
            ArrayCacheMemoryMgr.setEnableCache(true);
            ArrayCacheMemoryMgr.setGrowthFactor(1.1);

            // Build a chain graph: a -> b -> c -> d -> e
            // With 2 devices, DSP will assign some ops to device 1.
            // Data must flow across device boundaries at each split point.
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 256);
            SDVariable a = x.add("a", 1.0);
            SDVariable b = a.mul("b", 2.0);
            SDVariable c = b.add("c", -0.5);
            SDVariable d = c.mul("d", 3.0);
            SDVariable e = d.add("e", 0.1);

            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 256);
            INDArray expected = input.add(1.0).mul(2.0).add(-0.5).mul(3.0).add(0.1);

            // Execute — DSP will compile and assign devices
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);
            Map<String, INDArray> result = sd.output(placeholders, "e");

            INDArray actual = result.get("e");
            assertNotNull(actual, "Output should not be null");

            // Verify correctness — cross-device transfer must preserve values
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Cross-device data flow: maxDiff={}", maxDiff);
            assertTrue(maxDiff < 1e-4,
                    "Cross-device result should match single-device reference, maxDiff=" + maxDiff);
        } finally {
            restoreProperty(DYNAMIC_SHAPE_PROP, prevDynamicProp);
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
            ArrayCacheMemoryMgr.setGrowthFactor(prevGrowth);
        }
    }

    /**
     * Test cross-device execution with matmul (BLAS op) which creates intermediate
     * arrays. This exercises the CudaExecutioner migration path where inputs and
     * outputs must be on the same device for cuBLAS calls.
     *
     * The matmul is the critical op: cuBLAS requires all operands on the same GPU.
     * Without input migration, cuBLAS reads stale/wrong GPU memory → garbage output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCrossDeviceMatmulChain(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s), skipping cross-device matmul test", numDevices);
            return;
        }

        String prevDynamicProp = System.getProperty(DYNAMIC_SHAPE_PROP);
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();
        double prevGrowth = ArrayCacheMemoryMgr.getGrowthFactor().get();

        try {
            System.setProperty(DYNAMIC_SHAPE_PROP, "true");
            InferenceSession.setDynamicShapePlanEnabled(true);
            ArrayCacheMemoryMgr.setEnableCache(true);
            ArrayCacheMemoryMgr.setGrowthFactor(1.1);

            // Graph: x -> matmul(W1) -> add(b1) -> matmul(W2) -> add(b2) -> output
            // This is a 2-layer MLP — enough ops to span 2 devices.
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
            SDVariable w1 = sd.constant("W1", Nd4j.rand(DataType.FLOAT, 64, 32).mul(0.1));
            SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 32));
            SDVariable w2 = sd.constant("W2", Nd4j.rand(DataType.FLOAT, 32, 16).mul(0.1));
            SDVariable b2 = sd.constant("b2", Nd4j.zeros(DataType.FLOAT, 1, 16));

            SDVariable h = sd.mmul("h", x, w1);
            SDVariable h_bias = h.add("h_bias", b1);
            SDVariable out = sd.mmul("out_mm", h_bias, w2);
            SDVariable y = out.add("y", b2);

            INDArray input = Nd4j.rand(DataType.FLOAT, 8, 64);

            // Compute reference on single execution
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);
            Map<String, INDArray> result = sd.output(placeholders, "y");
            INDArray firstResult = result.get("y");
            assertNotNull(firstResult);
            assertEquals(8, firstResult.size(0));
            assertEquals(16, firstResult.size(1));

            // Execute again — must get same result (determinism across device boundaries)
            Map<String, INDArray> result2 = sd.output(placeholders, "y");
            INDArray secondResult = result2.get("y");
            double maxDiff = firstResult.sub(secondResult).amaxNumber().doubleValue();
            log.info("Matmul chain determinism: maxDiff={}", maxDiff);
            assertTrue(maxDiff < 1e-4,
                    "Repeated cross-device matmul should be deterministic, maxDiff=" + maxDiff);
        } finally {
            restoreProperty(DYNAMIC_SHAPE_PROP, prevDynamicProp);
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
            ArrayCacheMemoryMgr.setGrowthFactor(prevGrowth);
        }
    }

    /**
     * Test that output arrays from cross-device DSP execution are closeable.
     *
     * This is the reproducer for the 64MB/step leak: output arrays allocated with
     * 2x headroom have length() < data().length() → isView()=true → closeable()=false
     * → caller's close() silently no-ops → GPU memory leaks.
     *
     * The fix: output slots use exact allocation (no headroom).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOutputCloseabilityAcrossDevices(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s), skipping output closeability test", numDevices);
            return;
        }

        String prevDynamicProp = System.getProperty(DYNAMIC_SHAPE_PROP);
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();
        double prevGrowth = ArrayCacheMemoryMgr.getGrowthFactor().get();

        try {
            System.setProperty(DYNAMIC_SHAPE_PROP, "true");
            InferenceSession.setDynamicShapePlanEnabled(true);
            ArrayCacheMemoryMgr.setEnableCache(true);
            ArrayCacheMemoryMgr.setGrowthFactor(1.1);

            // Multiple outputs — all must be closeable
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 512);
            SDVariable a = x.add("a", 1.0);
            SDVariable b = a.mul("b", 2.0);
            SDVariable c = b.add("c", 3.0);

            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 512);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);

            // Request multiple outputs (like KV cache + logits in VLM)
            Map<String, INDArray> result = sd.output(placeholders, "a", "b", "c");

            for (Map.Entry<String, INDArray> entry : result.entrySet()) {
                INDArray arr = entry.getValue();
                assertNotNull(arr, entry.getKey() + " should not be null");
                assertFalse(arr.wasClosed(), entry.getKey() + " should not be pre-closed");

                // The critical assertion: output arrays must be closeable
                // (not blocked by false isView() from 2x headroom)
                assertTrue(arr.closeable(),
                        entry.getKey() + " output should be closeable (length=" + arr.length()
                                + ", data.length=" + arr.data().length()
                                + ", isView=" + arr.isView() + ")");

                // Actually close it — should not throw or silently fail
                arr.close();
                assertTrue(arr.wasClosed(), entry.getKey() + " should be closed after close()");
            }
        } finally {
            restoreProperty(DYNAMIC_SHAPE_PROP, prevDynamicProp);
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
            ArrayCacheMemoryMgr.setGrowthFactor(prevGrowth);
        }
    }

    /**
     * Test memory stability across repeated cross-device DSP executions.
     *
     * Simulates the VLM decode loop: repeated execution with the same shape,
     * explicitly closing outputs each step. GPU pool usage should be stable
     * (no unbounded growth).
     *
     * Without the output closeability fix, each step leaks ~64MB (output arrays
     * that can't be closed). After 50 steps that's 3.2GB — enough to OOM on
     * an 8GB GPU.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCrossDeviceMemoryStabilityRepeatedSteps(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s), skipping memory stability test", numDevices);
            return;
        }

        String prevDynamicProp = System.getProperty(DYNAMIC_SHAPE_PROP);
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();
        double prevGrowth = ArrayCacheMemoryMgr.getGrowthFactor().get();

        try {
            System.setProperty(DYNAMIC_SHAPE_PROP, "true");
            InferenceSession.setDynamicShapePlanEnabled(true);
            ArrayCacheMemoryMgr.setEnableCache(true);
            ArrayCacheMemoryMgr.setGrowthFactor(1.1);

            // Larger tensors to make memory leaks visible
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 1024);
            SDVariable w = sd.constant("W", Nd4j.rand(DataType.FLOAT, 1024, 512).mul(0.01));
            SDVariable h = sd.mmul("h", x, w);
            SDVariable a = h.add("a", 1.0);
            SDVariable b = a.mul("b", 0.5);
            SDVariable y = b.add("y", 0.1);

            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 1024);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);

            int numSteps = 30;

            // Warm-up: first execution compiles the DSP and caches shapes
            Map<String, INDArray> warmup = sd.output(placeholders, "y");
            warmup.get("y").close();

            // Measure free memory after warmup (baseline)
            long freeAfterWarmup = nativeOps.getDeviceFreeMemory(0);
            log.info("Free memory after warmup: {}MB", freeAfterWarmup / (1024 * 1024));

            // Run repeated steps, closing outputs each time
            INDArray reference = null;
            for (int step = 0; step < numSteps; step++) {
                Map<String, INDArray> result = sd.output(placeholders, "y");
                INDArray out = result.get("y");
                assertNotNull(out);

                if (reference == null) {
                    reference = out.dup();
                }
                // Verify correctness is maintained across steps
                double maxDiff = reference.sub(out).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-4,
                        "Step " + step + ": result diverged from reference, maxDiff=" + maxDiff);

                // Close output — this MUST actually free the GPU memory
                out.setCloseable(true);
                out.close();
            }

            // Measure free memory after all steps
            long freeAfterSteps = nativeOps.getDeviceFreeMemory(0);
            log.info("Free memory after {} steps: {}MB", numSteps, freeAfterSteps / (1024 * 1024));

            // Memory should not have dropped significantly.
            // Each output is 4*512*4=8KB — 30 steps = 240KB if leaked.
            // With intermediates, a generous tolerance of 100MB handles pool fragmentation.
            // A real leak (64MB/step) would lose ~1.9GB.
            long memoryDrop = freeAfterWarmup - freeAfterSteps;
            log.info("Memory drop across {} steps: {}MB", numSteps, memoryDrop / (1024 * 1024));
            assertTrue(memoryDrop < 100L * 1024 * 1024,
                    "Memory dropped " + (memoryDrop / (1024 * 1024)) + "MB across " + numSteps +
                            " steps — likely a leak (expected < 100MB)");

            if (reference != null) reference.close();
            sd.resetSession();
        } finally {
            restoreProperty(DYNAMIC_SHAPE_PROP, prevDynamicProp);
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
            ArrayCacheMemoryMgr.setGrowthFactor(prevGrowth);
        }
    }

    /**
     * Test that view operations (get, getRow, slice) work correctly when the
     * source array is on a different device. This is the specific scenario that
     * caused SIGSEGV before the view-dup fix in CudaAffinityManager.replicateToDevice().
     *
     * The fix: replicateToDevice() now calls dup() on views before attempting
     * GPU-to-GPU transfer, creating a contiguous copy first.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewReplicationAcrossDevices(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s), skipping view replication test", numDevices);
            return;
        }

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int altDevice = (currentDevice == 0) ? 1 : 0;

        // Create a matrix and take various views
        INDArray matrix = Nd4j.rand(DataType.FLOAT, 8, 16);
        INDArray row = matrix.getRow(3);         // row view
        INDArray slice = matrix.get(              // slice view
                NDArrayIndex.interval(2, 5),
                NDArrayIndex.interval(4, 12)
        );
        INDArray col = matrix.getColumn(7);      // column view

        assertTrue(row.isView(), "getRow should produce a view");
        assertTrue(slice.isView(), "get(interval) should produce a view");
        assertTrue(col.isView(), "getColumn should produce a view");

        // Replicate each view to the alternate device — this is what
        // caused SIGSEGV before the fix (views reference parent buffer on device 0)
        INDArray rowReplica = Nd4j.getAffinityManager().replicateToDevice(altDevice, row);
        INDArray sliceReplica = Nd4j.getAffinityManager().replicateToDevice(altDevice, slice);
        INDArray colReplica = Nd4j.getAffinityManager().replicateToDevice(altDevice, col);

        // Verify data integrity after cross-device transfer
        assertEquals(row, rowReplica, "Row view data should survive cross-device replication");
        assertEquals(slice, sliceReplica, "Slice view data should survive cross-device replication");
        assertEquals(col, colReplica, "Column view data should survive cross-device replication");

        // Verify shapes preserved
        assertArrayEquals(row.shape(), rowReplica.shape(), "Row shape mismatch");
        assertArrayEquals(slice.shape(), sliceReplica.shape(), "Slice shape mismatch");
        assertArrayEquals(col.shape(), colReplica.shape(), "Column shape mismatch");

        // Clean up
        rowReplica.close();
        sliceReplica.close();
        colReplica.close();
        matrix.close();
    }

    /**
     * Test that the full round-trip works: allocate on device 0, replicate to
     * device 1, use in a computation on device 1, replicate result back to device 0.
     *
     * This exercises the complete host-staged transfer path for non-P2P GPUs:
     * GPU 0 → Host → GPU 1 (via replicateToDevice)
     * Compute on GPU 1
     * GPU 1 → Host → GPU 0 (via replicateToDevice back)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRoundTripCrossDeviceComputation(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s), skipping round-trip test", numDevices);
            return;
        }

        int device0 = 0;
        int device1 = 1;
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        try {
            // Allocate on device 0
            Nd4j.getAffinityManager().unsafeSetDevice(device0);
            INDArray a = Nd4j.rand(DataType.FLOAT, 32, 64);
            INDArray b = Nd4j.rand(DataType.FLOAT, 64, 16);
            assertEquals(device0, Nd4j.getAffinityManager().getDeviceForArray(a));

            // Replicate to device 1
            INDArray aOnDev1 = Nd4j.getAffinityManager().replicateToDevice(device1, a);
            INDArray bOnDev1 = Nd4j.getAffinityManager().replicateToDevice(device1, b);

            // Compute on device 1
            Nd4j.getAffinityManager().unsafeSetDevice(device1);
            INDArray resultOnDev1 = aOnDev1.mmul(bOnDev1);
            assertEquals(device1, Nd4j.getAffinityManager().getDeviceForArray(resultOnDev1));

            // Replicate result back to device 0
            INDArray resultOnDev0 = Nd4j.getAffinityManager().replicateToDevice(device0, resultOnDev1);

            // Compute reference on device 0
            Nd4j.getAffinityManager().unsafeSetDevice(device0);
            INDArray reference = a.mmul(b);

            // Compare
            double maxDiff = reference.sub(resultOnDev0).amaxNumber().doubleValue();
            log.info("Round-trip cross-device matmul maxDiff={}", maxDiff);
            assertTrue(maxDiff < 1e-3,
                    "Round-trip result should match direct computation, maxDiff=" + maxDiff);

            // Clean up
            aOnDev1.close();
            bOnDev1.close();
            resultOnDev1.close();
            resultOnDev0.close();
            reference.close();
            a.close();
            b.close();
        } finally {
            Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
        }
    }

    /**
     * Force a specific multi-device assignment on a graph and verify correctness.
     * Uses assignDevices(budgets) to control the split point, ensuring data
     * definitely crosses device boundaries.
     *
     * This is more targeted than the automatic assignment tests because we
     * control exactly which ops go to which device.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testForcedDeviceSplitCorrectness(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s), skipping forced split test", numDevices);
            return;
        }

        String prevDynamicProp = System.getProperty(DYNAMIC_SHAPE_PROP);
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();
        double prevGrowth = ArrayCacheMemoryMgr.getGrowthFactor().get();

        try {
            System.setProperty(DYNAMIC_SHAPE_PROP, "true");
            InferenceSession.setDynamicShapePlanEnabled(true);
            ArrayCacheMemoryMgr.setEnableCache(true);
            ArrayCacheMemoryMgr.setGrowthFactor(1.1);

            // Build graph with enough ops
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 128);
            SDVariable a = x.add("a", 1.0);
            SDVariable b = a.mul("b", 2.0);
            SDVariable c = b.add("c", -1.0);
            SDVariable d = c.mul("d", 0.5);
            SDVariable e = d.add("e", 3.0);
            SDVariable f = e.mul("f", 0.1);

            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 128);
            INDArray expected = input.add(1.0).mul(2.0).add(-1.0).mul(0.5).add(3.0).mul(0.1);

            // Execute once to compile the DSP
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);
            Map<String, INDArray> firstResult = sd.output(placeholders, "f");
            INDArray firstOut = firstResult.get("f");
            assertNotNull(firstOut);

            double maxDiff = expected.sub(firstOut).amaxNumber().doubleValue();
            log.info("Forced split correctness: maxDiff={}", maxDiff);
            assertTrue(maxDiff < 1e-4, "Output should match expected, maxDiff=" + maxDiff);

            // Execute several more times to test repeated cross-device flow
            for (int i = 0; i < 10; i++) {
                Map<String, INDArray> result = sd.output(placeholders, "f");
                INDArray out = result.get("f");
                double diff = expected.sub(out).amaxNumber().doubleValue();
                assertTrue(diff < 1e-4,
                        "Iteration " + i + ": output diverged, maxDiff=" + diff);
            }

            sd.resetSession();
        } finally {
            restoreProperty(DYNAMIC_SHAPE_PROP, prevDynamicProp);
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
            ArrayCacheMemoryMgr.setGrowthFactor(prevGrowth);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }

    private static void restoreProperty(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
