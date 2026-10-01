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
import org.bytedeco.javacpp.LongPointer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Isolated;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * DSP segment inputs held in host-resident memory. Past the device budget, the CUDA pool serves
 * an allocation from pinned host or CPU-preferred managed memory tagged for the allocating device
 * (the full-GPU case). A consumer on the tagged device reads such a buffer in place, a consumer
 * on another device needs one cached copy, and a cross-device copy may itself be served that way.
 * None of these may allocate per call, move the caller's buffer, or fail the plan.
 */
@Slf4j
@Tag("cuda-only")
@Isolated("Mutates the process-wide CUDA device memory budget")
public class DspHostResidentMigrationTest extends BaseND4JTest {
    private static final long MiB = 1024L * 1024;
    // 16 MiB: below DynamicShapePlan's 32 MiB device-affinity threshold, so the
    // weight does not steer placement.
    private static final int ROWS = 2048;
    private static final int COLS = 2048;
    private static final int WARMUP_CALLS = 16;
    private static final int MEASURED_CALLS = 16;

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @AfterEach
    void resetBudget() {
        // -1 is the unbounded default; a leftover bound would push later tests onto failover memory.
        Nd4j.getEnvironment().setMaxDeviceMemory(-1L);
    }

    @Test
    void hostResidentConstantIsNotCopiedPerCall() {
        assumeCuda();
        INDArray w = hostResidentWeight(ROWS, COLS);
        try (SameDiff sd = SameDiff.create();
             INDArray x = Nd4j.rand(DataType.FLOAT, ROWS, COLS).subi(0.5)) {
            // One consumer of w, placed wherever automatic placement puts it.
            SDVariable xs = sd.placeHolder("x", DataType.FLOAT, ROWS, COLS);
            xs.add("out", sd.constant("w", w));
            DynamicShapePlan plan = sd.compileDynamicShapePlan("out");
            sd.compileNativeDynamicShapePlan("out");

            float[] xh = x.data().asFloat();
            float[] wh = w.data().asFloat();
            float[] expected = new float[xh.length];
            for (int i = 0; i < expected.length; i++) expected[i] = xh[i] + wh[i];
            checkExecutionDoesNotAllocatePerCall(sd, plan, x, w, expected, "single consumer");
        } finally {
            closeIfOpen(w);
        }
    }

    @Test
    void hostResidentConstantConsumedOnAnotherDeviceIsCopiedOnce() {
        assumeCuda();
        assumeTrue(ops().getAvailableDevices() >= 2, "requires two CUDA devices");
        INDArray w = hostResidentWeight(ROWS, COLS);
        try (SameDiff sd = SameDiff.create();
             INDArray x = Nd4j.rand(DataType.FLOAT, ROWS, COLS).subi(0.5)) {
            SDVariable xs = sd.placeHolder("x", DataType.FLOAT, ROWS, COLS);
            SDVariable weight = sd.constant("w", w);
            xs.add(weight).mul(2.0).sub(1.0).add("out", weight);
            DynamicShapePlan plan = sd.compileDynamicShapePlan("out");
            plan.assignDevices(Map.of(0, 1L, 1, 1L));
            int tagged = ops().dbDeviceId(w.data().opaqueBuffer());
            Set<Integer> consumers = consumerDevices(plan, "w");
            log.info("w tagged for device {}; consumers on devices {}", tagged, consumers);
            assumeTrue(consumers.stream().anyMatch(d -> d != tagged),
                    "placement kept every consumer of w on its tagged device " + tagged);
            sd.compileNativeDynamicShapePlan("out");

            float[] xh = x.data().asFloat();
            float[] wh = w.data().asFloat();
            float[] expected = new float[xh.length];
            for (int i = 0; i < expected.length; i++) expected[i] = ((xh[i] + wh[i]) * 2f - 1f) + wh[i];
            checkExecutionDoesNotAllocatePerCall(sd, plan, x, w, expected, "cross-device consumer");
        } finally {
            closeIfOpen(w);
        }
    }

    @Test
    void crossDeviceCopyMayBeServedByBudgetFailover() {
        assumeCuda();
        NativeOps ops = ops();
        assumeTrue(ops.getAvailableDevices() >= 2, "requires two CUDA devices");
        final int cols = 1024;
        final int grownRows = 1024;
        final long grownEdgeBytes = (long) grownRows * cols * 4;
        Environment env = Nd4j.getEnvironment();
        try (SameDiff sd = SameDiff.create()) {
            SDVariable xs = sd.placeHolder("x", DataType.FLOAT, -1, cols);
            xs.mul(2.0).add(1.0).mul(3.0).sub("out", 1.0);
            DynamicShapePlan plan = sd.compileDynamicShapePlan("out");
            plan.assignDevices(Map.of(0, 1L, 1, 1L));
            Set<Integer> devices = new TreeSet<>();
            for (DynamicShapeSlot slot : plan.getSlots()) devices.add(resolve(slot.getTargetDeviceId()));
            log.info("chain placed on devices {}", devices);
            assumeTrue(devices.size() >= 2, "placement did not split the chain across devices");
            sd.compileNativeDynamicShapePlan("out");

            for (int i = 0; i < 4; i++) runChain(sd, 64, cols, "warm rows=64 call " + i);

            // The cached copy of the cross-device edge is now too small, and a plan that has
            // executed cannot move the consumer segment. The budget pushes the replacement onto
            // host-resident failover memory, which the consumer must accept.
            long[] poolBefore = new long[ops.getAvailableDevices()];
            for (int d = 0; d < poolBefore.length; d++) {
                long[] stats = poolStats(ops, d);
                assumeTrue(stats[1] > 0, "CUDA async pool inactive on device " + d
                        + ": the device budget is not enforced");
                poolBefore[d] = stats[0];
            }
            env.setMaxDeviceMemory(1L);
            try {
                for (int i = 0; i < 3; i++) runChain(sd, grownRows, cols, "budgeted rows=" + grownRows + " call " + i);
            } finally {
                env.setMaxDeviceMemory(-1L);
            }
            for (int d = 0; d < poolBefore.length; d++) {
                long growth = poolStats(ops, d)[0] - poolBefore[d];
                assertTrue(growth < grownEdgeBytes / 2, "device " + d + " pool grew by "
                        + growth / MiB + " MiB under an exhausted budget; the grown edge was not "
                        + "served by failover memory, so this case was not exercised");
            }
            // The failover copy stays in use once the budget is lifted.
            for (int i = 0; i < 2; i++) runChain(sd, grownRows, cols, "unbudgeted rows=" + grownRows + " call " + i);
            DspPlanAssertions.assertNoCaptureFailures(sd, "budget failover migration");
        }
    }

    /**
     * A FLOAT array whose device allocation was served by the pool's budget failover, so it is
     * host-resident while tagged for the current device.
     */
    private static INDArray hostResidentWeight(long rows, long cols) {
        Environment env = Nd4j.getEnvironment();
        // The pool applies the budget only once it is initialized.
        Nd4j.create(DataType.FLOAT, 1024).close();
        INDArray w;
        try (INDArray values = Nd4j.rand(DataType.FLOAT, rows, cols).subi(0.5)) {
            int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            long before = env.getDeviceCounter(device);
            env.setMaxDeviceMemory(1L);
            try {
                w = Nd4j.createUninitialized(DataType.FLOAT, rows, cols);
                w.assign(values);
            } finally {
                env.setMaxDeviceMemory(-1L);
            }
            long bytes = rows * cols * 4;
            long charged = env.getDeviceCounter(device) - before;
            assertEquals(device, ops().dbDeviceId(w.data().opaqueBuffer()), "failover memory keeps the current device tag");
            assertTrue(charged < bytes / 4, "precondition: w must be host-resident, but device " + device
                    + " was charged " + charged / MiB + " MiB of its " + bytes / MiB + " MiB");
        }
        return w;
    }

    private static void checkExecutionDoesNotAllocatePerCall(SameDiff sd, DynamicShapePlan plan, INDArray x,
                                                              INDArray w, float[] expected, String context) {
        int tagged = ops().dbDeviceId(w.data().opaqueBuffer());
        long bytes = w.length() * 4;
        int wIndex = Arrays.asList(plan.getExternalInputKeys()).indexOf("w");
        assertTrue(wIndex >= 0, "w must be a plan external input");
        log.info("{}: w tagged for device {}, consumers on devices {}", context, tagged, consumerDevices(plan, "w"));

        run(sd, x, expected, context + " first call");
        INDArray[] externals = sd.getOrCreateSession().getDynamicShapePlanExecutor().getExternalInputsSnapshot();
        assertNotNull(externals, "executor must retain its external inputs");
        assertSame(w, externals[wIndex], "precondition: the plan must read the host-resident caller array");

        for (int i = 1; i < WARMUP_CALLS; i++) run(sd, x, expected, context + " warmup call " + i);
        long before = totalDeviceCounter();
        for (int i = 0; i < MEASURED_CALLS; i++) run(sd, x, expected, context + " measured call " + i);
        long growth = totalDeviceCounter() - before;
        log.info("{}: device counters grew {} MiB over {} calls (w is {} MiB)",
                context, growth / MiB, MEASURED_CALLS, bytes / MiB);
        assertTrue(growth < bytes / 2, context + ": device counters grew " + growth / MiB + " MiB over "
                + MEASURED_CALLS + " calls; w (" + bytes / MiB + " MiB) is being copied per call");
        assertEquals(tagged, ops().dbDeviceId(w.data().opaqueBuffer()),
                context + ": execution must not migrate the caller's buffer");
        DspPlanAssertions.assertNoCaptureFailures(sd, context);
    }

    private static void run(SameDiff sd, INDArray x, float[] expected, String context) {
        Map<String, INDArray> result = sd.output(Map.of("x", x), "out");
        try {
            assertArrayEquals(expected, result.get("out").data().asFloat(), 1e-5f, context);
        } finally {
            result.values().forEach(INDArray::close);
        }
    }

    private static void runChain(SameDiff sd, int rows, int cols, String context) {
        try (INDArray x = Nd4j.rand(DataType.FLOAT, rows, cols).subi(0.5)) {
            float[] xh = x.data().asFloat();
            float[] expected = new float[xh.length];
            for (int i = 0; i < expected.length; i++) expected[i] = (xh[i] * 2f + 1f) * 3f - 1f;
            run(sd, x, expected, context);
        }
    }

    /** Devices of the slots that read the named external input. */
    private static Set<Integer> consumerDevices(DynamicShapePlan plan, String external) {
        int extIdx = Arrays.asList(plan.getExternalInputKeys()).indexOf(external);
        Set<Integer> devices = new TreeSet<>();
        for (DynamicShapeSlot slot : plan.getSlots()) {
            int[] sources = slot.getInputSourceIndices();
            byte[] types = slot.getInputSourceTypes();
            if (sources == null || types == null) continue;
            for (int i = 0; i < sources.length && i < types.length; i++) {
                if (sources[i] < 0 && types[i] != DynamicShapeSlot.SOURCE_OP_OUTPUT
                        && -(sources[i] + 1) == extIdx) {
                    devices.add(resolve(slot.getTargetDeviceId()));
                }
            }
        }
        return devices;
    }

    /** A slot without an assigned device runs on the calling thread's device. */
    private static int resolve(int targetDevice) {
        return targetDevice >= 0 ? targetDevice : Nd4j.getAffinityManager().getDeviceForCurrentThread();
    }

    private static long totalDeviceCounter() {
        long total = 0;
        for (int d = 0; d < ops().getAvailableDevices(); d++) total += Nd4j.getEnvironment().getDeviceCounter(d);
        return total;
    }

    /** {used, reserved} bytes of the device's async pool. */
    private static long[] poolStats(NativeOps ops, int device) {
        try (LongPointer used = new LongPointer(1); LongPointer reserved = new LongPointer(1)) {
            ops.getMemoryPoolStats(device, used, reserved);
            return new long[]{used.get(), reserved.get()};
        }
    }

    private static void closeIfOpen(INDArray array) {
        if (array != null && !array.wasClosed()) array.close();
    }

    private static NativeOps ops() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps();
    }

    private static void assumeCuda() {
        assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend")),
                "CUDA-only host-resident migration test");
    }
}
