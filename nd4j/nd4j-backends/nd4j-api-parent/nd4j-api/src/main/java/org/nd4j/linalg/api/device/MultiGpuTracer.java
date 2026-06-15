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

package org.nd4j.linalg.api.device;

import org.nd4j.linalg.api.buffer.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Comprehensive instrumentation for multi-GPU device decisions, data transfers,
 * stream usage, and buffer lifecycle. Toggleable via system property
 * {@code org.nd4j.multiGpu.debug=true}. All output uses slf4j debug level
 * with the {@code [MultiGpu]} prefix for easy filtering.
 *
 * <p>When disabled (default), all methods are no-ops with zero overhead
 * (static final boolean check is JIT-eliminated).</p>
 */
public class MultiGpuTracer {

    private static final Logger log = LoggerFactory.getLogger(MultiGpuTracer.class);

    /** Master toggle — set via {@code -Dorg.nd4j.multiGpu.debug=true} */
    public static final boolean ENABLED = Boolean.getBoolean("org.nd4j.multiGpu.debug");

    private MultiGpuTracer() {}

    /**
     * Trace a CUDA device switch (DeviceMemoryManager.switchDevice() call).
     *
     * @param caller    class/method that initiated the switch
     * @param fromDevice previous device ID
     * @param toDevice   new device ID
     * @param reason     human-readable reason for the switch
     */
    public static void traceDeviceSwitch(String caller, int fromDevice, int toDevice, String reason) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] DeviceSwitch thread={} from={} to={} reason={} caller={}",
                Thread.currentThread().getName(), fromDevice, toDevice, reason, caller);
    }

    /**
     * Trace a cross-device data transfer (replicateToDevice).
     *
     * @param srcDevice  source device ID
     * @param dstDevice  target device ID
     * @param shape      array shape
     * @param dtype      data type
     * @param bytes      total byte size
     * @param isView     whether source is a view
     * @param isConstant whether source is a model constant
     */
    public static void traceDataTransfer(int srcDevice, int dstDevice, long[] shape,
                                          DataType dtype, long bytes, boolean isView, boolean isConstant) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] DataTransfer thread={} src={} dst={} shape={} dtype={} bytes={} view={} constant={}",
                Thread.currentThread().getName(), srcDevice, dstDevice,
                Arrays.toString(shape), dtype, bytes, isView, isConstant);
    }

    /**
     * Trace a CUDA stream operation (creation, retention, release).
     *
     * @param op          stream operation type (e.g., "create", "retain", "release", "refetch")
     * @param threadName  thread name
     * @param device      device ID
     * @param streamAddr  native stream pointer address
     */
    public static void traceStreamOp(String op, String threadName, int device, long streamAddr) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] StreamOp thread={} op={} device={} streamAddr=0x{}",
                threadName, op, device, Long.toHexString(streamAddr));
    }

    /**
     * Trace a buffer allocation.
     *
     * @param slotIdx slot index (-1 if not DSP-managed)
     * @param device  device ID
     * @param bytes   allocation size in bytes
     * @param method  allocation method (e.g., "pool", "direct", "workspace", "cache-reuse")
     */
    public static void traceBufferAlloc(int slotIdx, int device, long bytes, String method) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] BufferAlloc thread={} slot={} device={} bytes={} method={}",
                Thread.currentThread().getName(), slotIdx, device, bytes, method);
    }

    /**
     * Trace a buffer free/deallocation.
     *
     * @param slotIdx slot index (-1 if not DSP-managed)
     * @param device  device ID
     * @param bytes   freed size in bytes
     * @param method  free method (e.g., "dbFreeBuffersOnStream", "buf.close", "deferred")
     */
    public static void traceBufferFree(int slotIdx, int device, long bytes, String method) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] BufferFree thread={} slot={} device={} bytes={} method={}",
                Thread.currentThread().getName(), slotIdx, device, bytes, method);
    }

    /**
     * Trace an op execution event with device and slot info.
     *
     * @param slotIdx       slot index (-1 if not DSP)
     * @param device        target device ID
     * @param opName        operation name
     * @param inputDevices  device IDs of each input array (may contain -1 for unknown)
     * @param outputDevices device IDs of each output array (may contain -1 for unknown)
     */
    public static void traceOpExec(int slotIdx, int device, String opName,
                                    int[] inputDevices, int[] outputDevices) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] OpExec thread={} slot={} device={} op={} inputs={} outputs={}",
                Thread.currentThread().getName(), slotIdx, device, opName,
                formatDeviceArray(inputDevices), formatDeviceArray(outputDevices));
    }

    /**
     * Verify that the actual CUDA device matches the expected device.
     * Logs a warning if they differ.
     *
     * @param opName         operation name
     * @param expectedDevice expected device ID
     * @param actualDevice   actual device ID (from cudaGetDevice equivalent)
     */
    public static void verifyDeviceContext(String opName, int expectedDevice, int actualDevice) {
        if (!ENABLED) return;
        if (expectedDevice == actualDevice) {
            log.debug("[MultiGpu] DeviceVerify thread={} op={} expected={} actual={} OK",
                    Thread.currentThread().getName(), opName, expectedDevice, actualDevice);
        } else {
            log.warn("[MultiGpu] DeviceVerify thread={} op={} expected={} actual={} MISMATCH",
                    Thread.currentThread().getName(), opName, expectedDevice, actualDevice);
        }
    }

    /**
     * Trace device assignment decisions during DSP compilation.
     *
     * @param deviceId       device being assigned
     * @param slotsForDevice number of slots assigned to this device
     * @param totalSlots     total slot count
     * @param memoryMB       available memory budget in MB for this device
     * @param totalMemoryMB  total memory budget across all devices in MB
     * @param p2p            whether this device has P2P access
     */
    public static void traceDeviceAssignment(int deviceId, int slotsForDevice, int totalSlots,
                                              long memoryMB, long totalMemoryMB, boolean p2p) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] DeviceAssign thread={} device={} slots={}/{} memoryMB={}/{}MB p2p={}",
                Thread.currentThread().getName(), deviceId, slotsForDevice, totalSlots,
                memoryMB, totalMemoryMB, p2p);
    }

    /**
     * Trace parallel execution lifecycle events (worker spawn, completion, error).
     *
     * @param event  event type (e.g., "worker-start", "worker-done", "parallel-begin", "parallel-end")
     * @param detail additional detail string
     */
    public static void traceParallelExec(String event, String detail) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] ParallelExec thread={} event={} {}",
                Thread.currentThread().getName(), event, detail);
    }

    /**
     * Trace cross-device input migration in DSP execution.
     *
     * @param slotIdx      slot index
     * @param inputIdx     input index within the op
     * @param fromDevice   source device
     * @param toDevice     target device
     * @param bytes        data size in bytes
     * @param isView       whether the input is a view
     * @param isConstant   whether the input is a model constant
     * @param cacheHit     whether a cached replica was used
     */
    public static void traceInputMigration(int slotIdx, int inputIdx, int fromDevice, int toDevice,
                                            long bytes, boolean isView, boolean isConstant, boolean cacheHit) {
        if (!ENABLED) return;
        log.debug("[MultiGpu] InputMigration thread={} slot={} input={} from={} to={} bytes={} view={} constant={} cacheHit={}",
                Thread.currentThread().getName(), slotIdx, inputIdx, fromDevice, toDevice,
                bytes, isView, isConstant, cacheHit);
    }

    private static String formatDeviceArray(int[] devices) {
        if (devices == null || devices.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < devices.length; i++) {
            if (i > 0) sb.append(",");
            sb.append("dev").append(devices[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
