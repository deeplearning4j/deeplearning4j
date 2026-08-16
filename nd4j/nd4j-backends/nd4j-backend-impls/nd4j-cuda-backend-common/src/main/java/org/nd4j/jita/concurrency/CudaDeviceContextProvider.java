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

package org.nd4j.jita.concurrency;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.device.DeviceContext;
import org.nd4j.linalg.api.device.DeviceContextProvider;
import org.nd4j.linalg.api.device.MultiGpuTracer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * CUDA implementation of DeviceContextProvider. Consolidates the device-switching
 * logic that was previously scattered across CudaAffinityManager.unsafeSetDevice(),
 * CudaZeroHandler.getCudaContext(), and DynamicShapePlanExecutor.getFreshExecStream().
 *
 * This is the SINGLE place that:
 * 1. Updates the Java affinity map
 * 2. Traces device switches via MultiGpuTracer
 * 3. Calls nativeOps.setDevice() (triggering C++ ContextBuffers::release() + lazy reinit)
 * 4. Resets the CudaZeroHandler ThreadLocal context
 * 5. Fetches fresh stream pointers via buildContext()
 *
 * buildContext() is the SINGLE place that does:
 *   defaultLaunchContext() -> lcExecutionStream() -> retainReference()
 * Previously duplicated at 15+ call sites.
 */
public class CudaDeviceContextProvider implements DeviceContextProvider {
    private static final Logger log = LoggerFactory.getLogger(CudaDeviceContextProvider.class);

    private final Map<Long, Integer> affinityMap;

    public CudaDeviceContextProvider(Map<Long, Integer> affinityMap) {
        this.affinityMap = affinityMap;
    }

    @Override
    public DeviceContext getCurrentContext() {
        return buildContext();
    }

    @Override
    public DeviceContext switchDevice(int deviceId, String caller, String reason) {
        long threadId = Thread.currentThread().getId();
        Integer previousDevice = affinityMap.get(threadId);
        int prevDev = (previousDevice != null) ? previousDevice : -1;

        // Publish the intended affinity before native initialization so callbacks observe the
        // target device. If native switching fails, the catch block reconciles this map with the
        // actual CUDA device before the failure escapes.
        affinityMap.put(threadId, deviceId);

        if (prevDev != deviceId) {
            MultiGpuTracer.traceDeviceSwitch(caller, prevDev, deviceId, reason);
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            // Actually set device (triggers C++ ContextBuffers::release() + lazy reinit)
            nativeOps.setDevice(deviceId);

            // Reset cached CudaContext ThreadLocal so it's recreated with fresh streams
            AtomicAllocator.getInstance().getMemoryHandler().resetCachedContext();

            // Return fresh context with valid stream pointers
            return buildContext();
        } catch (RuntimeException | Error failure) {
            reconcileAffinityAfterFailedSwitch(
                    threadId, previousDevice, deviceId, nativeOps, failure);
            throw failure;
        }
    }

    /**
     * Keep Java affinity state aligned with CUDA even when a switch fails part way through.
     * Native {@code setDevice} may fail before changing devices or a later context reset may fail
     * after it changed devices, so restoring the old map value unconditionally is also incorrect.
     */
    private void reconcileAffinityAfterFailedSwitch(
            long threadId,
            Integer previousDevice,
            int requestedDevice,
            NativeOps nativeOps,
            Throwable switchFailure) {
        try {
            int actualDevice = nativeOps.getDevice();
            if (actualDevice >= 0) {
                affinityMap.put(threadId, actualDevice);
            } else if (previousDevice != null) {
                affinityMap.put(threadId, previousDevice);
            } else {
                affinityMap.remove(threadId);
            }
            log.warn(
                    "CUDA device switch to {} failed; reconciled thread {} affinity to native device {}",
                    requestedDevice, threadId, actualDevice);
        } catch (RuntimeException | Error queryFailure) {
            if (previousDevice != null) {
                affinityMap.put(threadId, previousDevice);
            } else {
                affinityMap.remove(threadId);
            }
            switchFailure.addSuppressed(queryFailure);
            log.warn(
                    "CUDA device switch to {} failed and native device could not be queried; "
                            + "restored thread {} affinity to {}",
                    requestedDevice, threadId, previousDevice);
        }
    }

    @Override
    public Pointer getFreshExecutionStream() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
            if (lc != null) {
                Pointer stream = nativeOps.lcExecutionStream(lc);
                if (stream != null && stream.address() != 0) {
                    stream.retainReference();
                    return stream;
                }
            }
        } catch (Exception e) {
            log.debug("Failed to get fresh execution stream: {}", e.getMessage());
        }
        return null;
    }

    @Override
    public int getCurrentDeviceId() {
        long threadId = Thread.currentThread().getId();
        Integer deviceId = affinityMap.get(threadId);
        return (deviceId != null) ? deviceId : 0;
    }

    @Override
    public int getDeviceCount() {
        CudaEnvironment.getInstance().notifyConfigurationApplied();
        return CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().size();
    }

    @Override
    public void syncExecutionStream() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Pointer stream = getFreshExecutionStream();
        if (stream != null) {
            nativeOps.streamSynchronize(stream);
        }
    }

    @Override
    public boolean supportsStreams() {
        return true;
    }

    @Override
    public void ensureHostAccess(INDArray array) {
        if (array == null || array.isEmpty() || array.isS())
            return;
        // Ensure host pointer exists for the data buffer
        ((BaseCudaDataBuffer) array.data()).lazyAllocateHostPointer();
        // Commit pending CUDA operations first so device data is finalized
        Nd4j.getExecutioner().commit();
        // Bidirectional sync: first H→D (so device has any host-only data, e.g.,
        // from FlatBuffer deserialization), then force D→H (so host has the final
        // authoritative data including any device-side writes from ops like assign).
        // Without the H→D step, force D→H on a deserialized array would overwrite
        // correct host data with zeros from the uninitialized device buffer.
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.dbSyncToSpecial(array.data().opaqueBuffer());
        nativeOps.dbForceSyncToPrimary(array.data().opaqueBuffer());
    }

    /**
     * Build a fresh DeviceContext with valid stream pointers from the current
     * thread's C++ ContextBuffers. This is the SINGLE place that fetches streams.
     *
     * C++ ContextBuffers::execStream() is self-healing: if the device ID mismatches,
     * it calls release() + initialize() automatically, so the returned pointer is
     * always valid for the current device.
     */
    private DeviceContext buildContext() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int deviceId = getCurrentDeviceId();
        Pointer execStream = null;
        Pointer copyStream = null;

        try {
            OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
            if (lc != null) {
                Pointer stream = nativeOps.lcExecutionStream(lc);
                if (stream != null && stream.address() != 0) {
                    stream.retainReference();
                    execStream = stream;
                }

                Pointer special = nativeOps.lcCopyStream(lc);
                if (special != null && special.address() != 0) {
                    special.retainReference();
                    copyStream = special;
                }
            }
        } catch (Exception e) {
            log.debug("Failed to build device context for device {}: {}", deviceId, e.getMessage());
        }

        return new DeviceContext(deviceId, execStream, copyStream);
    }
}
