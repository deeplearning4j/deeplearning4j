/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * See the NOTICE file distributed with this work for additional
 * * information regarding copyright ownership.
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * * License for the specific language governing permissions and limitations
 * * under the License.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */

package org.nd4j.linalg.vulkan;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceContext;
import org.nd4j.linalg.api.device.DeviceContextProvider;
import org.nd4j.linalg.api.device.MultiGpuTracer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.nd4j.nativeblas.OpaqueLaunchContext;

import java.util.Map;
import java.util.Objects;

/**
 * Vulkan implementation of {@link DeviceContextProvider}.
 */
public class VulkanDeviceContextProvider implements DeviceContextProvider {
    private final Map<Long, Integer> affinityMap;
    private final Nd4jVulkan nativeOps;

    public VulkanDeviceContextProvider(Map<Long, Integer> affinityMap, Nd4jVulkan nativeOps) {
        this.affinityMap = Objects.requireNonNull(affinityMap, "affinityMap");
        this.nativeOps = Objects.requireNonNull(nativeOps, "nativeOps");
    }

    @Override
    public DeviceContext getCurrentContext() {
        return buildContext();
    }

    @Override
    public DeviceContext switchDevice(int deviceId, String caller, String reason) {
        int deviceCount = getDeviceCount();
        if (deviceId < 0 || deviceId >= deviceCount) {
            throw new IllegalArgumentException(
                    "Invalid Vulkan device ID " + deviceId + "; available device count is " + deviceCount);
        }

        int previousDevice = nativeOps.getDevice();
        nativeOps.clearLastError();
        int status = nativeOps.setDevice(deviceId);
        int errorCode = nativeOps.lastErrorCode();
        int currentDevice = nativeOps.getDevice();
        if (status != 1 || errorCode != 0 || currentDevice != deviceId) {
            String errorMessage = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "Vulkan native device switch failed: requested " + deviceId
                            + ", current " + currentDevice + ", status " + status
                            + ", error " + errorCode + ": " + errorMessage);
        }

        affinityMap.put(Thread.currentThread().getId(), currentDevice);
        if (previousDevice != currentDevice) {
            MultiGpuTracer.traceDeviceSwitch(caller, previousDevice, currentDevice, reason);
        }

        return buildContext();
    }

    @Override
    public Pointer getFreshExecutionStream() {
        return buildContext().getExecutionStream();
    }

    @Override
    public int getCurrentDeviceId() {
        int deviceId = nativeOps.getDevice();
        int deviceCount = getDeviceCount();
        if (deviceId < 0 || deviceId >= deviceCount) {
            throw new IllegalStateException(
                    "Vulkan native backend returned invalid current device " + deviceId
                            + "; available device count is " + deviceCount);
        }
        affinityMap.put(Thread.currentThread().getId(), deviceId);
        return deviceId;
    }

    @Override
    public int getDeviceCount() {
        int deviceCount = nativeOps.getAvailableDevices();
        if (deviceCount < 0) {
            throw new IllegalStateException(
                    "Vulkan native backend returned invalid available device count: " + deviceCount);
        }
        return deviceCount;
    }

    @Override
    public void syncExecutionStream() {
        Pointer executionStream = getFreshExecutionStream();
        nativeOps.clearLastError();
        int status = nativeOps.streamSynchronize(executionStream);
        int errorCode = nativeOps.lastErrorCode();
        if (status != 1 || errorCode != 0) {
            String errorMessage = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "Vulkan execution stream synchronization failed with status " + status
                            + " and error " + errorCode + ": " + errorMessage);
        }
    }

    @Override
    public boolean supportsStreams() {
        return true;
    }

    @Override
    public void ensureHostAccess(INDArray array) {
        if (array == null || array.isEmpty() || array.isS()) {
            return;
        }

        DataBuffer dataBuffer = array.data();
        if (!(dataBuffer instanceof VulkanDataBuffer)) {
            throw new IllegalArgumentException(
                    "Vulkan host access requires VulkanDataBuffer, but found "
                            + (dataBuffer == null ? "null" : dataBuffer.getClass().getName()));
        }

        VulkanDataBuffer vulkanBuffer = (VulkanDataBuffer) dataBuffer;
        int previousDevice = getCurrentDeviceId();
        int arrayDevice = vulkanBuffer.targetDevice();
        try {
            if (previousDevice != arrayDevice) {
                switchDevice(
                        arrayDevice, VulkanDeviceContextProvider.class.getName(),
                        "ensure host access for Vulkan array");
            }
            VulkanRuntime.forNativeOps(nativeOps).executioner().commit();
            vulkanBuffer.syncToPrimary();
        } finally {
            if (previousDevice != arrayDevice) {
                switchDevice(
                        previousDevice, VulkanDeviceContextProvider.class.getName(),
                        "restore caller device after Vulkan host access");
            }
        }
    }

    private DeviceContext buildContext() {
        int deviceId = getCurrentDeviceId();
        OpaqueLaunchContext launchContext = nativeOps.defaultLaunchContext();
        if (launchContext == null || launchContext.isNull()) {
            throw new IllegalStateException(
                    "Vulkan native backend returned no launch context for device " + deviceId);
        }

        Pointer executionStream = retainRequiredStream(
                nativeOps.lcExecutionStream(launchContext), "execution", deviceId);
        Pointer copyStream = retainRequiredStream(
                nativeOps.lcCopyStream(launchContext), "copy", deviceId);
        return new DeviceContext(deviceId, executionStream, copyStream);
    }

    private Pointer retainRequiredStream(Pointer stream, String streamType, int deviceId) {
        if (stream == null || stream.isNull() || stream.address() == 0) {
            throw new IllegalStateException(
                    "Vulkan native backend returned no " + streamType + " stream for device " + deviceId);
        }
        stream.retainReference();
        return stream;
    }

}
