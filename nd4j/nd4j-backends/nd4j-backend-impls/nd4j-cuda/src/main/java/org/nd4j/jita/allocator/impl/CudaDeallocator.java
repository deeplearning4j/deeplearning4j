/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.jita.allocator.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.profiler.data.eventlogger.EventLogger;
import org.nd4j.linalg.profiler.data.eventlogger.EventType;
import org.nd4j.linalg.profiler.data.eventlogger.LogEvent;
import org.nd4j.linalg.profiler.data.eventlogger.ObjectAllocationType;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

@Slf4j
public class CudaDeallocator implements Deallocator {

    private OpaqueDataBuffer opaqueDataBuffer;
    private LogEvent logEvent;
    private volatile boolean isConstant;
    private volatile boolean deallocated = false;
    private final int targetDevice;
    private final long allocationBytes;
    private final Object lock = new Object();

    public CudaDeallocator(@NonNull BaseCudaDataBuffer buffer) {
        opaqueDataBuffer = buffer.getOpaqueDataBuffer();
        isConstant = buffer.isConstant();
        targetDevice = buffer.targetDevice();
        long pointBytes = buffer.getAllocationPoint() != null ? buffer.getAllocationPoint().getNumberOfBytes() : 0L;
        long bufferBytes = buffer.length() * (long) buffer.getElementSize();
        allocationBytes = Math.max(pointBytes, bufferBytes);
        if(EventLogger.getInstance().isEnabled()) {
            logEvent = LogEvent.builder()
                    .attached(buffer.isAttached())
                    .isConstant(buffer.isConstant())
                    .eventType(EventType.DEALLOCATION)
                    .objectAllocationType(ObjectAllocationType.DATA_BUFFER)
                    .associatedWorkspace(Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread().getId())
                    .build();

        }

    }

    @Override
    public void deallocate() {
        // During JVM shutdown, skip native deallocation to avoid calling free()
        // on potentially corrupted heap metadata. The OS reclaims all process memory on exit.
        if (DeallocatorService.getShutdownInProgress().get()) {
            return;
        }

        synchronized (lock) {
            if (isConstant || deallocated) {
                return;
            }

            if (opaqueDataBuffer == null || opaqueDataBuffer.isNull()) {
                deallocated = true;
                return;
            }

            if (!opaqueDataBuffer.tryMarkForDeallocation()) {
                deallocated = true;
                return;
            }

            deallocated = true;
        }

        if(logEvent != null) {
            logEvent.setEventTimeMs(System.currentTimeMillis());
            EventLogger.getInstance().log(logEvent);
        }

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean switchedDevice = false;

        if (currentDevice != targetDevice) {
            DeviceMemoryManager.getInstance().switchDevice(targetDevice, "CudaDeallocator.deallocate", "dealloc");
            switchedDevice = true;
        }

        try {
            Nd4j.getExecutioner().commit();
            NativeOpsHolder.getInstance().getDeviceNativeOps().dbClose(opaqueDataBuffer);
            opaqueDataBuffer.setNull();
            if (allocationBytes > 0) {
                DeviceMemoryManager.getInstance().recordDeallocation(DeviceDescriptor.cuda(targetDevice), allocationBytes);
            }
        } finally {
            if (switchedDevice) {
                DeviceMemoryManager.getInstance().switchDevice(currentDevice, "CudaDeallocator.deallocate", "restore-device");
            }
        }
    }

    @Override
    public boolean isConstant() {
        return isConstant;
    }

    /**
     * Updates the constant flag for this deallocator.
     * This is called when DataBuffer.setConstant() is invoked to ensure
     * that buffers marked as constant (like shape info) are never freed.
     *
     * @param constant true if this buffer should never be deallocated
     */
    @Override
    public void setConstant(boolean constant) {
        synchronized (lock) {
            // If already deallocated, setting constant won't help - but don't throw,
            // as this could be called during cleanup
            if (deallocated && constant) {
                log.warn("setConstant(true) called on already-deallocated buffer - this may indicate a race condition");
            }
            this.isConstant = constant;
        }
        // Also update the log event if present (outside lock - not critical)
        if (logEvent != null) {
            logEvent.setConstant(constant);
        }
    }
}
