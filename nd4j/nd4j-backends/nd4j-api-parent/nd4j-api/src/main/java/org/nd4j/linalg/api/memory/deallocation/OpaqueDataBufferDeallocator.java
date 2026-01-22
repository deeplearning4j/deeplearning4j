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

package org.nd4j.linalg.api.memory.deallocation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

/**
 * Deallocator for OpaqueDataBuffer instances.
 * This class integrates OpaqueDataBuffer with the DeallocatorService,
 * ensuring reliable cleanup of native DataBuffer memory without relying on
 * unreliable Java finalizers.
 *
 * <p>When an OpaqueDataBuffer is created, an instance of this deallocator
 * is registered with the DeallocatorService, which will call deallocate()
 * when the Java object becomes unreachable.</p>
 *
 * @author Adam Gibson
 * @see DeallocatorService
 * @see OpaqueDataBuffer
 */
@Slf4j
public class OpaqueDataBufferDeallocator implements Deallocatable, Deallocator {
    private OpaqueDataBuffer buffer;
    private final long uniqueId;
    private final int targetDevice;
    private volatile boolean deallocated = false;
    private volatile boolean constant = false;

    /**
     * Creates a new deallocator for the given OpaqueDataBuffer.
     *
     * @param buffer The OpaqueDataBuffer to manage
     * @param uniqueId Unique identifier for tracking
     * @param targetDevice The device this buffer is allocated on
     */
    public OpaqueDataBufferDeallocator(OpaqueDataBuffer buffer, long uniqueId, int targetDevice) {
        if (buffer == null) {
            throw new IllegalArgumentException("OpaqueDataBuffer cannot be null");
        }
        this.buffer = buffer;
        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
    }

    @Override
    public void deallocate() {
        // Check constant flag first - constant buffers (like shape info) should never be freed
        if (constant) {
            return;
        }

        if (deallocated) {
            return;
        }

        synchronized (this) {
            if (constant || deallocated) {
                return;
            }

            try {
                if (buffer != null && !buffer.isNull()) {
                    if (log.isTraceEnabled()) {
                        log.trace("Deallocating OpaqueDataBuffer with uniqueId: {}", uniqueId);
                    }

                    int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                    int bufferDevice = targetDevice;
                    try {
                        bufferDevice = buffer.deviceId();
                    } catch (Exception e) {
                        // Fall back to targetDevice if native query fails
                    }

                    int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
                    if (deviceCount <= 0 || bufferDevice < 0 || bufferDevice >= deviceCount) {
                        if (Nd4j.getEnvironment().isDebug()) {
                            log.debug("OpaqueDataBuffer deallocator: invalid buffer deviceId={} (devices={}); using targetDevice={}",
                                    bufferDevice, deviceCount, targetDevice);
                        }
                        bufferDevice = targetDevice;
                    }
                    if (deviceCount > 0 && (bufferDevice < 0 || bufferDevice >= deviceCount)) {
                        if (Nd4j.getEnvironment().isDebug()) {
                            log.debug("OpaqueDataBuffer deallocator: invalid targetDevice={} (devices={}); using currentDevice={}",
                                    targetDevice, deviceCount, currentDevice);
                        }
                        bufferDevice = currentDevice;
                    }
                    boolean switchedDevice = false;

                    if (currentDevice != bufferDevice) {
                        // Switch to the actual buffer device and reset cached context
                        Nd4j.getAffinityManager().unsafeSetDevice(bufferDevice);
                        switchedDevice = true;
                    }

                    try {
                        // Synchronize the device before closing to ensure all async ops are complete
                        Nd4j.getExecutioner().commit();

                        // Call native cleanup
                        // Note: setNull() is called immediately after to prevent any potential
                        // double-free from JavaCPP's NativeDeallocator. The retainReference()
                        // called during creation should prevent automatic deallocation.
                        Nd4j.getNativeOps().dbClose(buffer);
                        buffer.setNull();
                    } finally {
                        // Restore original device context
                        if (switchedDevice) {
                            Nd4j.getAffinityManager().unsafeSetDevice(currentDevice);
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Error deallocating OpaqueDataBuffer with uniqueId: " + uniqueId, e);
            } finally {
                buffer = null;
                deallocated = true;
            }
        }
    }

    @Override
    public long getUniqueId() {
        return uniqueId;
    }

    @Override
    public Deallocator deallocator() {
        return this;
    }

    @Override
    public int targetDevice() {
        return targetDevice;
    }

    @Override
    public boolean isConstant() {
        return constant;
    }

    /**
     * Updates the constant flag for this deallocator.
     *
     * @param constant true if this buffer should never be deallocated
     */
    @Override
    public void setConstant(boolean constant) {
        synchronized (this) {
            // If already deallocated, setting constant won't help - but don't throw,
            // as this could be called during cleanup
            if (deallocated && constant) {
                log.warn("setConstant(true) called on already-deallocated OpaqueDataBuffer - this may indicate a race condition");
            }
            this.constant = constant;
        }
    }

    /**
     * Returns whether this deallocator has already been invoked.
     *
     * @return true if deallocate() has been called
     */
    public boolean isDeallocated() {
        return deallocated;
    }

    /**
     * Returns the OpaqueDataBuffer being managed (may be null if deallocated).
     *
     * @return The managed buffer or null
     */
    public OpaqueDataBuffer getBuffer() {
        return buffer;
    }
}
