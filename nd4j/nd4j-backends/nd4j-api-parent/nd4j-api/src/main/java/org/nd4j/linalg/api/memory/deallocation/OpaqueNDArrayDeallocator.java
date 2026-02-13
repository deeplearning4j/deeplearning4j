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
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueNDArray;

/**
 * Deallocator for OpaqueNDArray instances.
 * This class integrates OpaqueNDArray with the DeallocatorService,
 * ensuring reliable cleanup of native memory without relying on
 * unreliable Java finalizers.
 *
 * <p>When an OpaqueNDArray is created, an instance of this deallocator
 * is registered with the DeallocatorService, which will call deallocate()
 * when the Java object becomes unreachable.</p>
 *
 * @author Adam Gibson
 * @see DeallocatorService
 * @see OpaqueNDArray
 */
@Slf4j
public class OpaqueNDArrayDeallocator implements Deallocatable, Deallocator {
    private OpaqueNDArray array;
    private final long uniqueId;
    private final int targetDevice;
    private volatile boolean deallocated = false;
    private volatile boolean constant = false;

    /**
     * Creates a new deallocator for the given OpaqueNDArray.
     *
     * @param array The OpaqueNDArray to manage
     * @param uniqueId Unique identifier for tracking
     * @param targetDevice The device this array is allocated on
     */
    public OpaqueNDArrayDeallocator(OpaqueNDArray array, long uniqueId, int targetDevice) {
        if (array == null) {
            throw new IllegalArgumentException("OpaqueNDArray cannot be null");
        }
        this.array = array;
        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
    }

    @Override
    public void deallocate() {
        // Check constant flag first - constant arrays should never be freed
        // This mirrors the behavior in OpaqueDataBufferDeallocator
        if (constant) {
            return;
        }

        if (deallocated) {
            return;
        }

        // During JVM shutdown, skip native deallocation to avoid calling free()
        // on potentially corrupted heap metadata. The OS reclaims all process memory on exit.
        if (DeallocatorService.getShutdownInProgress().get()) {
            return;
        }

        synchronized (this) {
            if (constant || deallocated) {
                return;
            }

            try {
                if (array != null && !array.isNull()) {
                    if (log.isTraceEnabled()) {
                        log.trace("Deallocating OpaqueNDArray with uniqueId: {}", uniqueId);
                    }

                    int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                    int arrayDevice = targetDevice;
                    try {
                        int actualDevice = array.deviceId();
                        if (actualDevice >= 0) {
                            arrayDevice = actualDevice;
                        }
                    } catch (Exception e) {
                        // Fall back to targetDevice if native query fails
                    }

                    int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
                    if (deviceCount <= 0 || arrayDevice < 0 || arrayDevice >= deviceCount) {
                        if (Nd4j.getEnvironment().isDebug()) {
                            log.debug("OpaqueNDArray deallocator: invalid array deviceId={} (devices={}); using targetDevice={}",
                                    arrayDevice, deviceCount, targetDevice);
                        }
                        arrayDevice = targetDevice;
                    }
                    if (deviceCount > 0 && (arrayDevice < 0 || arrayDevice >= deviceCount)) {
                        if (Nd4j.getEnvironment().isDebug()) {
                            log.debug("OpaqueNDArray deallocator: invalid targetDevice={} (devices={}); using currentDevice={}",
                                    targetDevice, deviceCount, currentDevice);
                        }
                        arrayDevice = currentDevice;
                    }
                    boolean switchedDevice = false;

                    if (currentDevice != arrayDevice) {
                        // Switch to the actual array device and reset cached context
                        DeviceMemoryManager.getInstance().switchDevice(arrayDevice, "OpaqueNDArrayDeallocator.deallocate", "dealloc");
                        switchedDevice = true;
                    }

                    try {
                        // Synchronize the device before deleting to ensure all async ops are complete
                        Nd4j.getExecutioner().commit();

                        // Call native cleanup
                        // Note: setNull() is called immediately after to prevent any potential
                        // double-free from JavaCPP's NativeDeallocator. The retainReference()
                        // called during creation should prevent automatic deallocation.
                        Nd4j.getNativeOps().deleteNDArray(array);
                        array.setNull();
                    } finally {
                        // Restore original device context
                        if (switchedDevice) {
                            DeviceMemoryManager.getInstance().switchDevice(currentDevice, "OpaqueNDArrayDeallocator.deallocate", "restore-device");
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Error deallocating OpaqueNDArray with uniqueId: " + uniqueId, e);
            } finally {
                array = null;
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

    @Override
    public void setConstant(boolean constant) {
        this.constant = constant;
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
     * Returns the OpaqueNDArray being managed (may be null if deallocated).
     *
     * @return The managed array or null
     */
    public OpaqueNDArray getArray() {
        return array;
    }
}
