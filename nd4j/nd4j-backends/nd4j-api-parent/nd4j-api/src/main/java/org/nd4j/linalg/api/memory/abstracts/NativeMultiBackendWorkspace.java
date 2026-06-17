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

package org.nd4j.linalg.api.memory.abstracts;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

/**
 * Java wrapper for the C++ MultiBackendWorkspace.
 *
 * <p>Provides multi-device workspace management with coherence tracking
 * and cross-device transfers. Delegates to C++ via NativeOps.</p>
 *
 * <p>This class implements {@link AutoCloseable} for proper cleanup.
 * Use try-with-resources or explicit {@link #close()} to free native memory.</p>
 *
 * <h3>Device Types:</h3>
 * <ul>
 *   <li>0 = CPU</li>
 *   <li>1 = CUDA GPU</li>
 *   <li>2 = ROCm GPU</li>
 *   <li>3 = TPU</li>
 * </ul>
 *
 * <h3>Coherence States:</h3>
 * <ul>
 *   <li>0 = INVALID (stale/not present)</li>
 *   <li>1 = SHARED (valid, shared with other devices)</li>
 *   <li>2 = EXCLUSIVE (valid, only copy)</li>
 *   <li>3 = MODIFIED (modified, only valid copy)</li>
 * </ul>
 */
@Slf4j
public class NativeMultiBackendWorkspace implements AutoCloseable {

    public static final int DEVICE_TYPE_CPU = 0;
    public static final int DEVICE_TYPE_CUDA = 1;

    public static final int COHERENCE_INVALID = 0;
    public static final int COHERENCE_SHARED = 1;
    public static final int COHERENCE_EXCLUSIVE = 2;
    public static final int COHERENCE_MODIFIED = 3;

    @Getter
    private Pointer nativeHandle;
    private final NativeOps nativeOps;
    private volatile boolean closed = false;

    /**
     * Create a multi-backend workspace.
     *
     * @param initialSize        initial allocation size in bytes
     * @param primaryDeviceType  device type (0=CPU, 1=CUDA)
     * @param primaryDeviceIndex device index (e.g., GPU 0, GPU 1)
     */
    public NativeMultiBackendWorkspace(long initialSize, int primaryDeviceType, int primaryDeviceIndex) {
        this.nativeOps = Nd4j.getNativeOps();
        this.nativeHandle = nativeOps.createNativeMultiBackendWorkspace(initialSize, primaryDeviceType, primaryDeviceIndex);
        if (nativeHandle == null || nativeHandle.isNull()) {
            throw new IllegalStateException("Failed to create native multi-backend workspace");
        }
    }

    /**
     * Allocate bytes from the primary device workspace.
     */
    public Pointer allocateBytes(long numBytes) {
        checkNotClosed();
        return nativeOps.nativeMbwAllocateBytes(nativeHandle, numBytes);
    }

    /**
     * Enter workspace scope. Resets offsets for fresh allocation cycle.
     */
    public void scopeIn() {
        checkNotClosed();
        nativeOps.nativeMbwScopeIn(nativeHandle);
    }

    /**
     * Exit workspace scope. Resets offsets, making memory reusable.
     */
    public void scopeOut() {
        checkNotClosed();
        nativeOps.nativeMbwScopeOut(nativeHandle);
    }

    /**
     * Transfer data from source device to target device.
     */
    public void transferTo(int srcDeviceType, int srcDeviceIndex,
                           int dstDeviceType, int dstDeviceIndex) {
        checkNotClosed();
        nativeOps.nativeMbwTransferTo(nativeHandle, srcDeviceType, srcDeviceIndex,
                dstDeviceType, dstDeviceIndex);
    }

    /**
     * Get coherence state for a device.
     *
     * @return coherence state (0=INVALID, 1=SHARED, 2=EXCLUSIVE, 3=MODIFIED)
     */
    public int getCoherenceState(int deviceType, int deviceIndex) {
        checkNotClosed();
        return nativeOps.nativeMbwGetCoherenceState(nativeHandle, deviceType, deviceIndex);
    }

    /**
     * Get total allocated size across all devices.
     */
    public long getTotalAllocatedSize() {
        checkNotClosed();
        return nativeOps.nativeMbwGetTotalAllocatedSize(nativeHandle);
    }

    /**
     * Allocate bytes on a specific device.
     *
     * @param numBytes   bytes to allocate
     * @param deviceType device type (0=CPU, 1=CUDA)
     * @param deviceIndex device index
     * @return pointer to allocated memory
     */
    public Pointer allocateBytesOnDevice(long numBytes, int deviceType, int deviceIndex) {
        checkNotClosed();
        return nativeOps.nativeMbwAllocateBytesOnDevice(nativeHandle, numBytes, deviceType, deviceIndex);
    }

    /**
     * Synchronize a specific device (e.g., cudaDeviceSynchronize for CUDA).
     */
    public void syncDevice(int deviceType, int deviceIndex) {
        checkNotClosed();
        nativeOps.nativeMbwSyncDevice(nativeHandle, deviceType, deviceIndex);
    }

    /**
     * Synchronize all devices.
     */
    public void syncAllDevices() {
        checkNotClosed();
        nativeOps.nativeMbwSyncAllDevices(nativeHandle);
    }

    /**
     * Get allocated size on a specific device.
     */
    public long getAllocatedSizeOnDevice(int deviceType, int deviceIndex) {
        checkNotClosed();
        return nativeOps.nativeMbwGetAllocatedSizeOnDevice(nativeHandle, deviceType, deviceIndex);
    }

    /**
     * Get current offset on primary device.
     */
    public long getCurrentOffset() {
        checkNotClosed();
        return nativeOps.nativeMbwGetCurrentOffset(nativeHandle);
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("NativeMultiBackendWorkspace has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed && nativeHandle != null && !nativeHandle.isNull()) {
            closed = true;
            nativeOps.destroyNativeMultiBackendWorkspace(nativeHandle);
            nativeHandle = null;
        }
    }
}
