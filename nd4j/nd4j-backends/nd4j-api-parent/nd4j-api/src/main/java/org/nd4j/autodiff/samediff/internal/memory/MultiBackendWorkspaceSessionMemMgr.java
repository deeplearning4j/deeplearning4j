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

package org.nd4j.autodiff.samediff.internal.memory;

import lombok.Getter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.abstracts.NativeMultiBackendWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeOps;

import java.util.Arrays;

/**
 * A SessionMemMgr backed by {@link NativeMultiBackendWorkspace} for accelerator workspaces.
 *
 * <p>Primary allocations go to the exact device backend selected at construction
 * (the active affinity manager supplies the default). Host pointers are allocated from
 * the CPU workspace and paired with device pointers when constructing {@link INDArray}
 * instances. The selected native backend owns memory reuse,
 * transfers, and coherence.</p>
 *
 * <p>Unlike {@link WorkspaceSessionMemMgr}, this manager uses the multi-backend workspace
 * which supports cross-device transfers and coherence tracking.</p>
 */
public class MultiBackendWorkspaceSessionMemMgr implements SessionMemMgr {

    private final long initialDeviceBytes;
    private final int deviceIndex;
    private final int nativeDeviceType;
    private final NativeOps nativeOps;
    @Getter
    private NativeMultiBackendWorkspace multiBackendWorkspace;
    private Pointer nativeWorkspacePtr;
    private boolean scopeActive = false;

    /**
     * Create a multi-backend workspace session memory manager.
     *
     * @param initialDeviceBytes initial accelerator workspace size in bytes
     * @param deviceIndex       backend device index
     */
    public MultiBackendWorkspaceSessionMemMgr(long initialDeviceBytes, int deviceIndex) {
        this(initialDeviceBytes,
                Nd4j.getAffinityManager().getDeviceType(deviceIndex),
                deviceIndex);
    }

    /**
     * Create a session manager for an explicit accelerator backend.
     *
     * <p>The exact backend is resolved once and retained for the complete lifetime
     * of every native handle owned by this manager.</p>
     *
     * @param initialDeviceBytes initial accelerator workspace size in bytes
     * @param deviceType        exact accelerator backend
     * @param deviceIndex       device index within that backend
     */
    public MultiBackendWorkspaceSessionMemMgr(long initialDeviceBytes,
                                               DeviceType deviceType,
                                               int deviceIndex) {
        this.initialDeviceBytes = initialDeviceBytes;
        this.deviceIndex = deviceIndex;
        this.nativeDeviceType = nativeDeviceType(deviceType);
        this.nativeOps = MultiBackendNativeOpsHolder.getInstance()
                .getOpsForDeviceType(nativeOpsDeviceType(deviceType));
    }

    private static DeviceType nativeOpsDeviceType(DeviceType deviceType) {
        if (deviceType == DeviceType.CUDA_GPU || deviceType == DeviceType.GPU) {
            return DeviceType.CUDA_GPU;
        }
        if (deviceType == DeviceType.VULKAN_GPU) {
            return DeviceType.VULKAN_GPU;
        }
        throw new IllegalStateException(
                "Native multi-backend workspace has no implementation for device type " + deviceType);
    }

    private static int nativeDeviceType(DeviceType deviceType) {
        DeviceType nativeOpsDeviceType = nativeOpsDeviceType(deviceType);
        if (nativeOpsDeviceType == DeviceType.CUDA_GPU) {
            return NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA;
        }
        return NativeMultiBackendWorkspace.DEVICE_TYPE_VULKAN;
    }

    @Override
    public void scopeIn() {
        if (scopeActive) {
            return;
        }

        if (multiBackendWorkspace == null) {
            multiBackendWorkspace = new NativeMultiBackendWorkspace(
                    initialDeviceBytes,
                    nativeDeviceType,
                    deviceIndex,
                    nativeOps);
        }

        // Create the exact backend's native workspace for OpContext attachment.
        if (nativeWorkspacePtr == null) {
            nativeWorkspacePtr = nativeOps.createNativeWorkspace(initialDeviceBytes);
            if (nativeWorkspacePtr == null || nativeWorkspacePtr.isNull()) {
                throw new IllegalStateException("The active backend did not create its native workspace");
            }
        }

        multiBackendWorkspace.scopeIn();
        try {
            nativeOps.workspaceScopeIn(nativeWorkspacePtr);
            scopeActive = true;
        } catch (RuntimeException | Error failure) {
            try {
                multiBackendWorkspace.scopeOut();
            } catch (RuntimeException | Error rollbackFailure) {
                failure.addSuppressed(rollbackFailure);
            }
            throw failure;
        }
    }

    @Override
    public void scopeOut() {
        if (!scopeActive) {
            return;
        }

        try {
            if (nativeWorkspacePtr != null) {
                nativeOps.workspaceScopeOut(nativeWorkspacePtr);
            }
        } finally {
            try {
                if (multiBackendWorkspace != null) {
                    multiBackendWorkspace.scopeOut();
                }
            } finally {
                scopeActive = false;
            }
        }
    }

    @Override
    public Pointer getNativeWorkspacePointer() {
        return nativeWorkspacePtr;
    }

    @Override
    public boolean isWorkspaceBacked() {
        return true;
    }

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        if (detached || Shape.lengthOf(shape) == 0) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                return Nd4j.create(dataType, shape);
            }
        }
        return Nd4j.create(allocateNativeBuffer(Shape.lengthOf(shape), dataType), shape);
    }

    private DataBuffer allocateNativeBuffer(long length, DataType dataType) {
        if (length <= 0) {
            throw new IllegalArgumentException("Workspace allocations must contain at least one element");
        }
        if (!scopeActive || multiBackendWorkspace == null) {
            throw new IllegalStateException("Native multi-backend workspace scope is not active");
        }

        long bytes = Math.multiplyExact(length, dataType.width());
        try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            Pointer hostPtr = multiBackendWorkspace.allocateBytesOnDevice(
                    bytes, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0);
            Pointer devicePtr = multiBackendWorkspace.allocateBytesOnDevice(
                    bytes, nativeDeviceType, deviceIndex);
            if (hostPtr == null || hostPtr.isNull() || devicePtr == null || devicePtr.isNull()) {
                throw new IllegalStateException("The active backend returned a null workspace allocation");
            }
            return Nd4j.createBuffer(hostPtr, devicePtr, length, dataType);
        }
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        return allocate(detached, descriptor, false);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor, boolean requiresZeroed) {
        INDArray ret;
        if (detached || descriptor.isEmpty()) {
            try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                ret = Nd4j.create(descriptor, false);
            }
        } else {
            DataBuffer buffer = allocateNativeBuffer(descriptor.length(), descriptor.dataType());
            ret = Nd4j.create(buffer, descriptor);
        }

        if (requiresZeroed && !ret.isEmpty()) {
            ret.assign(0);
        }
        return ret;
    }

    @Override
    public INDArray ulike(INDArray arr) {
        return allocate(false, arr.dataType(), arr.shape());
    }

    @Override
    public INDArray dup(INDArray arr) {
        // Always dup outside workspace to prevent dangling references
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return arr.dup();
        }
    }

    @Override
    public void release(INDArray array) {
        // No-op - memory is recycled on scopeOut
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer) {
        return allocateFromDescriptor(detached, dataBuffer, false);
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer, boolean requiresZeroed) {
        long[] asJava = dataBuffer.asLong();
        DataType dataType = Shape.dataType(asJava);
        long[] shape = Shape.shape(asJava);
        boolean canonicalC = !Shape.isEmpty(asJava)
                && Shape.order(asJava) == 'c'
                && Shape.elementWiseStride(asJava) == 1
                && Arrays.equals(Shape.stride(asJava), Nd4j.getStrides(shape, 'c'));

        if (!canonicalC) {
            INDArray ret;
            if (detached || Shape.isEmpty(asJava)) {
                try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    ret = Nd4j.createFromDescriptor(dataBuffer);
                }
            } else {
                DataBuffer buffer = allocateNativeBuffer(Shape.lengthOf(shape), dataType);
                ret = Nd4j.createFromDescriptor(buffer, dataBuffer);
            }
            if (requiresZeroed && !ret.isEmpty()) {
                ret.assign(0);
            }
            return ret;
        }

        INDArray ret = allocate(detached, dataType, shape);
        if (requiresZeroed && !ret.isEmpty()) {
            ret.assign(0);
        }
        return ret;
    }

    @Override
    public void close() {
        if (scopeActive) {
            scopeOut();
        }

        try {
            if (nativeWorkspacePtr != null) {
                try {
                    nativeOps.destroyNativeWorkspace(nativeWorkspacePtr);
                } finally {
                    nativeWorkspacePtr = null;
                }
            }
        } finally {
            if (multiBackendWorkspace != null) {
                try {
                    multiBackendWorkspace.close();
                } finally {
                    multiBackendWorkspace = null;
                }
            }
        }
    }
}
