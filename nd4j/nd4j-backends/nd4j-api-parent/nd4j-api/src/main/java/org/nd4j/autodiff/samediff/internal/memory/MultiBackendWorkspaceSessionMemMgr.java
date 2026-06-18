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

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.abstracts.NativeMultiBackendWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * A SessionMemMgr backed by {@link NativeMultiBackendWorkspace} for GPU OOM spill-to-CPU failover.
 *
 * <p>Primary allocations go to the GPU device via the native multi-backend workspace.
 * Host pointers are allocated from the CPU workspace and paired with device pointers
 * when constructing {@link INDArray} instances. CUDA memory reuse and spillover are
 * handled by the native {@code CudaMemoryPool} and workspace implementations.</p>
 *
 * <p>Unlike {@link WorkspaceSessionMemMgr}, this manager uses the multi-backend workspace
 * which supports cross-device transfers and coherence tracking.</p>
 */
@Slf4j
public class MultiBackendWorkspaceSessionMemMgr implements SessionMemMgr {

    private final long initialGpuBytes;
    private final int gpuDeviceIndex;
    private NativeMultiBackendWorkspace multiBackendWorkspace;
    private Pointer nativeWorkspacePtr;
    private boolean scopeActive = false;

    /**
     * Create a multi-backend workspace session memory manager.
     *
     * @param initialGpuBytes initial GPU workspace size in bytes
     * @param gpuDeviceIndex  GPU device index (e.g., 0 for first GPU)
     */
    public MultiBackendWorkspaceSessionMemMgr(long initialGpuBytes, int gpuDeviceIndex) {
        this.initialGpuBytes = initialGpuBytes;
        this.gpuDeviceIndex = gpuDeviceIndex;
    }

    @Override
    public void scopeIn() {
        if (scopeActive) {
            return;
        }

        if (multiBackendWorkspace == null) {
            multiBackendWorkspace = new NativeMultiBackendWorkspace(
                    initialGpuBytes,
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA,
                    gpuDeviceIndex);
        }

        multiBackendWorkspace.scopeIn();
        scopeActive = true;

        // Create native workspace for OpContext attachment
        if (nativeWorkspacePtr == null) {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeWorkspacePtr = nativeOps.createNativeWorkspace(initialGpuBytes);
                if (nativeWorkspacePtr != null) {
                    nativeOps.workspaceScopeIn(nativeWorkspacePtr);
                }
            } catch (Exception e) {
                log.warn("Failed to create native workspace: {}", e.getMessage());
                nativeWorkspacePtr = null;
            }
        } else {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.workspaceScopeIn(nativeWorkspacePtr);
            } catch (Exception e) {
                log.warn("Failed to scope-in native workspace: {}", e.getMessage());
            }
        }
    }

    @Override
    public void scopeOut() {
        if (!scopeActive) {
            return;
        }

        if (nativeWorkspacePtr != null) {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.workspaceScopeOut(nativeWorkspacePtr);
            } catch (Exception e) {
                log.warn("Failed to scope-out native workspace: {}", e.getMessage());
            }
        }

        if (multiBackendWorkspace != null) {
            multiBackendWorkspace.scopeOut();
        }

        scopeActive = false;
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
        if (detached) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                return Nd4j.create(dataType, shape);
            }
        }
        if (scopeActive && multiBackendWorkspace != null
                && Nd4j.getExecutioner().type() == org.nd4j.linalg.api.ops.executioner.OpExecutioner.ExecutionerType.CUDA) {
            long length = Shape.lengthOf(shape);
            if (length > 0) {
                long bytes = length * dataType.width();
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    // Allocate host and device pointers from the native multi-backend workspace.
                    Pointer hostPtr = multiBackendWorkspace.allocateBytesOnDevice(
                            bytes, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0);
                    Pointer devicePtr = multiBackendWorkspace.allocateBytesOnDevice(
                            bytes, NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, gpuDeviceIndex);
                    if (hostPtr != null && !hostPtr.isNull() && devicePtr != null && !devicePtr.isNull()) {
                        DataBuffer buffer = Nd4j.createBuffer(hostPtr, devicePtr, length, dataType);
                        return Nd4j.create(buffer, shape);
                    }
                }
            }
        }

        // Fallback to standard allocation if native workspace is unavailable or allocation failed
        return Nd4j.create(dataType, shape);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        return allocate(detached, descriptor.dataType(), descriptor.getShape());
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
        long[] asJava = dataBuffer.asLong();
        if (Shape.isEmpty(asJava)) {
            INDArray ret = Nd4j.createFromDescriptor(dataBuffer);
            if (detached) {
                ret = ret.detach();
            }
            return ret;
        }

        DataType dataType = Shape.dataType(asJava);
        long[] shape = Shape.shape(asJava);
        return allocate(detached, dataType, shape);
    }

    /**
     * Get the underlying multi-backend workspace for direct access to coherence state,
     * allocated sizes, and device-specific operations.
     */
    public NativeMultiBackendWorkspace getMultiBackendWorkspace() {
        return multiBackendWorkspace;
    }

    @Override
    public void close() {
        if (scopeActive) {
            scopeOut();
        }

        if (nativeWorkspacePtr != null) {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.destroyNativeWorkspace(nativeWorkspacePtr);
            } catch (Exception e) {
                log.warn("Failed to destroy native workspace: {}", e.getMessage());
            }
            nativeWorkspacePtr = null;
        }

        if (multiBackendWorkspace != null) {
            multiBackendWorkspace.close();
            multiBackendWorkspace = null;
        }
    }
}
