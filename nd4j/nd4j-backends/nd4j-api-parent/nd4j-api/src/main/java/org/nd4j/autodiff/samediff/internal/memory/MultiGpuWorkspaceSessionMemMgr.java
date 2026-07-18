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
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * A SessionMemMgr that distributes work across multiple GPUs based on free memory.
 *
 * <p>Each instance selects the GPU with the most free memory via
 * {@link DeviceMemoryManager#selectBestGpu()}. The manager creates
 * a per-thread native workspace on the selected GPU. Cross-device spillover and host
 * fallback are handled by the native CUDA memory pool/workspace implementations (when
 * enabled), not explicitly by this class.</p>
 *
 * <p>Usage:</p>
 * <pre>
 * SameDiff sd = ...;
 * sd.enableMultiGpuWorkspaceMode(256 * 1024 * 1024L); // 256MB per GPU
 * </pre>
 */
@Slf4j
public class MultiGpuWorkspaceSessionMemMgr implements SessionMemMgr {

    private final long workspaceBytesPerGpu;
    private final int numGpus;
    private final int assignedGpu;

    // Delegate to a WorkspaceSessionMemMgr that runs on the assigned GPU
    private final WorkspaceSessionMemMgr delegate;

    /**
     * Create a multi-GPU workspace session memory manager.
     *
     * @param workspaceBytesPerGpu workspace size per GPU in bytes
     * @param numGpus             total number of available GPUs
     */
    public MultiGpuWorkspaceSessionMemMgr(long workspaceBytesPerGpu, int numGpus) {
        this.workspaceBytesPerGpu = workspaceBytesPerGpu;
        this.numGpus = Math.max(numGpus, 1);

        // Select GPU with the most free memory via the single device selection mechanism
        this.assignedGpu = DeviceMemoryManager.getInstance().selectBestGpu();

        // Set device affinity for this thread
        try {
            DeviceMemoryManager.getInstance().switchDevice(assignedGpu, "MultiGpuWorkspaceSessionMemMgr.init", "workspace-init");
        } catch (Exception e) {
            log.warn("Failed to set device affinity to GPU {}: {}", assignedGpu, e.getMessage());
        }

        this.delegate = new WorkspaceSessionMemMgr(workspaceBytesPerGpu);
        log.debug("Thread {} assigned to GPU {} ({} total GPUs, selected by most free memory)",
                Thread.currentThread().getId(), assignedGpu, this.numGpus);
    }

    /**
     * Get the GPU device index assigned to this manager (and its thread).
     */
    public int getAssignedGpu() {
        return assignedGpu;
    }

    @Override
    public void scopeIn() {
        // Ensure we're on the right device
        try {
            DeviceMemoryManager.getInstance().switchDevice(assignedGpu, "MultiGpuWorkspaceSessionMemMgr.scopeIn", "scope-in");
        } catch (Exception e) {
            // ignore - best effort
        }
        delegate.scopeIn();
    }

    @Override
    public void scopeOut() {
        delegate.scopeOut();
    }

    @Override
    public Pointer getNativeWorkspacePointer() {
        return delegate.getNativeWorkspacePointer();
    }

    @Override
    public boolean isWorkspaceBacked() {
        return true;
    }

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        return delegate.allocate(detached, dataType, shape);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        return delegate.allocate(detached, descriptor);
    }

    @Override
    public INDArray ulike(INDArray arr) {
        return delegate.ulike(arr);
    }

    @Override
    public INDArray dup(INDArray arr) {
        return delegate.dup(arr);
    }

    @Override
    public void release(INDArray array) {
        delegate.release(array);
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer) {
        return delegate.allocateFromDescriptor(detached, dataBuffer);
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer, boolean requiresZeroed) {
        return delegate.allocateFromDescriptor(detached, dataBuffer, requiresZeroed);
    }

    @Override
    public void close() {
        delegate.close();
    }
}
