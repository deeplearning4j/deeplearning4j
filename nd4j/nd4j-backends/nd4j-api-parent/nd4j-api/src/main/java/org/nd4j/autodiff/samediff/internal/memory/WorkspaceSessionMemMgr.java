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
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * A SessionMemMgr implementation that backs allocations with a MemoryWorkspace.
 *
 * <p>All intermediate arrays allocated during a SameDiff forward pass are bump-allocated
 * from the workspace, avoiding individual malloc/free calls. Memory is recycled when
 * scopeOut()/scopeIn() is called between inference passes.</p>
 *
 * <p><b>Important:</b> Arrays that need to survive beyond the workspace scope (output arrays
 * returned to the user) must be allocated as "detached" - these are heap-allocated via
 * {@link Nd4j#create} and are not affected by workspace cycling.</p>
 *
 * <p>This implementation also creates a native C++ Workspace that can be attached to
 * OpContext, allowing C++ ops to use bump allocation for their internal temporaries
 * instead of calling cudaMalloc/malloc per op.</p>
 *
 * <p>This is particularly beneficial for autoregressive inference where the same graph
 * is executed hundreds of times with similar-sized intermediates.</p>
 */
@Slf4j
public class WorkspaceSessionMemMgr implements SessionMemMgr {

    private static final String WORKSPACE_ID_PREFIX = "SAMEDIFF_SESSION_WS_";
    private static volatile boolean JVM_SHUTTING_DOWN = false;

    static {
        try {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> JVM_SHUTTING_DOWN = true));
        } catch (Exception e) {
            // ignore - may already be shutting down
        }
    }

    private final WorkspaceConfiguration workspaceConfig;
    private final long initialSize;
    private final String workspaceId;
    private MemoryWorkspace workspace;
    private boolean scopeActive = false;

    // Native C++ workspace pointer for attaching to OpContext.
    // Stored as both Pointer (for passing to native methods) and raw address.
    // The Pointer returned by createNativeWorkspace may carry a JavaCPP deallocator
    // that would double-free the workspace when GC runs. We retain the pointer to
    // prevent premature GC and manage the lifecycle manually via destroyNativeWorkspace.
    private Pointer nativeWorkspacePtr;
    private volatile boolean nativeWorkspaceDestroyed = false;

    /**
     * Create a workspace-backed session memory manager with the specified initial size.
     *
     * @param initialSize Initial workspace size in bytes. If the workspace needs more,
     *                    it will spill to regular allocations and learn for next cycle.
     */
    public WorkspaceSessionMemMgr(long initialSize) {
        this.initialSize = initialSize;
        this.workspaceConfig = WorkspaceConfiguration.builder()
                .initialSize(initialSize)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();
        // Thread-safe workspace ID to prevent cross-thread conflicts
        this.workspaceId = WORKSPACE_ID_PREFIX + Thread.currentThread().getId();
        initNativeWorkspace();
    }

    /**
     * Create with a custom workspace configuration.
     */
    public WorkspaceSessionMemMgr(WorkspaceConfiguration config) {
        this.initialSize = config.getInitialSize();
        this.workspaceConfig = config;
        this.workspaceId = WORKSPACE_ID_PREFIX + Thread.currentThread().getId();
        initNativeWorkspace();
    }

    /**
     * Native workspace size: 8MB of pinned host memory for C++ op temporaries.
     * With ~4400 ops per decoder step and each op allocating ~128 bytes of shape info,
     * one full step needs ~2.2MB. 8MB provides comfortable headroom.
     * Anything beyond this spills to cudaHostAlloc (CUDA) or aligned_alloc (CPU),
     * which are still safe from glibc heap corruption.
     */
    private static final long NATIVE_WORKSPACE_BYTES = 8L * 1024 * 1024;

    /**
     * Initialize the native C++ workspace for OpContext attachment.
     * The native workspace provides bump-allocated memory for C++ op temporaries
     * (shape calculations, index arrays, etc.) via the ALLOCATE macro.
     *
     * Without this workspace, all C++ temporaries are allocated via aligned_alloc/free
     * on the glibc heap. Buffer overruns in C++ ops corrupt adjacent malloc metadata,
     * causing cumulative heap corruption that manifests as SIGABRT during JVM shutdown
     * after ~256 decode steps.
     *
     * The workspace is HOST-ONLY (no GPU memory) to avoid the CudaMemoryPool conflict
     * that previously required disabling it. On CUDA, host memory uses cudaHostAlloc
     * (pinned memory), which is separate from the glibc heap and CudaMemoryPool.
     */
    private void initNativeWorkspace() {
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeWorkspacePtr = nativeOps.createNativeWorkspace(NATIVE_WORKSPACE_BYTES);
            if (nativeWorkspacePtr != null) {
                // Initial scopeIn to prepare for first use
                nativeOps.workspaceScopeIn(nativeWorkspacePtr);
                log.debug("Created native workspace: {}MB host-only", NATIVE_WORKSPACE_BYTES / (1024 * 1024));
            }
        } catch (Exception e) {
            log.warn("Failed to create native workspace, C++ ops will use heap allocation: {}", e.getMessage());
            nativeWorkspacePtr = null;
        }
    }

    /**
     * Enter workspace scope. Must be called before forward pass execution.
     * All subsequent allocations (non-detached) will come from the workspace.
     */
    public void scopeIn() {
        if (scopeActive) {
            return;
        }

        workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, workspaceId);
        scopeActive = true;

        // Reset native workspace: free spills from previous cycle, reinit for new cycle.
        // This ensures C++ op temporaries from the previous forward pass are recycled.
        if (nativeWorkspacePtr != null && !nativeWorkspaceDestroyed) {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.workspaceScopeIn(nativeWorkspacePtr);
            } catch (Exception e) {
                log.debug("Native workspace scopeIn: {}", e.getMessage());
            }
        }
    }

    /**
     * Exit workspace scope. Must be called after forward pass execution.
     * Resets workspace offsets, making memory reusable for the next pass.
     */
    public void scopeOut() {
        if (!scopeActive || workspace == null) {
            return;
        }

        // Reset native workspace bump offset so memory is reusable for next forward pass.
        if (nativeWorkspacePtr != null && !nativeWorkspaceDestroyed) {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.workspaceScopeOut(nativeWorkspacePtr);
            } catch (Exception e) {
                log.debug("Native workspace scopeOut: {}", e.getMessage());
            }
        }

        workspace.close();
        scopeActive = false;
    }

    /**
     * Check if workspace scope is currently active.
     */
    public boolean isScopeActive() {
        return scopeActive;
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
            // Detached arrays must survive beyond workspace scope - heap allocate.
            // Use scopeOutOfWorkspaces to ensure allocation is outside any active workspace,
            // following the established pattern used in ComputationGraph and MultiLayerNetwork.
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                return Nd4j.create(dataType, shape);
            }
        }

        if (scopeActive && workspace != null) {
            // Allocate from workspace
            try (MemoryWorkspace ws = workspace.notifyScopeBorrowed()) {
                return Nd4j.create(dataType, shape);
            }
        }

        // Fallback to heap allocation if workspace is not active
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

    /**
     * Duplicate an array. Always allocates the copy outside of workspace scope
     * to prevent dangling pointers when the workspace resets.
     */
    @Override
    public INDArray dup(INDArray arr) {
        // Always dup outside workspace to prevent dangling references
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            return arr.dup();
        }
    }

    @Override
    public void release(INDArray array) {
        // No-op for workspace-backed arrays - memory is recycled on scopeOut.
        // Detached (heap) arrays will be GC'd normally.
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

        if (detached) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                return Nd4j.create(dataType, shape);
            }
        }

        if (scopeActive && workspace != null) {
            try (MemoryWorkspace ws = workspace.notifyScopeBorrowed()) {
                return Nd4j.create(dataType, shape);
            }
        }

        return Nd4j.create(dataType, shape);
    }

    @Override
    public void close() {
        if (scopeActive) {
            scopeOut();
        }

        if (nativeWorkspacePtr != null && !nativeWorkspaceDestroyed && !JVM_SHUTTING_DOWN) {
            nativeWorkspaceDestroyed = true;
            try {
                Nd4j.getExecutioner().commit();
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                nativeOps.destroyNativeWorkspace(nativeWorkspacePtr);
            } catch (Exception e) {
                log.debug("Native workspace cleanup: {}", e.getMessage());
            }
            nativeWorkspacePtr = null;
        }

        if (workspace != null) {
            try {
                Nd4j.getWorkspaceManager().destroyWorkspace(workspace);
            } catch (Exception e) {
                log.debug("Java workspace cleanup: {}", e.getMessage());
            }
            workspace = null;
        }
    }
}
