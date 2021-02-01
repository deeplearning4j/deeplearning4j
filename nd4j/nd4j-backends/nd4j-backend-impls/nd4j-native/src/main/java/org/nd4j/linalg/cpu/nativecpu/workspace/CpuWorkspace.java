/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.cpu.nativecpu.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.LongPointer;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.List;
import java.util.Queue;

/**
 * CPU-only MemoryWorkspace implementation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CpuWorkspace extends Nd4jWorkspace implements Deallocatable {

    protected LongPointer mmap;

    public CpuWorkspace(@NonNull WorkspaceConfiguration configuration) {
        super(configuration);
    }

    public CpuWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        super(configuration, workspaceId);
    }

    public CpuWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId, Integer deviceId) {
        super(configuration, workspaceId);
        this.deviceId = deviceId;
    }


    public String getUniqueId() {
        return "Workspace_" + getId() + "_" + Nd4j.getDeallocatorService().nextValue();
    }

    @Override
    public Deallocator deallocator() {
        /*
        return new Deallocator() {
            @Override
            public void deallocate() {
                log.info("Deallocator invoked!");
            }
        };
        */
         return new CpuWorkspaceDeallocator(this);
    }

    @Override
    public int targetDevice() {
        return 0;
    }

    @Override
    protected void init() {
        super.init();

        if (workspaceConfiguration.getPolicyLocation() == LocationPolicy.RAM) {

            if (currentSize.get() > 0) {
                isInit.set(true);

                if (isDebug.get())
                    log.info("Allocating [{}] workspace of {} bytes...", id, currentSize.get());

                workspace.setHostPointer(new PagedPointer(memoryManager.allocate(currentSize.get() + SAFETY_OFFSET, MemoryKind.HOST, true)));
                AllocationsTracker.getInstance().markAllocated(AllocationKind.WORKSPACE, 0, currentSize.get() + SAFETY_OFFSET);
            }
        } else if (workspaceConfiguration.getPolicyLocation() == LocationPolicy.MMAP) {
            long flen = tempFile.length();
            mmap = NativeOpsHolder.getInstance().getDeviceNativeOps().mmapFile(null, tempFile.getAbsolutePath(), flen);

            if (mmap == null)
                throw new RuntimeException("MMAP failed");

            workspace.setHostPointer(new PagedPointer(mmap.get(0)));
        }
    }

    @Override
    protected void clearPinnedAllocations(boolean extended) {
        if (isDebug.get())
            log.info("Workspace [{}] device_{} threadId {} cycle {}: clearing pinned allocations...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), Thread.currentThread().getId(), cyclesCount.get());

        while (!pinnedAllocations.isEmpty()) {
            PointersPair pair = pinnedAllocations.peek();
            if (pair == null)
                throw new RuntimeException();

            long stepNumber = pair.getAllocationCycle();
            long stepCurrent = stepsCount.get();

            if (isDebug.get())
                log.info("Allocation step: {}; Current step: {}", stepNumber, stepCurrent);

            if (stepNumber + 2 < stepCurrent|| extended) {
                pinnedAllocations.remove();

                NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(pair.getHostPointer());

                pinnedCount.decrementAndGet();
                pinnedAllocationsSize.addAndGet(pair.getRequiredMemory() * -1);
            } else {
                break;
            }
        }
    }

    protected long mappedFileSize() {
        if (workspaceConfiguration.getPolicyLocation() != LocationPolicy.MMAP)
            return 0;

        return tempFile.length();
    }

    @Override
    protected void clearExternalAllocations() {
        if (isDebug.get())
            log.info("Workspace [{}] device_{} threadId {} guid [{}]: clearing external allocations...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), Thread.currentThread().getId(), guid);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        for (PointersPair pair: externalAllocations) {
            if (pair.getHostPointer() != null)
                nativeOps.freeHost(pair.getHostPointer());
        }
        externalAllocations.clear();
        externalCount.set(0);
        spilledAllocationsSize.set(0);
    }

    @Override
    public synchronized void destroyWorkspace(boolean extended) {
        if (isDebug.get())
            log.info("Destroying workspace...");

        val sizez = currentSize.getAndSet(0);
        hostOffset.set(0);
        deviceOffset.set(0);

        if (extended)
            clearExternalAllocations();

        clearPinnedAllocations(extended);

        if (workspaceConfiguration.getPolicyLocation() == LocationPolicy.RAM) {
            if (workspace.getHostPointer() != null) {
                NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(workspace.getHostPointer());

                AllocationsTracker.getInstance().markReleased(AllocationKind.WORKSPACE, 0, sizez);
            }
        } else if (workspaceConfiguration.getPolicyLocation() == LocationPolicy.MMAP) {
            if (workspace.getHostPointer() != null)
                NativeOpsHolder.getInstance().getDeviceNativeOps().munmapFile(null, mmap, tempFile.length());
        }

        workspace.setDevicePointer(null);
        workspace.setHostPointer(null);
    }

    @Override
    protected void resetWorkspace() {
        //Pointer.memset(workspace.getHostPointer(), 0, currentSize.get() + SAFETY_OFFSET);
    }

    protected PointersPair workspace() {
        return workspace;
    }

    protected Queue<PointersPair> pinnedPointers() {
        return pinnedAllocations;
    }

    protected List<PointersPair> externalPointers() {
        return externalAllocations;
    }

    @Override
    public long getPrimaryOffset() {
        return getHostOffset();
    }
}
