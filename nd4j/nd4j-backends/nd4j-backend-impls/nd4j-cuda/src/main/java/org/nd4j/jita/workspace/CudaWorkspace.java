/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.jita.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jCuda;

/**
 * CUDA-aware MemoryWorkspace implementation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaWorkspace extends Nd4jWorkspace {


    public CudaWorkspace(@NonNull WorkspaceConfiguration configuration) {
        super(configuration);
    }

    public CudaWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        super(configuration, workspaceId);
    }

    public CudaWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId, Integer deviceId) {
        super(configuration, workspaceId);
        this.deviceId = deviceId;
    }

    @Override
    protected void init() {
        if (workspaceConfiguration.getPolicyLocation() == LocationPolicy.MMAP) {
            throw new ND4JIllegalStateException("CUDA do not support MMAP workspaces yet");
        }

        super.init();

        if (currentSize.get() > 0) {
            //log.info("Allocating {} bytes at DEVICE & HOST space...", currentSize.get());
            isInit.set(true);

            long bytes = currentSize.get();

            if (isDebug.get())
                log.info("Allocating [{}] workspace on device_{}, {} bytes...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), bytes);

            if (isDebug.get()) {
                Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();
            }

            Pointer ptr = memoryManager.allocate((bytes + SAFETY_OFFSET), MemoryKind.HOST, false);
            if (ptr == null)
                throw new ND4JIllegalStateException("Can't allocate memory for workspace");

            workspace.setHostPointer(new PagedPointer(ptr));

            if (workspaceConfiguration.getPolicyMirroring() != MirroringPolicy.HOST_ONLY)
                workspace.setDevicePointer(new PagedPointer(memoryManager.allocate((bytes + SAFETY_OFFSET), MemoryKind.DEVICE, false)));

            //log.info("Workspace [{}] initialized successfully", id);
        }
    }

    @Override
    public PagedPointer alloc(long requiredMemory, DataType type, boolean initialize) {
        return this.alloc(requiredMemory, MemoryKind.DEVICE, type, initialize);
    }


    @Override
    public synchronized void destroyWorkspace(boolean extended) {
        currentSize.set(0);
        reset();

        if (extended)
            clearExternalAllocations();

        clearPinnedAllocations(extended);

        if (workspace.getHostPointer() != null)
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(workspace.getHostPointer());

        if (workspace.getDevicePointer() != null)
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(workspace.getDevicePointer(), null);

        workspace.setDevicePointer(null);
        workspace.setHostPointer(null);

    }


    @Override
    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataType type, boolean initialize) {
        long numElements = requiredMemory / Nd4j.sizeOfDataType(type);

        if (!isUsed.get()) {
            if (disabledCounter.incrementAndGet() % 10 == 0)
                log.warn("Worskpace was turned off, and wasn't enabled after {} allocations", disabledCounter.get());

            if (kind == MemoryKind.DEVICE) {
                PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.DEVICE, initialize), numElements);
                externalAllocations.add(new PointersPair(null, pointer));
                return pointer;
            } else {
                PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);
                externalAllocations.add(new PointersPair(pointer, null));
                return pointer;
            }


        }


        long div = requiredMemory % 8;
        if (div!= 0)
            requiredMemory += div;

        boolean trimmer = (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && requiredMemory + cycleAllocations.get() > initialBlockSize.get() && initialBlockSize.get() > 0 && kind == MemoryKind.DEVICE) || trimmedMode.get();

        if (trimmer && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE && !trimmedMode.get()) {
            trimmedMode.set(true);
            trimmedStep.set(stepsCount.get());
        }

        if (kind == MemoryKind.DEVICE) {
            if (deviceOffset.get() + requiredMemory <= currentSize.get() && !trimmer && Nd4j.getWorkspaceManager().getDebugMode() != DebugMode.SPILL_EVERYTHING) {
                cycleAllocations.addAndGet(requiredMemory);
                long prevOffset = deviceOffset.getAndAdd(requiredMemory);

                if (workspaceConfiguration.getPolicyMirroring() == MirroringPolicy.HOST_ONLY)
                    return null;

                val ptr = workspace.getDevicePointer().withOffset(prevOffset, numElements);

                if (isDebug.get())
                    log.info("Workspace [{}] device_{}: alloc array of {} bytes, capacity of {} elements; prevOffset: {}; newOffset: {}; size: {}; address: {}", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), requiredMemory, numElements, prevOffset, deviceOffset.get(), currentSize.get(), ptr.address());

                if (initialize) {
                    val context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

                    int ret = NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(ptr, 0, requiredMemory, 0, context.getSpecialStream());
                    if (ret == 0)
                        throw new ND4JIllegalStateException("memset failed device_" + Nd4j.getAffinityManager().getDeviceForCurrentThread());

                    context.syncSpecialStream();
                }

                return ptr;
            } else {

                // spill
                if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && currentSize.get() > 0 && !trimmer && Nd4j.getWorkspaceManager().getDebugMode() != DebugMode.SPILL_EVERYTHING) {
                    //log.info("End of space reached. Current offset: {}; requiredMemory: {}", deviceOffset.get(), requiredMemory);
                    reset();
                    resetPlanned.set(true);
                    return alloc(requiredMemory, kind, type, initialize);
                }

                if (!trimmer)
                    spilledAllocationsSize.addAndGet(requiredMemory);
                else
                    pinnedAllocationsSize.addAndGet(requiredMemory);

                if (isDebug.get()) {
                    log.info("Workspace [{}] device_{}: spilled DEVICE array of {} bytes, capacity of {} elements", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), requiredMemory, numElements);
                }
                //Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();

                AllocationShape shape = new AllocationShape(requiredMemory / Nd4j.sizeOfDataType(type), Nd4j.sizeOfDataType(type), type);

                cycleAllocations.addAndGet(requiredMemory);

                if (workspaceConfiguration.getPolicyMirroring() == MirroringPolicy.HOST_ONLY)
                    return null;

                switch (workspaceConfiguration.getPolicySpill()) {
                    case REALLOCATE:
                    case EXTERNAL:
                        if (!trimmer) {
                            externalCount.incrementAndGet();
                            //
                            //AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().malloc(shape, null, AllocationStatus.DEVICE).getDevicePointer()
                            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.DEVICE, initialize), numElements);
                            //pointer.setLeaked(true);
                            pointer.isLeaked();

                            externalAllocations.add(new PointersPair(null, pointer));

                            return pointer;
                        } else {
                            pinnedCount.incrementAndGet();

                            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.DEVICE, initialize), numElements);
                            //pointer.setLeaked(true);
                            pointer.isLeaked();

                            pinnedAllocations.add(new PointersPair(stepsCount.get(), requiredMemory, null, pointer));

                            return pointer;
                        }
                    case FAIL:
                    default: {
                        throw new ND4JIllegalStateException("Can't allocate memory: Workspace is full");
                    }
                }
            }
        } else if (kind == MemoryKind.HOST) {
            if (hostOffset.get() + requiredMemory <= currentSize.get() && !trimmer && Nd4j.getWorkspaceManager().getDebugMode() != DebugMode.SPILL_EVERYTHING) {

                long prevOffset = hostOffset.getAndAdd(requiredMemory);

                PagedPointer ptr = workspace.getHostPointer().withOffset(prevOffset, numElements);

                // && workspaceConfiguration.getPolicyMirroring() == MirroringPolicy.HOST_ONLY
                if (initialize)
                    Pointer.memset(ptr, 0, requiredMemory);

                return ptr;
            } else {
           //     log.info("Spilled HOST array of {} bytes, capacity of {} elements", requiredMemory, numElements);

                AllocationShape shape = new AllocationShape(requiredMemory / Nd4j.sizeOfDataType(type), Nd4j.sizeOfDataType(type), type);

                switch (workspaceConfiguration.getPolicySpill()) {
                    case REALLOCATE:
                    case EXTERNAL:
                        if (!trimmer) {
                            //memoryManager.allocate(requiredMemory, MemoryKind.HOST, true)
                            //AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().malloc(shape, null, AllocationStatus.DEVICE).getDevicePointer()
                            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);
                            //pointer.setLeaked(true);

                            externalAllocations.add(new PointersPair(pointer, null));

                            return pointer;
                        } else {
                            //AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().malloc(shape, null, AllocationStatus.DEVICE).getDevicePointer()
                            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);
                            //pointer.setLeaked(true);
                            pointer.isLeaked();

                            pinnedAllocations.add(new PointersPair(stepsCount.get(), 0L, pointer, null));

                            return pointer;
                        }
                    case FAIL:
                    default: {
                        throw new ND4JIllegalStateException("Can't allocate memory: Workspace is full");
                    }
                }
            }
        } else throw new ND4JIllegalStateException("Unknown MemoryKind was passed in: " + kind);

        //throw new ND4JIllegalStateException("Shouldn't ever reach this line");
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

            if (stepNumber + 2 < stepCurrent || extended) {
                pinnedAllocations.remove();

                if (pair.getDevicePointer() != null) {
                    NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(pair.getDevicePointer(), null);
                    pinnedCount.decrementAndGet();

                    if (isDebug.get())
                        log.info("deleting external device allocation ");
                }

                if (pair.getHostPointer() != null) {
                    NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(pair.getHostPointer());

                    if (isDebug.get())
                        log.info("deleting external host allocation ");
                }

                pinnedAllocationsSize.addAndGet(pair.getRequiredMemory() * -1);
            } else {
                break;
            }
        }
    }

    @Override
    protected void clearExternalAllocations() {
        if (isDebug.get())
            log.info("Workspace [{}] device_{} threadId {} guid [{}]: clearing external allocations...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), Thread.currentThread().getId(), guid);

        Nd4j.getExecutioner().commit();

        try {
            for (PointersPair pair : externalAllocations) {
                if (pair.getHostPointer() != null) {
                    NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(pair.getHostPointer());

                    if (isDebug.get())
                        log.info("deleting external host allocation... ");
                }

                if (pair.getDevicePointer() != null) {
                    NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(pair.getDevicePointer(), null);

                    if (isDebug.get())
                        log.info("deleting external device allocation... ");
                }
            }
        } catch (Exception e) {
            log.error("RC: Workspace [{}] device_{} threadId {} guid [{}]: clearing external allocations...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), Thread.currentThread().getId(), guid);
            throw new RuntimeException(e);
        }

        spilledAllocationsSize.set(0);
        externalCount.set(0);
        externalAllocations.clear();
    }

    @Override
    protected void resetWorkspace() {
        if (currentSize.get() < 1)
            return;


/*
        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        //log.info("workspace: {}, size: {}", workspace.getDevicePointer().address(), currentSize.get());

        NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(workspace.getDevicePointer(), 0, currentSize.get() + SAFETY_OFFSET, 0, context.getSpecialStream());

        Pointer.memset(workspace.getHostPointer(), 0, currentSize.get() + SAFETY_OFFSET);

        context.getSpecialStream().synchronize();
        */
    }
}
