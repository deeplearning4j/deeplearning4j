package org.nd4j.jita.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.atomic.AtomicLong;

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

    @Override
    protected void init() {
        super.init();

        if (currentSize.get() > 0) {
            //log.info("Allocating {} bytes at DEVICE & HOST space...", currentSize.get());
            isInit.set(true);

            log.info("Allocating [{}] workspace of {} bytes...", id, currentSize.get());
            Pointer ptr = memoryManager.allocate(currentSize.get() + SAFETY_OFFSET, MemoryKind.HOST, false);
            if (ptr == null)
                throw new ND4JIllegalStateException("Can't allocate memory for workspace");

            workspace.setHostPointer(new PagedPointer(ptr));


            workspace.setDevicePointer(new PagedPointer(memoryManager.allocate(currentSize.get() + SAFETY_OFFSET, MemoryKind.DEVICE, false)));
        }
    }

    @Override
    public PagedPointer alloc(long requiredMemory, DataBuffer.Type type, boolean initialize) {
        return this.alloc(requiredMemory, MemoryKind.DEVICE, type, initialize);
    }

    @Override
    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type type, boolean initialize) {
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

//        log.info("Allocating {} memory from Workspace... reqMem: {} bytes; Type: {}", kind, requiredMemory, type);

        if (kind == MemoryKind.DEVICE) {
            if (deviceOffset.get() + requiredMemory <= currentSize.get()) {

                long prevOffset = deviceOffset.getAndAdd(requiredMemory);

                PagedPointer ptr = workspace.getDevicePointer().withOffset(prevOffset, numElements);

                if (initialize) {
                    //CudaContext context = AtomicAllocator.getInstance().getMemoryHandler().getCudaContext();

                    CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

                    int ret = NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(ptr, 0, requiredMemory, 0, context.getSpecialStream());
                    if (ret == 0)
                        throw new ND4JIllegalStateException("memset failed.");

                    context.syncSpecialStream();
                }

                return ptr;
            } else {
                // spill
                spilledAllocations.addAndGet(requiredMemory);

                if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && currentSize.get() > 0) {
                    hostOffset.set(0);
                    deviceOffset.set(0);
                    return alloc(requiredMemory, kind, type, initialize);
                }

                //log.info("Workspace [{}]: spilled DEVICE array of {} bytes, capacity of {} elements", id, requiredMemory, numElements);

                AllocationShape shape = new AllocationShape(requiredMemory / Nd4j.sizeOfDataType(type), Nd4j.sizeOfDataType(type), type);

                switch (workspaceConfiguration.getPolicySpill()) {
                    case EXTERNAL:
                        cycleAllocations.addAndGet(requiredMemory);
                        //
                        //AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().malloc(shape, null, AllocationStatus.DEVICE).getDevicePointer()
                        PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.DEVICE, initialize), numElements);
                        //pointer.setLeaked(true);

                        externalAllocations.add(new PointersPair(null, pointer));

                        return pointer;
                    case REALLOCATE: {
                        // TODO: basically reallocate (if possible), and call for alloc once again
                        throw new UnsupportedOperationException("Not implemented yet");
                    }
                    case FAIL:
                    default: {
                        throw new ND4JIllegalStateException("Can't allocate memory: Workspace is full");
                    }
                }
            }
        } else if (kind == MemoryKind.HOST) {
            if (hostOffset.get() + requiredMemory <= currentSize.get()) {

                long prevOffset = hostOffset.getAndAdd(requiredMemory);

                PagedPointer ptr = workspace.getHostPointer().withOffset(prevOffset, numElements);

                if (initialize)
                    Pointer.memset(ptr, 0, requiredMemory);

                return ptr;
            } else {
           //     log.info("Spilled HOST array of {} bytes, capacity of {} elements", requiredMemory, numElements);

                AllocationShape shape = new AllocationShape(requiredMemory / Nd4j.sizeOfDataType(type), Nd4j.sizeOfDataType(type), type);

                switch (workspaceConfiguration.getPolicySpill()) {
                    case EXTERNAL:

                        //memoryManager.allocate(requiredMemory, MemoryKind.HOST, true)
                        //AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().malloc(shape, null, AllocationStatus.DEVICE).getDevicePointer()
                        PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);
                        //pointer.setLeaked(true);

                        externalAllocations.add(new PointersPair(pointer, null));

                        return pointer;
                    case REALLOCATE: {
                        // TODO: basically reallocate (if possible), and call for alloc once again
                        throw new UnsupportedOperationException("Not implemented yet");
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
    protected void clearExternalAllocations() {
        for (PointersPair pair : externalAllocations) {
            if (pair.getHostPointer() != null)
                NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(pair.getHostPointer());

            if (pair.getDevicePointer() != null)
                NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(pair.getDevicePointer(), null);
        }
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
