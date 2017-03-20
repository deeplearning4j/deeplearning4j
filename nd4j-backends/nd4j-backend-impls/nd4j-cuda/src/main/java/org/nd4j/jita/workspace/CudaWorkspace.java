package org.nd4j.jita.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.atomic.AtomicLong;

/**
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
            log.info("Allocating {} bytes at DEVICE & HOST space...", currentSize.get());
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

            workspace.setHostPointer(new PagedPointer(nativeOps.mallocHost(currentSize.get(), 0)));
            workspace.setDevicePointer(new PagedPointer(nativeOps.mallocDevice(currentSize.get(), null, 0)));
        }
    }

    @Override
    public PagedPointer alloc(long requiredMemory, DataBuffer.Type type) {
        return this.alloc(requiredMemory, MemoryKind.DEVICE, type);
    }

    @Override
    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type type) {
        long numElements = requiredMemory / Nd4j.sizeOfDataType(type);


        log.info("Allocating {} memory from Workspace...", kind);

        if (kind == MemoryKind.DEVICE) {
            if (deviceOffset.get() + requiredMemory <= currentSize.get()) {
                long prevOffset = deviceOffset.getAndAdd(requiredMemory);

                // FIXME: handle alignment here

                return workspace.getDevicePointer().withOffset(prevOffset, numElements);
            } else {
                // spill
                spilledAllocations.addAndGet(requiredMemory);

                if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED) {
                    resetPlanned.set(true);
                }

                log.info("Spilled DEVICE array of {} bytes, capacity of {} elements", requiredMemory, numElements);

                switch (workspaceConfiguration.getPolicySpill()) {
                    case EXTERNAL:
                        cycleAllocations.addAndGet(requiredMemory);
                        PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.DEVICE, true), numElements);

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
        } else if (kind == MemoryKind.HOST) {
            if (hostOffset.get() + requiredMemory <= currentSize.get()) {
                long prevOffset = hostOffset.getAndAdd(requiredMemory);

                // FIXME: handle alignment here

                return workspace.getHostPointer().withOffset(prevOffset, numElements);
            } else {
                log.info("Spilled HOST array of {} bytes, capacity of {} elements", requiredMemory, numElements);

                switch (workspaceConfiguration.getPolicySpill()) {
                    case EXTERNAL:
                        cycleAllocations.addAndGet(requiredMemory);
                        PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, true), numElements);

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
}
