package org.nd4j.linalg.memory.abstracts;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Nd4jWorkspace implements AutoCloseable, MemoryWorkspace {
    protected int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();


    protected AtomicLong currentSize = new AtomicLong(0);
    protected AtomicLong currentOffset = new AtomicLong(0);

    protected PointersPair workspace = new PointersPair();

    protected MemoryManager memoryManager;

    @Getter protected final WorkspaceConfiguration workspaceConfiguration;

    // TODO: it should be something like our PointersPair
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    // this memory manager implementation will be used to allocate real memory for this workspace



    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this.workspaceConfiguration = configuration;

        this.memoryManager = Nd4j.getMemoryManager();

        init();
    }

    public long getCurrentOffset() {
        return currentOffset.get();
    }

    public long getCurrentSize() {
        return currentSize.get();
    }

    protected void init() {
        //  we want params validation here

        // and actual workspace allocation
        currentSize.set(workspaceConfiguration.getInitialSize());

        log.info("Allocating workspace of {} bytes...", currentSize.get());

        if (currentSize.get() > 0)
            workspace.setHostPointer(new PagedPointer(memoryManager.allocate(currentSize.get(), MemoryKind.HOST, true)));
    }

    public PagedPointer alloc(long requiredMemory, DataBuffer.Type type) {
        return this.alloc(requiredMemory, MemoryKind.HOST, type);
    }

    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type type) {
        /*
            just two options here:
            1) reqMem + currentOffset < totalSize, we just return pointer + offset
            2) go for either realloc or external allocation
         */
        if (currentOffset.get() + requiredMemory < currentSize.get()) {
            // FIXME: check for alignment here
            long prevOffset = currentOffset.getAndAdd(requiredMemory);

            log.info("Allocating array of {} bytes, capacity of {} elements", requiredMemory, requiredMemory / Nd4j.sizeOfDataType(type));

            return workspace.getHostPointer().withOffset(prevOffset, requiredMemory / Nd4j.sizeOfDataType(type));
        } else {
            switch (workspaceConfiguration.getPolicySpill()) {
                case EXTERNAL:
                    return new PagedPointer(new FloatPointer(requiredMemory / Nd4j.sizeOfDataType(type)), requiredMemory / Nd4j.sizeOfDataType(type));
                case REALLOCATE: {
                        throw new UnsupportedOperationException("Not implemented yet");
                    }
                case FAIL:
                default: {
                    throw new ND4JIllegalStateException("Can't allocate memory: Workspace is full");
                }
            }
        }
    }

    public void free(Pointer pointer) {
        // no-op for main page(s), purge for external stuff
    }


    @Override
    public void close() throws Exception {
        // TODO: to be implemented
        /*
            Basically all we want here, is:
            1) memset primary page(s)
            2) purge external allocations
         */
    }
}
