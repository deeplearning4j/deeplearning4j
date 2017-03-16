package org.nd4j.linalg.memory.abstracts;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Nd4jWorkspace implements MemoryWorkspace {
    protected int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();


    protected AtomicLong currentSize = new AtomicLong(0);
    protected AtomicLong currentOffset = new AtomicLong(0);

    protected PointersPair workspace = new PointersPair();

    protected MemoryManager memoryManager;

    protected AtomicBoolean isLearning = new AtomicBoolean(true);


    protected AtomicLong cycleAllocations = new AtomicLong(0);
    protected AtomicLong spilledAllocations = new AtomicLong(0);
    protected AtomicLong maxCycle = new AtomicLong(0);

    @Getter protected final WorkspaceConfiguration workspaceConfiguration;

    // TODO: it should be something like our PointersPair
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    // this memory manager implementation will be used to allocate real memory for this workspace


    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this.workspaceConfiguration = configuration;

        this.memoryManager = Nd4j.getMemoryManager();

        // and actual workspace allocation
        currentSize.set(workspaceConfiguration.getInitialSize());

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


        log.info("Allocating workspace of {} bytes...", currentSize.get());

        if (currentSize.get() > 0) {
            workspace.setHostPointer(new PagedPointer(memoryManager.allocate(currentSize.get() + 1024, MemoryKind.HOST, true)));

            Pointer.memset(workspace.getHostPointer(), 0, currentSize.get() + 1024);
        }
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
        long numElements = requiredMemory / Nd4j.sizeOfDataType(type);

        if (currentOffset.get() + requiredMemory <= currentSize.get()) {
            // FIXME: check for alignment here
            long prevOffset = currentOffset.getAndAdd(requiredMemory);

            log.info("Allocating array of {} bytes, capacity of {} elements, prevOffset:", requiredMemory, numElements);

            return workspace.getHostPointer().withOffset(prevOffset, numElements);
        } else {
            spilledAllocations.addAndGet(requiredMemory);

            log.info("Spilled array of {} bytes, capacity of {} elements", requiredMemory, numElements);

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

        if (cycleAllocations.get() > maxCycle.get())
            maxCycle.set(cycleAllocations.get());


        Pointer.memset(workspace.getHostPointer(), 0, currentSize.get());

        currentOffset.set(0);
        externalAllocations.clear();
    }

    @Override
    public void initializeWorkspace() {
        if (workspaceConfiguration.getPolicyLearning() != LearningPolicy.NONE) {
            currentSize.set(Math.min(maxCycle.get(), workspaceConfiguration.getMaxSize()));
            init();
        }
    }

    @Override
    public void destroyWorkspace() {
        if (workspace.getHostPointer() != null && workspace.getHostPointer().getOriginalPointer() != null && workspace.getHostPointer().getOriginalPointer() instanceof BytePointer)
            workspace.getHostPointer().getOriginalPointer().deallocate();

        workspace.setHostPointer(null);
        currentSize.set(0);
        currentOffset.set(0);
    }

    @Override
    public void notifyScopeEntered() {
        cycleAllocations.set(0);
        currentOffset.set(0);
    }

    @Override
    public void notifyScopeLeft() {
        try {
            close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
