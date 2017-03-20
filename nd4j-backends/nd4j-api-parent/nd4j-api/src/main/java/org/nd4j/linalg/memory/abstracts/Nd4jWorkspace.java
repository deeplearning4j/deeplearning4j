package org.nd4j.linalg.memory.abstracts;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
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
public abstract class Nd4jWorkspace implements MemoryWorkspace {
    protected int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
    protected long threadId;

    @Getter protected String id;

    protected AtomicLong currentSize = new AtomicLong(0);
    protected AtomicLong hostOffset = new AtomicLong(0);
    protected AtomicLong deviceOffset = new AtomicLong(0);

    protected PointersPair workspace = new PointersPair();

    protected MemoryManager memoryManager;

    protected AtomicBoolean isLearning = new AtomicBoolean(true);
    protected AtomicBoolean isUsed = new AtomicBoolean(true);

    protected AtomicLong disabledCounter = new AtomicLong(0);


    protected AtomicLong cyclesCount = new AtomicLong(0);
    protected AtomicLong lastCycleAllocations = new AtomicLong(0);
    protected AtomicLong cycleAllocations = new AtomicLong(0);
    protected AtomicLong spilledAllocations = new AtomicLong(0);
    protected AtomicLong maxCycle = new AtomicLong(0);
    protected AtomicBoolean resetPlanned = new AtomicBoolean(false);
    protected AtomicBoolean isOpen = new AtomicBoolean(false);

    @Getter protected final WorkspaceConfiguration workspaceConfiguration;

    // TODO: it should be something like our PointersPair
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    protected MemoryWorkspace previousWorkspace;

    // this memory manager implementation will be used to allocate real memory for this workspace

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this(configuration, DEFAULT_ID);
    }

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        this.workspaceConfiguration = configuration;
        this.id = workspaceId;

        this.memoryManager = Nd4j.getMemoryManager();

        // and actual workspace allocation
        currentSize.set(workspaceConfiguration.getInitialSize());

        if (workspaceConfiguration.getPolicyLearning() == LearningPolicy.OVER_TIME && workspaceConfiguration.getCyclesBeforeInitialization() < 1)
            log.warn("Workspace initialization OVER_TIME was selected, but number of cycles isn't positive value!");

        init();
    }

    public long getHostOffset() {
        return hostOffset.get();
    }

    public long getCurrentSize() {
        return currentSize.get();
    }

    protected void init() {
        //  we want params validation here


        log.info("Allocating workspace of {} bytes...", currentSize.get());

        if (currentSize.get() > 0) {
            if (workspaceConfiguration.getPolicyAllocation() == AllocationPolicy.OVERALLOCATE && workspaceConfiguration.getOverallocationLimit() > 0)
                currentSize.addAndGet((long) (currentSize.get() * workspaceConfiguration.getOverallocationLimit()));

            if (workspaceConfiguration.getMaxSize() > 0 && currentSize.get() > workspaceConfiguration.getMaxSize())
                currentSize.set(workspaceConfiguration.getMaxSize());

        }
    }

    public PagedPointer alloc(long requiredMemory, DataBuffer.Type type) {
        return alloc(requiredMemory, MemoryKind.HOST, type);
    }

    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type type) {
        /*
            just two options here:
            1) reqMem + hostOffset < totalSize, we just return pointer + offset
            2) go for either realloc or external allocation
         */
        long numElements = requiredMemory / Nd4j.sizeOfDataType(type);

        // shortcut made to skip workspace
        if (!isUsed.get()) {
            if (disabledCounter.incrementAndGet() % 10 == 0)
                log.warn("Worskpace was turned off, and wasn't enabled after {} allocations", disabledCounter.get());

            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, true), numElements);
            return pointer;
        }

        if (hostOffset.get() + requiredMemory <= currentSize.get()) {
            // FIXME: check for alignment here
            long prevOffset = hostOffset.getAndAdd(requiredMemory);

            //log.info("Allocating array of {} bytes, capacity of {} elements, prevOffset:", requiredMemory, numElements);

            return workspace.getHostPointer().withOffset(prevOffset, numElements);
        } else {
            spilledAllocations.addAndGet(requiredMemory);

            if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED) {
                resetPlanned.set(true);
            }

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
    public void initializeWorkspace() {
        if (workspaceConfiguration.getPolicyLearning() != LearningPolicy.NONE) {
            if (workspaceConfiguration.getMaxSize() > 0)
                currentSize.set(Math.min(maxCycle.get(), workspaceConfiguration.getMaxSize()));
            else
                currentSize.set(maxCycle.get());

            init();
        }
    }

    @Override
    public void destroyWorkspace() {
        if (workspace.getHostPointer() != null && workspace.getHostPointer().getOriginalPointer() != null && workspace.getHostPointer().getOriginalPointer() instanceof BytePointer)
            workspace.getHostPointer().getOriginalPointer().deallocate();

        workspace.setHostPointer(null);
        currentSize.set(0);
        hostOffset.set(0);
        deviceOffset.set(0);
        cycleAllocations.set(0);
        maxCycle.set(0);
    }


    @Override
    public void close() {
        Nd4j.getMemoryManager().setCurrentWorkspace(previousWorkspace);
        isOpen.set(false);
        /*
            Basically all we want here, is:
            1) memset primary page(s)
            2) purge external allocations
         */

        if (!isUsed.get()) {
            log.warn("Worskpace was turned off, and wasn't ever turned on back again");
            isUsed.set(true);
        }


        if (cycleAllocations.get() > maxCycle.get())
            maxCycle.set(cycleAllocations.get());

        if (currentSize.get() == 0 && workspaceConfiguration.getPolicyLearning() == LearningPolicy.FIRST_LOOP && maxCycle.get() > 0)
            initializeWorkspace();

        lastCycleAllocations.set(cycleAllocations.get());

        disabledCounter.set(0);

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT) {
            hostOffset.set(0);
            deviceOffset.set(0);
            externalAllocations.clear();
        }
    }

    @Override
    public MemoryWorkspace notifyScopeEntered() {
        previousWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        Nd4j.getMemoryManager().setCurrentWorkspace(this);
        isOpen.set(true);

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT) {
            hostOffset.set(0);
            deviceOffset.set(0);
            externalAllocations.clear();

            if (currentSize.get() > 0)
                resetWorkspace();
        } else if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && (resetPlanned.get() || currentSize.get() == hostOffset.get() ) && currentSize.get() > 0) {
            hostOffset.set(0);
            deviceOffset.set(0);
            resetPlanned.set(false);

            if (currentSize.get() > 0)
                resetWorkspace();


            log.info("Resetting workspace at the end of loop...");
        }

        cycleAllocations.set(0);
        disabledCounter.set(0);


        if (currentSize.get() > 0 && workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT)
            Pointer.memset(workspace.getHostPointer(), 0, currentSize.get());


        if (currentSize.get() == 0 && workspaceConfiguration.getPolicyLearning() == LearningPolicy.OVER_TIME && workspaceConfiguration.getCyclesBeforeInitialization() == cyclesCount.intValue())
            initializeWorkspace();


        cyclesCount.incrementAndGet();
        return this;
    }

    protected abstract void resetWorkspace();

    @Override
    public MemoryWorkspace notifyScopeLeft() {
        close();
        return this;
    }

    @Override
    public void toggleWorkspaceUse(boolean isEnabled) {
        isUsed.set(isEnabled);
    }

    @Override
    public long getLastCycleAllocations() {
        return lastCycleAllocations.get();
    }

    @Override
    public long getMaxCycleAllocations() {
        return maxCycle.get();
    }

    /**
     * This method returns True if scope was opened, and not closed yet.
     *
     * @return
     */
    @Override
    public boolean isScopeActive() {
        return isOpen.get();
    }
}
