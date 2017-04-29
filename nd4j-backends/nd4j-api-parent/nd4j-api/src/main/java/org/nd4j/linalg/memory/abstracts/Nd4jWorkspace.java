package org.nd4j.linalg.memory.abstracts;

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Basic implementation for MemoryWorkspace interface, further extended in corresponding backends
 *
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class Nd4jWorkspace implements MemoryWorkspace {
    @Getter protected int deviceId;
    @Getter protected Long threadId;

    protected static final long SAFETY_OFFSET = 1024;

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
    protected AtomicLong stepsCount = new AtomicLong(0);
    protected int stepsNumber = 0;

    protected AtomicLong lastCycleAllocations = new AtomicLong(0);
    protected AtomicLong cycleAllocations = new AtomicLong(0);
    protected AtomicLong spilledAllocationsSize = new AtomicLong(0);
    protected AtomicLong pinnedAllocationsSize = new AtomicLong(0);
    protected AtomicLong maxCycle = new AtomicLong(0);
    protected AtomicBoolean resetPlanned = new AtomicBoolean(false);
    protected AtomicBoolean isOpen = new AtomicBoolean(false);
    protected AtomicBoolean isInit = new AtomicBoolean(false);
    protected AtomicBoolean isOver = new AtomicBoolean(false);
    protected AtomicBoolean isBorrowed = new AtomicBoolean(false);

    protected AtomicInteger tagScope = new AtomicInteger(0);

    protected AtomicBoolean isDebug = new AtomicBoolean(false);
    protected AtomicInteger externalCount = new AtomicInteger(0);
    protected AtomicInteger pinnedCount = new AtomicInteger(0);

    protected AtomicBoolean trimmedMode = new AtomicBoolean(false);
    protected AtomicLong trimmedStep = new AtomicLong(0);

    @Getter protected final WorkspaceConfiguration workspaceConfiguration;

    // external allocations are purged at the end of loop
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    // pinned allocations are purged with delay, used for circular mode only
    protected Queue<PointersPair> pinnedAllocations = new LinkedTransferQueue<>();

    protected MemoryWorkspace previousWorkspace;
    protected MemoryWorkspace borrowingWorkspace;

    protected AtomicLong initialBlockSize = new AtomicLong(0);

    protected String guid;

    // this memory manager implementation will be used to allocate real memory for this workspace

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this(configuration, DEFAULT_ID);
    }

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        this.workspaceConfiguration = configuration;
        this.id = workspaceId;
        this.threadId = Thread.currentThread().getId();
        this.guid = java.util.UUID.randomUUID().toString();
        this.memoryManager = Nd4j.getMemoryManager();
        this.deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // and actual workspace allocation
        currentSize.set(workspaceConfiguration.getInitialSize());

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED) {
            if (workspaceConfiguration.getOverallocationLimit() < 1.0)
                throw new ND4JIllegalStateException("For cyclic workspace overallocation should be positive integral value.");

            stepsNumber = (int) (workspaceConfiguration.getOverallocationLimit() + 1);
            log.info("Steps: {}", stepsNumber);
        }

        //if (workspaceConfiguration.getPolicyLearning() == LearningPolicy.OVER_TIME && workspaceConfiguration.getCyclesBeforeInitialization() < 1)
            //log.warn("Workspace [{}]: initialization OVER_TIME was selected, but number of cycles isn't positive value!", id);

        init();
    }

    public long getStepNumber() {
        return stepsCount.get();
    }

    public long getSpilledSize() {
        return spilledAllocationsSize.get();
    }

    public long getPinnedSize() {
        return pinnedAllocationsSize.get();
    }

    public long getInitialBlockSize() {
        return initialBlockSize.get();
    }

    /**
     * This method returns parent Workspace, if any. Null if there's none.
     *
     * @return
     */
    @Override
    public MemoryWorkspace getParentWorkspace() {
        return previousWorkspace;
    }


    public long getDeviceOffset() {
        return deviceOffset.get();
    }

    public long getHostOffset() {
        return hostOffset.get();
    }

    public long getCurrentSize() {
        return currentSize.get();
    }

    protected void init() {
        //  we want params validation here

        if (currentSize.get() > 0) {
            if (!isOver.get()) {
                if (workspaceConfiguration.getPolicyAllocation() == AllocationPolicy.OVERALLOCATE && workspaceConfiguration.getOverallocationLimit() > 0) {
                    currentSize.addAndGet((long) (currentSize.get() * workspaceConfiguration.getOverallocationLimit()));
                    isOver.set(true);
                }
            }

            if (workspaceConfiguration.getMaxSize() > 0 && currentSize.get() > workspaceConfiguration.getMaxSize())
                currentSize.set(workspaceConfiguration.getMaxSize());

        }
    }

    public PagedPointer alloc(long requiredMemory, DataBuffer.Type type, boolean initialize) {
        return alloc(requiredMemory, MemoryKind.HOST, type, initialize);
    }

    /**
     * This method enabled debugging mode for this workspace
     *
     * @param reallyEnable
     */
    @Override
    public void enableDebug(boolean reallyEnable) {
        this.isDebug.set(reallyEnable);
    }

    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type type, boolean initialize) {
        /*
            just two options here:
            1) reqMem + hostOffset < totalSize, we just return pointer + offset
            2) go for either realloc or external allocation
         */
        long div = requiredMemory % 8;
        if (div!= 0)
            requiredMemory += div;

        long numElements = requiredMemory / Nd4j.sizeOfDataType(type);

        // shortcut made to skip workspace
        if (!isUsed.get()) {
            if (disabledCounter.incrementAndGet() % 10 == 0)
                log.warn("Worskpace was turned off, and wasn't enabled after {} allocations", disabledCounter.get());

            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);

            externalAllocations.add(new PointersPair(pointer, null));

            return pointer;
        }

        boolean trimmer = (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && requiredMemory + cycleAllocations.get() > initialBlockSize.get() && initialBlockSize.get() > 0) || trimmedMode.get();

        if (trimmer && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE && !trimmedMode.get()) {
            trimmedMode.set(true);
            trimmedStep.set(stepsCount.get());
        }

        if (hostOffset.get() + requiredMemory <= currentSize.get() && !trimmer) {
            // just alignment to 8 bytes

            cycleAllocations.addAndGet(requiredMemory);
            long prevOffset = hostOffset.getAndAdd(requiredMemory);

            if (isDebug.get())
                log.info("Workspace [{}]: Allocating array of {} bytes, capacity of {} elements, prevOffset:", id, requiredMemory, numElements);

            PagedPointer ptr = workspace.getHostPointer().withOffset(prevOffset, numElements);

            if (initialize)
                Pointer.memset(ptr, 0, requiredMemory);

            return ptr;
        } else {
            if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && currentSize.get() > 0 && !trimmer) {
                hostOffset.set(0);
                deviceOffset.set(0);
                resetPlanned.set(true);
                //stepsCount.incrementAndGet();
                return alloc(requiredMemory, kind, type, initialize);
            }

            if (!trimmer)
                spilledAllocationsSize.addAndGet(requiredMemory);
            else
                pinnedAllocationsSize.addAndGet(requiredMemory);

            if (isDebug.get())
                log.info("Workspace [{}]: step: {}, spilled  {} bytes, capacity of {} elements",  id, stepsCount.get(), requiredMemory, numElements);

            switch (workspaceConfiguration.getPolicySpill()) {
                case REALLOCATE:
                case EXTERNAL:
                    cycleAllocations.addAndGet(requiredMemory);
                    if (!trimmer) {
                        externalCount.incrementAndGet();

                        PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);

                        externalAllocations.add(new PointersPair(pointer, null));

                        return pointer;
                    } else {
                        pinnedCount.incrementAndGet();
                        PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);

                        pinnedAllocations.add(new PointersPair(stepsCount.get(), requiredMemory, pointer, null));


                        return pointer;
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
        if ((currentSize.get() < maxCycle.get() || currentSize.get() < cycleAllocations.get()) && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE && (workspaceConfiguration.getMaxSize() == 0 || (maxCycle.get() < workspaceConfiguration.getMaxSize()))) {
            if (workspaceConfiguration.getPolicyReset() != ResetPolicy.ENDOFBUFFER_REACHED)
                destroyWorkspace();
            isInit.set(false);
        }

        if (trimmedMode.get() && trimmedStep.get() + 2 < stepsCount.get()) {
            destroyWorkspace(false);
            isInit.set(false);
            isOver.set(false);
        }

        if (!isInit.get())
            if (workspaceConfiguration.getPolicyLearning() != LearningPolicy.NONE) {
                if (workspaceConfiguration.getMaxSize() > 0)
                    currentSize.set(Math.min(maxCycle.get(), workspaceConfiguration.getMaxSize()));
                else
                    currentSize.set(maxCycle.get());

                initialBlockSize.set(currentSize.get());

                if (!isOver.get()) {
                    if (workspaceConfiguration.getPolicyAllocation() == AllocationPolicy.OVERALLOCATE && workspaceConfiguration.getOverallocationLimit() > 0 && currentSize.get() > 0) {
                        currentSize.set(currentSize.get() + (long) (currentSize.get() * workspaceConfiguration.getOverallocationLimit()));
                        isOver.set(true);
                    }
                }

                if (workspaceConfiguration.getMinSize() > 0 && currentSize.get() < workspaceConfiguration.getMinSize())
                    currentSize.set(workspaceConfiguration.getMinSize());

                if (externalCount.get() > 0 && (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT || resetPlanned.get())) {
                    clearExternalAllocations();
                    spilledAllocationsSize.set(0);
                    externalCount.set(0);
                    resetPlanned.set(false);
                }


                init();
            }
    }

    public int getNumberOfExternalAllocations() {
        return externalCount.get();
    }

    public int getNumberOfPinnedAllocations() {
        return pinnedCount.get();
    }

    @Override
    public void destroyWorkspace() {
        destroyWorkspace(true);
    }


    @Override
    public void destroyWorkspace(boolean extended) {
        if (workspace.getHostPointer() != null && workspace.getHostPointer().getOriginalPointer() != null && workspace.getHostPointer().getOriginalPointer() instanceof BytePointer)
            workspace.getHostPointer().getOriginalPointer().deallocate();

        workspace.setHostPointer(null);
        currentSize.set(0);
        hostOffset.set(0);
        deviceOffset.set(0);

        externalCount.set(0);

        if (extended)
            clearExternalAllocations();

        //cycleAllocations.set(0);
        //maxCycle.set(0);
    }

    /**
     * This method TEMPORARY enters this workspace, without reset applied
     *
     * @return
     */
    @Override
    public MemoryWorkspace notifyScopeBorrowed() {
        if (isBorrowed.get())
            throw new ND4JIllegalStateException("Workspace ["+id+"]: Can't borrow from borrowed workspace");

        borrowingWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        isBorrowed.set(true);

        Nd4j.getMemoryManager().setCurrentWorkspace(this);

        return this;
    }

    @Override
    public void close() {
        if (isBorrowed.get()) {
            isBorrowed.set(false);
            Nd4j.getMemoryManager().setCurrentWorkspace(borrowingWorkspace);
            return;
        }

        if (tagScope.get() > 0) {
            if (tagScope.decrementAndGet() == 0){
                Nd4j.getMemoryManager().setCurrentWorkspace(this);
            }
            return;
        }


//        if (Nd4j.getExecutioner() instanceof GridExecutioner)
//            ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        Nd4j.getMemoryManager().setCurrentWorkspace(previousWorkspace);
        isOpen.set(false);
        //isDebug.set(false);



        cyclesCount.incrementAndGet();
        if (cyclesCount.get() % stepsNumber == 0) {
            stepsCount.incrementAndGet();
        }
        /*
            Basically all we want here, is:
            1) memset primary page(s)
            2) purge external allocations
         */

        if (!isUsed.get()) {
            log.warn("Worskpace was turned off, and wasn't ever turned on back again");
            isUsed.set(true);
        }


        if (cycleAllocations.get() > maxCycle.get()) {
            log.info("Workspace [{}], current cycle: {}; max cycle: {}", id, cycleAllocations.get(), maxCycle.get());
            maxCycle.set(cycleAllocations.get());
        }


        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();


        if (workspaceConfiguration.getPolicyLearning() != LearningPolicy.NONE && maxCycle.get() > 0) {
            //log.info("Delayed workspace {}, device_{} initialization starts...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread());

            if (externalCount.get() > 0 && (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT || resetPlanned.get())) {
                clearExternalAllocations();
                resetPlanned.set(false);
            }

            if ((workspaceConfiguration.getPolicyLearning() == LearningPolicy.OVER_TIME && workspaceConfiguration.getCyclesBeforeInitialization() == cyclesCount.intValue()) || (workspaceConfiguration.getPolicyLearning() == LearningPolicy.FIRST_LOOP && currentSize.get() == 0)) {
                //log.info("Initializing on cycle {}", cyclesCount.get());
                initializeWorkspace();
            } else if (currentSize.get() > 0 && cycleAllocations.get() > 0 && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE && workspaceConfiguration.getPolicyReset() != ResetPolicy.ENDOFBUFFER_REACHED) {
                //log.info("Reinit on cycle {}", cyclesCount.get());
                initializeWorkspace();
            }
        }



        if (pinnedCount.get() > 0)
            clearPinnedAllocations();

        if (trimmedMode.get() && trimmedStep.get() + 2 < stepsCount.get()) {
            initialBlockSize.set(cycleAllocations.get());
            initializeWorkspace();
            trimmedMode.set(false);
            trimmedStep.set(0);

            hostOffset.set(0);
            deviceOffset.set(0);
        }

        lastCycleAllocations.set(cycleAllocations.get());

        disabledCounter.set(0);
        cycleAllocations.set(0);

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT) {
            hostOffset.set(0);
            deviceOffset.set(0);
        }
    }

    protected abstract void clearPinnedAllocations();

    protected abstract void clearExternalAllocations();

    @Override
    public MemoryWorkspace notifyScopeEntered() {
        // we should block stuff since we're going to invalidate spilled allocations
        // TODO: block on spilled allocations probably?

        MemoryWorkspace prev = Nd4j.getMemoryManager().getCurrentWorkspace();

        if (prev == this && isOpen.get()) {
            tagScope.incrementAndGet();
            return this;
        }

        previousWorkspace = prev;

        Nd4j.getMemoryManager().setCurrentWorkspace(this);
        isOpen.set(true);

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT) {
            hostOffset.set(0);
            deviceOffset.set(0);
        }

        if (externalCount.get() > 0 && (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT || resetPlanned.get())) {
            clearExternalAllocations();
            externalCount.set(0);
            resetPlanned.set(false);
        }

        cycleAllocations.set(0);
        disabledCounter.set(0);

        return this;
    }

    public void reset() {
        hostOffset.set(0);
        deviceOffset.set(0);
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
    public long getThisCycleAllocations() {
        return cycleAllocations.get();
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

    @Override
    public MemoryWorkspace tagOutOfScopeUse() {
        tagScope.incrementAndGet();
        return this;
    }

    @Override
    public String toString() {
        return "Nd4jWorkspace{" +
                "id='" + id + '\'' +
                ", currentSize=" + currentSize.get() +
                '}';
    }

    @Data
    public static class GarbageWorkspaceReference extends WeakReference<MemoryWorkspace> {
        private PagedPointer pointerDevice;
        private PagedPointer pointerHost;
        private String id;
        private Long threadId;

        public GarbageWorkspaceReference(MemoryWorkspace referent, ReferenceQueue<? super MemoryWorkspace> queue) {
            super(referent, queue);
            this.pointerDevice = ((Nd4jWorkspace) referent).workspace.getDevicePointer();
            this.pointerHost = ((Nd4jWorkspace) referent).workspace.getHostPointer();
            this.id = referent.getId();
            this.threadId = referent.getThreadId();
        }
    }
}
