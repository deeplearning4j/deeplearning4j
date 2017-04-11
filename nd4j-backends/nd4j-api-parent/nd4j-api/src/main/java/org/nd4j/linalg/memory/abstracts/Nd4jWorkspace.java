package org.nd4j.linalg.memory.abstracts;

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
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
    protected int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
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
    protected AtomicLong lastCycleAllocations = new AtomicLong(0);
    protected AtomicLong cycleAllocations = new AtomicLong(0);
    protected AtomicLong spilledAllocations = new AtomicLong(0);
    protected AtomicLong maxCycle = new AtomicLong(0);
    protected AtomicBoolean resetPlanned = new AtomicBoolean(false);
    protected AtomicBoolean isOpen = new AtomicBoolean(false);
    protected AtomicBoolean isInit = new AtomicBoolean(false);
    protected AtomicBoolean isOver = new AtomicBoolean(false);
    protected AtomicBoolean isBorrowed = new AtomicBoolean(false);

    protected AtomicInteger tagScope = new AtomicInteger(0);

    @Getter protected final WorkspaceConfiguration workspaceConfiguration;

    // TODO: it should be something like our PointersPair
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    protected MemoryWorkspace previousWorkspace;
    protected MemoryWorkspace borrowingWorkspace;

    // this memory manager implementation will be used to allocate real memory for this workspace

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this(configuration, DEFAULT_ID);
    }

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        this.workspaceConfiguration = configuration;
        this.id = workspaceId;
        this.threadId = Thread.currentThread().getId();

        this.memoryManager = Nd4j.getMemoryManager();

        // and actual workspace allocation
        currentSize.set(workspaceConfiguration.getInitialSize());

        if (workspaceConfiguration.getPolicyLearning() == LearningPolicy.OVER_TIME && workspaceConfiguration.getCyclesBeforeInitialization() < 1)
            log.warn("Workspace [{}]: initialization OVER_TIME was selected, but number of cycles isn't positive value!", id);

        init();
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

//            log.info("Workspace [{}] spilled  {} bytes, capacity of {} elements",  id, requiredMemory, numElements);

            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);

            externalAllocations.add(new PointersPair(pointer, null));

            return pointer;
        }

        if (hostOffset.get() + requiredMemory <= currentSize.get()) {
            // just alignment to 8 bytes

            long prevOffset = hostOffset.getAndAdd(requiredMemory);

//            log.info("Workspace [{}]: Allocating array of {} bytes, capacity of {} elements, prevOffset:", id, requiredMemory, numElements);

            PagedPointer ptr = workspace.getHostPointer().withOffset(prevOffset, numElements);

            if (initialize)
                Pointer.memset(ptr, 0, requiredMemory);

            return ptr;
        } else {
            if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && currentSize.get() > 0) {
                hostOffset.set(0);
                deviceOffset.set(0);
                return alloc(requiredMemory, kind, type, initialize);
            }

            spilledAllocations.addAndGet(requiredMemory);

            //log.info("Workspace [{}]: spilled  {} bytes, capacity of {} elements",  id, requiredMemory, numElements);

            switch (workspaceConfiguration.getPolicySpill()) {
                case EXTERNAL:
                    cycleAllocations.addAndGet(requiredMemory);
                    PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize), numElements);

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

            if (!isOver.get()) {
                if (workspaceConfiguration.getPolicyAllocation() == AllocationPolicy.OVERALLOCATE && workspaceConfiguration.getOverallocationLimit() > 0 && currentSize.get() > 0) {
                    currentSize.set(currentSize.get() + (long) (currentSize.get() * workspaceConfiguration.getOverallocationLimit()));
                    isOver.set(true);
                }
            }

            if (workspaceConfiguration.getMinSize() > 0 && currentSize.get() < workspaceConfiguration.getMinSize())
                currentSize.set(workspaceConfiguration.getMinSize());

            if (!isInit.get())
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
        }
    }

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

        if (externalAllocations.size() > 0) {
            if (Nd4j.getExecutioner() instanceof GridExecutioner)
                ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(this);
        isOpen.set(true);

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT) {
            hostOffset.set(0);
            deviceOffset.set(0);

            if (currentSize.get() > 0)
                resetWorkspace();

        } else if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && (resetPlanned.get() || currentSize.get() == hostOffset.get() ) && currentSize.get() > 0) {
            //hostOffset.set(0);
            //deviceOffset.set(0);
            //resetPlanned.set(false);

            //if (currentSize.get() > 0)
                //resetWorkspace();

            //log.info("Resetting workspace at the end of loop... {} bytes ", hostOffset.get());
        }

        clearExternalAllocations();
        externalAllocations.clear();

        cycleAllocations.set(0);
        disabledCounter.set(0);

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
