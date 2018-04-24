package org.nd4j.linalg.memory.abstracts;

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemoryManager;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
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
 * Basic implementation for
 * MemoryWorkspace interface,
 * further extended in corresponding backends
 *
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class Nd4jWorkspace implements MemoryWorkspace {
    @Getter
    protected int deviceId;
    @Getter
    protected Long threadId;

    protected Type workspaceType = Type.SCOPED;

    protected static final long SAFETY_OFFSET = 1024L;

    @Getter
    protected String id;

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
    protected int stepsNumber = 1;

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

    @Getter
    protected final WorkspaceConfiguration workspaceConfiguration;

    // external allocations are purged at the end of loop
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    // pinned allocations are purged with delay, used for circular mode only
    protected Queue<PointersPair> pinnedAllocations = new LinkedTransferQueue<>();

    @Setter
    protected MemoryWorkspace previousWorkspace;
    protected MemoryWorkspace borrowingWorkspace;

    protected AtomicLong initialBlockSize = new AtomicLong(0);

    protected String guid;

    protected File tempFile;

    protected AtomicLong generationId = new AtomicLong(0);

    // this memory manager implementation will be used to allocate real memory for this workspace

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this(configuration, DEFAULT_ID);
    }

    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        this.workspaceConfiguration = configuration;
        this.id = workspaceId;
        this.threadId = Thread.currentThread().getId();
        this.guid = Nd4j.getWorkspaceManager().getUUID();
        this.memoryManager = Nd4j.getMemoryManager();
        this.deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // and actual workspace allocation
        currentSize.set(workspaceConfiguration.getInitialSize());

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED)
            workspaceType = Type.CIRCULAR;
        else
            workspaceType = Type.SCOPED;

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED
                        && workspaceConfiguration.getPolicyAllocation() == AllocationPolicy.OVERALLOCATE) {
            if (workspaceConfiguration.getOverallocationLimit() < 1.0)
                throw new ND4JIllegalStateException(
                                "For cyclic workspace overallocation should be positive integral value.");

            stepsNumber = (int) (workspaceConfiguration.getOverallocationLimit() + 1);
            log.debug("Steps: {}", stepsNumber);
        }

        //if (workspaceConfiguration.getPolicyLearning() == LearningPolicy.OVER_TIME && workspaceConfiguration.getCyclesBeforeInitialization() < 1)
        //log.warn("Workspace [{}]: initialization OVER_TIME was selected, but number of cycles isn't positive value!", id);

        // validate mmap option
        if (configuration.getPolicyLocation() == LocationPolicy.MMAP) {
            // file path should be either non-null
            if (configuration.getTempFilePath() != null) {
                    tempFile = new File(configuration.getTempFilePath());

                    if (tempFile.length() == 0 || tempFile.length() < configuration.getInitialSize()) {
                        if (configuration.getInitialSize() > 0) {
                            try {
                                fillFile(tempFile, configuration.getInitialSize());
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        } else {
                            throw new ND4JIllegalStateException("Memory-mapped file should have positive length.");
                        }
                    } else {
                        configuration.setInitialSize(tempFile.length());
                    }
            } else if (configuration.getInitialSize() > 0) {
                try {
                    tempFile = File.createTempFile("workspace", "tempMMAP");
                    tempFile.deleteOnExit();

                    // fill temp file with zeroes, up to initialSize bytes
                    fillFile(tempFile, configuration.getInitialSize());

                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else
                throw new ND4JIllegalStateException("MMAP target file path should be non-null or workspace initialSize should be >0 for temp file");
        }

        init();
    }

    @Override
    public Type getWorkspaceType() {
        return this.workspaceType;
    }

    public static void fillFile(File file, long length) throws Exception {
        byte[] buffer = new byte[16384];
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = (byte) 0;
        }

        try (FileOutputStream fos = new FileOutputStream(file); BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            long written = 0;
            while (written < length) {
                fos.write(buffer);
                written += buffer.length;
            }
        }
    }

    @Override
    public long getGenerationId() {
        return generationId.get();
    }

    /**
     * This method returns step number. Viable only in circular mode.
     * @return
     */
    public long getStepNumber() {
        return stepsCount.get();
    }

    /**
     * This method returns number of bytes in spilled allocations.
     * @return
     */
    public long getSpilledSize() {
        return spilledAllocationsSize.get();
    }

    /**
     * This method returns number of bytes in pinned allocations.
     * @return
     */
    public long getPinnedSize() {
        return pinnedAllocationsSize.get();
    }

    /**
     * This method returns number of bytes for first block of circular workspace.
     * @return
     */
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

    /**
     * This method returns current device memory offset within workspace
     * @return
     */
    public long getDeviceOffset() {
        return deviceOffset.get();
    }

    /**
     * This method returns current host memory offset within workspace
     * @return
     */
    public long getHostOffset() {
        return hostOffset.get();
    }

    /**
     * This method returns current amount of memory allocated for workspace.
     *
     * PLEASE NOTE: It shows only amount of HOST memory.
     * If current backend assumes DEVICE/HOST memory pair,
     * DEVICE memory will probably have the same size, but won't be accounted in this value.
     * @return
     */
    public long getCurrentSize() {
        return currentSize.get();
    }

    protected void init() {
        //  we want params validation here

        if (currentSize.get() > 0) {
            if (!isOver.get()) {
                if (workspaceConfiguration.getPolicyAllocation() == AllocationPolicy.OVERALLOCATE
                                && workspaceConfiguration.getOverallocationLimit() > 0) {
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
            2) go for either external spilled, or pinned allocation
         */
        // we enforce 8 byte alignment to ensure CUDA doesn't blame us
        long div = requiredMemory % 8;
        if (div != 0)
            requiredMemory += div;

        long numElements = requiredMemory / Nd4j.sizeOfDataType(type);

        // shortcut made to skip workspace
        if (!isUsed.get()) {
            if (disabledCounter.incrementAndGet() % 10 == 0)
                log.warn("Workspace was turned off, and wasn't enabled after {} allocations", disabledCounter.get());

            PagedPointer pointer = new PagedPointer(memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize),
                            numElements);

            externalAllocations.add(new PointersPair(pointer, null));

            return pointer;
        }

        /*
            Trimmed mode is possible for cyclic workspace mode. Used in AsyncDataSetIterator, MQ, etc.
            Basically idea is simple: if one of datasets coming out of iterator has size higher then expected - we should reallocate workspace to match this size.
            So, we switch to trimmed mode, and all allocations will be "pinned", and eventually workspace will be reallocated.
         */
        boolean trimmer = (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED
                        && requiredMemory + cycleAllocations.get() > initialBlockSize.get()
                        && initialBlockSize.get() > 0) || trimmedMode.get();

        if (trimmer && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE && !trimmedMode.get()) {
            trimmedMode.set(true);
            trimmedStep.set(stepsCount.get());
        }

        // if size is enough - allocate from workspace
        if (hostOffset.get() + requiredMemory <= currentSize.get() && !trimmer) {
            // just alignment to 8 bytes

            cycleAllocations.addAndGet(requiredMemory);
            long prevOffset = hostOffset.getAndAdd(requiredMemory);
            deviceOffset.set(hostOffset.get());

            PagedPointer ptr = workspace.getHostPointer().withOffset(prevOffset, numElements);

            if (isDebug.get())
                log.info("Workspace [{}]: Allocating array of {} bytes, capacity of {} elements, prevOffset: {}; currentOffset: {}; address: {}",
                                id, requiredMemory, numElements, prevOffset, hostOffset.get(), ptr.address());

            if (initialize)
                Pointer.memset(ptr, 0, requiredMemory);

            return ptr;
        } else {
            // if current workspace isn't enough - we allocate it separately as spilled (or pinned, in case of circular mode)

            // in case of circular mode - we just reset offsets, and start from the beginning of the workspace
            if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED && currentSize.get() > 0
                            && !trimmer) {
                reset();
                resetPlanned.set(true);
                return alloc(requiredMemory, kind, type, initialize);
            }

            // updating respective counters
            if (!trimmer)
                spilledAllocationsSize.addAndGet(requiredMemory);
            else
                pinnedAllocationsSize.addAndGet(requiredMemory);

            if (isDebug.get())
                log.info("Workspace [{}]: step: {}, spilled  {} bytes, capacity of {} elements", id, stepsCount.get(),
                                requiredMemory, numElements);

            switch (workspaceConfiguration.getPolicySpill()) {
                case REALLOCATE:
                case EXTERNAL:
                    cycleAllocations.addAndGet(requiredMemory);
                    if (!trimmer) {
                        externalCount.incrementAndGet();

                        PagedPointer pointer = new PagedPointer(
                                        memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize),
                                        numElements);

                        externalAllocations.add(new PointersPair(pointer, null));

                        return pointer;
                    } else {
                        pinnedCount.incrementAndGet();
                        PagedPointer pointer = new PagedPointer(
                                        memoryManager.allocate(requiredMemory, MemoryKind.HOST, initialize),
                                        numElements);

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
        // we can reallocate this workspace to larger size if that's needed and allowed by configuration
        if ((currentSize.get() < maxCycle.get() || currentSize.get() < cycleAllocations.get())
                        && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE
                        && (workspaceConfiguration.getMaxSize() == 0
                                        || (maxCycle.get() < workspaceConfiguration.getMaxSize()))) {
            if (workspaceConfiguration.getPolicyReset() != ResetPolicy.ENDOFBUFFER_REACHED) {
                destroyWorkspace(true);
                isInit.set(false);
            }
        }

        // if we're in cyclic mode, we do reallocations only after 2 full cycles, to avoid race conditions
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

                // if we're on cyclic mode, let's add 30% to size, just to reduce number of reallocations
                if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED) {
                    currentSize.set((long) (currentSize.get() * 1.3));
                    currentSize.addAndGet(8 - (currentSize.get() % 8));
                    maxCycle.set(currentSize.get());
                }

                // we're updating single block size for circular mode, will be used for alignment later
                initialBlockSize.set(currentSize.get());

                // handliong optional overallocation here, however it's usually good idea to use it everywhere, to avoid frequent realloc calls
                if (!isOver.get()) {
                    if (workspaceConfiguration.getPolicyAllocation() == AllocationPolicy.OVERALLOCATE
                                    && workspaceConfiguration.getOverallocationLimit() > 0 && currentSize.get() > 0) {
                        currentSize.set(currentSize.get()
                                        + (long) (currentSize.get() * workspaceConfiguration.getOverallocationLimit()));
                        isOver.set(true);
                    }
                }

                if (workspaceConfiguration.getMinSize() > 0 && currentSize.get() < workspaceConfiguration.getMinSize())
                    currentSize.set(workspaceConfiguration.getMinSize());

                // purge spilled allocations
                if (externalCount.get() > 0 && (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT
                                || resetPlanned.get())) {
                    clearExternalAllocations();
                    resetPlanned.set(false);
                }

                // calling for implementation-specific workspace initialization. basically allocation happens there
                init();
            }
    }

    /**
     * This method returns number of spilled allocations, that can be purged at the end of block
     * @return
     */
    public int getNumberOfExternalAllocations() {
        return externalCount.get();
    }

    /**
     * This method returns number of pinned allocations, they can be purged after 2 steps.
     *
     * PLEASE NOTE: This method can return non-zero calues only for circular workspace mode
     * @return
     */
    public int getNumberOfPinnedAllocations() {
        return pinnedCount.get();
    }

    @Override
    public void destroyWorkspace() {
        destroyWorkspace(true);
    }


    /**
     * This method basically deallocates workspace memory
     *
     * @param extended
     */
    @Override
    public void destroyWorkspace(boolean extended) {
        if (workspace.getHostPointer() != null && workspace.getHostPointer().getOriginalPointer() != null
                        && workspace.getHostPointer().getOriginalPointer() instanceof BytePointer)
            workspace.getHostPointer().getOriginalPointer().deallocate();

        workspace.setHostPointer(null);
        currentSize.set(0);
        reset();

        if (extended) {
            clearExternalAllocations();
        }

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
            throw new ND4JIllegalStateException("Workspace [" + id + "]: Can't borrow from borrowed workspace");

        borrowingWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        isBorrowed.set(true);

        Nd4j.getMemoryManager().setCurrentWorkspace(this);

        return this;
    }

    public long getCyclesCount() {
        return cyclesCount.get();
    }

    @Override
    public void close() {
        // first we check if this workspace was borrowed. if yes - just close without reset.
        if (isBorrowed.get()) {
            isBorrowed.set(false);
            Nd4j.getMemoryManager().setCurrentWorkspace(borrowingWorkspace);
            return;
        }

        // next we check, if the same workspace was opened multiple times sequentially. then we just decrement counter, without reset
        if (tagScope.get() > 0) {
            if (tagScope.decrementAndGet() == 0) {
                Nd4j.getMemoryManager().setCurrentWorkspace(this);
            }
            return;
        }

        // this is for safety. We have to be sure that no ops were left non-processed
        //Furthermore, need to commit before marking workspace as closed, to avoid (incorrectly) hitting scope panic
        Nd4j.getExecutioner().commit();

        // since this workspace block is finished, we restore previous one. Even if it's null
        Nd4j.getMemoryManager().setCurrentWorkspace(previousWorkspace);
        isOpen.set(false);

        // just counter for cycles/blocks
        cyclesCount.incrementAndGet();
        if (cyclesCount.get() > 1 & (cyclesCount.get() - 1) % stepsNumber == 0) {
            // this counter is for cyclic mode, it counts generations, full loops over buffer
            stepsCount.incrementAndGet();
        }
        /*
            Basically all we want here, is:
            1) memset primary page(s)
            2) purge external allocations
         */

        if (!isUsed.get()) {
            log.warn("Workspace was turned off, and wasn't ever turned on back again");
            isUsed.set(true);
        }

        // if during this cycle we've used more memory then before - increase max count. we'll use it in future for optional reallocation
        if (cycleAllocations.get() > maxCycle.get()) {
            if (isDebug.get())
                log.info("Workspace [{}] device_{}, current cycle: {}; max cycle: {}", id,
                                Nd4j.getAffinityManager().getDeviceForCurrentThread(), cycleAllocations.get(),
                                maxCycle.get());

            maxCycle.set(cycleAllocations.get());
        }

        // checking, if we should reallocate this workspace to higher amount of memory
        if (workspaceConfiguration.getPolicyLearning() != LearningPolicy.NONE && maxCycle.get() > 0) {
            //log.info("Delayed workspace {}, device_{} initialization starts...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread());

            // if we're going to resize - we're probably safe to purge spilled allocations
            if (externalCount.get() > 0 && (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT
                            || resetPlanned.get())) {
                clearExternalAllocations();
                resetPlanned.set(false);
            }

            if ((workspaceConfiguration.getPolicyLearning() == LearningPolicy.OVER_TIME
                            && workspaceConfiguration.getCyclesBeforeInitialization() == cyclesCount.intValue())
                            || (workspaceConfiguration.getPolicyLearning() == LearningPolicy.FIRST_LOOP
                                            && currentSize.get() == 0)) {
                //log.info("Initializing on cycle {}", cyclesCount.get());
                initializeWorkspace();
            } else if (currentSize.get() > 0 && cycleAllocations.get() > 0
                            && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE
                            && workspaceConfiguration.getPolicyReset() != ResetPolicy.ENDOFBUFFER_REACHED) {
                //log.debug("Reinit on cycle {}; step: {}", cyclesCount.get(), stepsCount.get());
                initializeWorkspace();
            }
        }


        // clearing pinned allocations that are old enough
        if (pinnedCount.get() > 0)
            clearPinnedAllocations(false);

        // if we're in trimmed mode (preparing for reallocation of circular buffer) - we can do it 2 generations after
        if (trimmedMode.get() && trimmedStep.get() + 2 < stepsCount.get()) {
            initialBlockSize.set(maxCycle.get());
            initializeWorkspace();
            trimmedMode.set(false);
            trimmedStep.set(0);

            reset();
        }

        lastCycleAllocations.set(cycleAllocations.get());

        disabledCounter.set(0);


        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT) {
            reset();
        } else if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED
                        && currentSize.get() > 0) {

            // for variable input we want to ensure alignment to max block, to avoid accidental buffer overruns
            long diff = initialBlockSize.get() - cycleAllocations.get();

            // we don't care about offsets if that's trimmed mode, offsets will be reset anyway upon reallocation
            if (diff > 0 && !trimmedMode.get() && deviceOffset.get() > 0) {

                if (isDebug.get())
                    log.info("Worskpace [{}]: Align to [{}]; diff: [{}]; block size: [{}]; currentOffset: [{}]; workspaceSize: [{}]; trimmedMode: {}",
                                    id, initialBlockSize.get(), diff, cycleAllocations.get(), deviceOffset.get(),
                                    currentSize.get(), trimmedMode.get());

                deviceOffset.getAndAdd(diff);
                hostOffset.getAndAdd(diff);
            }
        }

        cycleAllocations.set(0);
    }

    protected abstract void clearPinnedAllocations(boolean extended);

    protected abstract void clearExternalAllocations();

    @Override
    public MemoryWorkspace notifyScopeEntered() {
        // we should block stuff since we're going to invalidate spilled allocations
        // TODO: block on spilled allocations probably?

        MemoryWorkspace prev = Nd4j.getMemoryManager().getCurrentWorkspace();

        // if we're opening the same workspace - just increase counter, and skip everything else
        if (prev == this && isOpen.get()) {
            tagScope.incrementAndGet();
            return this;
        }

        // we'll need this in close() call, to restore previous workspace (if any)
        previousWorkspace = prev;

        Nd4j.getMemoryManager().setCurrentWorkspace(this);
        isOpen.set(true);

        // resetting workspace to 0 offset (if anything), not applicable to circular mode, sure
        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT) {
            reset();
        }

        // if we have any spilled allocations left from last cycle - purge them.
        if (externalCount.get() > 0
                        && (workspaceConfiguration.getPolicyReset() == ResetPolicy.BLOCK_LEFT || resetPlanned.get())) {
            clearExternalAllocations();
            resetPlanned.set(false);
        }

        cycleAllocations.set(0);
        disabledCounter.set(0);

        generationId.incrementAndGet();

        return this;
    }

    /**
     * This method reset host/device offsets within workspace
     *
     * PLEASE NOTE: Never call this method unless you realize all consequences
     */
    public void reset() {
        //log.info("Resetting at device: {}; host: {};", deviceOffset.get(), hostOffset.get());
        hostOffset.set(0);
        deviceOffset.set(0);
    }

    protected abstract void resetWorkspace();

    /**
     * This method is shortcut to close() method
     *
     * @return
     */
    @Override
    public MemoryWorkspace notifyScopeLeft() {
        close();
        return this;
    }

    /**
     * This method allows to temporary disable this workspace, and issue allocations directly.
     * @param isEnabled
     */
    @Override
    public void toggleWorkspaceUse(boolean isEnabled) {
        isUsed.set(isEnabled);
    }

    /**
     * This method returns number of bytes allocated during last full cycle
     * @return
     */
    @Override
    public long getLastCycleAllocations() {
        return lastCycleAllocations.get();
    }

    /**
     * This method returns number of bytes allocated during THIS cycle
     * @return
     */
    @Override
    public long getThisCycleAllocations() {
        return cycleAllocations.get();
    }

    /**
     * This method returns number of bytes of biggest cycle
     * @return
     */
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
        return "Nd4jWorkspace{" + "id='" + id + '\'' + ", currentSize=" + currentSize.get() + '}';
    }

    @Data
    public static class GarbageWorkspaceReference extends WeakReference<MemoryWorkspace> {
        private PointersPair pointersPair;
        private String id;
        private Long threadId;
        private Queue<PointersPair> pinnedPointers;
        private List<PointersPair> externalPointers;
        private String key;

        public GarbageWorkspaceReference(MemoryWorkspace referent, ReferenceQueue<? super MemoryWorkspace> queue) {
            super(referent, queue);
            this.pointersPair = ((Nd4jWorkspace) referent).workspace;

            this.id = referent.getId();
            this.threadId = referent.getThreadId();
            this.pinnedPointers = ((Nd4jWorkspace) referent).pinnedAllocations;
            this.externalPointers = ((Nd4jWorkspace) referent).externalAllocations;

            this.key = id + "_" + threadId;
        }
    }
}
