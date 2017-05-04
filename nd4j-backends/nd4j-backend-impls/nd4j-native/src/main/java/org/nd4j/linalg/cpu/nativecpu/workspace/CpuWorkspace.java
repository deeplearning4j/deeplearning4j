package org.nd4j.linalg.cpu.nativecpu.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * CPU-only MemoryWorkspace implementation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CpuWorkspace extends Nd4jWorkspace {
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

    @Override
    protected void init() {
        super.init();

        if (currentSize.get() > 0) {
            isInit.set(true);


            log.debug("Allocating [{}] workspace of {} bytes...", id, currentSize.get());

            workspace.setHostPointer(new PagedPointer(memoryManager.allocate(currentSize.get() + SAFETY_OFFSET, MemoryKind.HOST, true)));
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

    @Override
    protected void clearExternalAllocations() {
        if (isDebug.get())
            log.info("Workspace [{}] device_{} threadId {} guid [{}]: clearing external allocations...", id, Nd4j.getAffinityManager().getDeviceForCurrentThread(), Thread.currentThread().getId(), guid);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        for (PointersPair pair: externalAllocations) {
            nativeOps.freeHost(pair.getHostPointer());
        }
        externalAllocations.clear();
        externalCount.set(0);
        spilledAllocationsSize.set(0);
    }

    @Override
    public synchronized void destroyWorkspace(boolean extended) {
        currentSize.set(0);
        hostOffset.set(0);
        deviceOffset.set(0);

        if (extended)
            clearExternalAllocations();

        stepsCount.set(Long.MAX_VALUE - 100);
        clearPinnedAllocations(extended);

        if (workspace.getHostPointer() != null)
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(workspace.getHostPointer());

        workspace.setDevicePointer(null);
        workspace.setHostPointer(null);
    }

    @Override
    protected void resetWorkspace() {
        //Pointer.memset(workspace.getHostPointer(), 0, currentSize.get() + SAFETY_OFFSET);
    }
}
