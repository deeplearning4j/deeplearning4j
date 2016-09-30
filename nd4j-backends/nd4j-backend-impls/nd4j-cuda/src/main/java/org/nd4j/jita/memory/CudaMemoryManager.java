package org.nd4j.jita.memory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.ops.executioner.CudaGridExecutioner;
import org.nd4j.linalg.memory.BasicMemoryManager;
import org.nd4j.linalg.memory.MemoryKind;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * @author raver119@gmail.com
 */
public class CudaMemoryManager extends BasicMemoryManager {

    /**
     * This method returns
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     * @param kind
     * @param initialize
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        AtomicAllocator allocator = AtomicAllocator.getInstance();

        if (kind == MemoryKind.HOST) {
            return allocator.getMemoryHandler().alloc(AllocationStatus.HOST, null, null, initialize).getHostPointer();
        } else if (kind == MemoryKind.DEVICE) {
            return allocator.getMemoryHandler().alloc(AllocationStatus.HOST, null, null, initialize).getDevicePointer();
        } else throw new RuntimeException("Unknown MemoryKind requested: " + kind);
    }

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    @Override
    public void collect(INDArray... arrays) {
        // we basically want to free memory, without touching INDArray itself.
        // so we don't care when gc is going to release object: memory is already cached

        ((CudaGridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        int cnt = -1;
        AtomicAllocator allocator = AtomicAllocator.getInstance();
        for(INDArray array: arrays) {
            cnt++;
            // we don't collect views, since they don't have their own memory
            if (array == null || array.isView())
                continue;

            AllocationPoint point = allocator.getAllocationPoint(array);

            if (point.getAllocationStatus() == AllocationStatus.HOST)
                allocator.getMemoryHandler().free(point, AllocationStatus.HOST);
            else if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
                allocator.getMemoryHandler().free(point, AllocationStatus.DEVICE);
                allocator.getMemoryHandler().free(point, AllocationStatus.HOST);
            } else if (point.getAllocationStatus() == AllocationStatus.DEALLOCATED) {
                // do nothing
            } else throw new RuntimeException("Unknown AllocationStatus: " + point.getAllocationStatus() + " for argument: " + cnt);

            point.setAllocationStatus(AllocationStatus.DEALLOCATED);
        }
    }

    /**
     * This method purges all cached memory chunks
     * PLEASE NOTE: This method SHOULD NOT EVER BE USED in concurrent environments
     */
    @Override
    public synchronized void purgeCaches() {
        // reset device cache offset
        Nd4j.getConstantHandler().purgeConstants();

        // reset TADs
        ((CudaGridExecutioner) Nd4j.getExecutioner()).getTadManager().purgeBuffers();

        // purge shapes
        Nd4j.getShapeInfoProvider().purgeCache();

        // purge memory cache
        AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().purgeCache();

    }
}
