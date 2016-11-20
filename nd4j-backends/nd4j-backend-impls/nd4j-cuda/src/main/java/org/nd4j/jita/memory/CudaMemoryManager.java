package org.nd4j.jita.memory;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.ops.executioner.CudaGridExecutioner;
import org.nd4j.linalg.memory.BasicMemoryManager;
import org.nd4j.linalg.memory.MemoryKind;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * @author raver119@gmail.com
 */
@Slf4j
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
     * PLEASE NOTE: This method SHOULD NOT EVER BE USED without being 146% clear of all consequences.
     */
    @Override
    public synchronized void purgeCaches() {
        // reset device cache offset
//        Nd4j.getConstantHandler().purgeConstants();

        // reset TADs
//        ((CudaGridExecutioner) Nd4j.getExecutioner()).getTadManager().purgeBuffers();

        // purge shapes
//        Nd4j.getShapeInfoProvider().purgeCache();

        // purge memory cache
        AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().purgeCache();

    }

    /**
     * This method provides basic memcpy functionality with respect to target environment
     *
     * @param dstBuffer
     * @param srcBuffer
     */
    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        if (dstBuffer instanceof CompressedDataBuffer && !(srcBuffer instanceof CompressedDataBuffer)) {
            // destination is compressed, source isn't
            AllocationPoint srcPoint = AtomicAllocator.getInstance().getAllocationPoint(srcBuffer);
            long size = srcBuffer.getElementSize() * srcBuffer.length();
            if (!srcPoint.isActualOnHostSide()) {
                // copying device -> host

                AtomicAllocator.getInstance().synchronizeHostData(srcBuffer);

               // Pointer src = AtomicAllocator.getInstance().getPointer(srcBuffer, context);

               // NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstBuffer.addressPointer(), src, size, 2, context.getSpecialStream());
               // context.syncSpecialStream();

            }// else {
                // copying host -> host
                Pointer src = AtomicAllocator.getInstance().getHostPointer(srcBuffer);

                Pointer.memcpy(dstBuffer.addressPointer(), src, size);
           // }

        } else if (!(dstBuffer instanceof CompressedDataBuffer) && srcBuffer instanceof CompressedDataBuffer) {
            // destination is NOT compressed, source is compressed
            AllocationPoint dstPoint = AtomicAllocator.getInstance().getAllocationPoint(dstBuffer);
            long size = srcBuffer.getElementSize() * srcBuffer.length();

            Pointer.memcpy(dstBuffer.addressPointer(), srcBuffer.addressPointer(), size);
            dstPoint.tickHostWrite();

        } else if (dstBuffer instanceof CompressedDataBuffer && srcBuffer instanceof CompressedDataBuffer) {
            // both buffers are compressed, just fire memcpy


            Pointer.memcpy(dstBuffer.addressPointer(), srcBuffer.addressPointer(), srcBuffer.length() * srcBuffer.getElementSize());
        } else {
            // both buffers are NOT compressed
            AtomicAllocator.getInstance().memcpy(dstBuffer, srcBuffer);
        }
    }
}
