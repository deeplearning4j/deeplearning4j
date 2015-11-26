package org.nd4j.linalg.jcublas.buffer.allocation;

import com.google.common.collect.Table;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.allocation.MemoryStrategy;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;

/**
 * Pinned memory:
 * http://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/
 *
 *
 * @author Adam Gibson
 */
public class PinnedMemoryStrategy implements MemoryStrategy {
    @Override
    public void setData(DataBuffer buffer, int offset, int stride, int length) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;

    }

    @Override
    public void setData(DataBuffer buffer, int offset) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;

    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, CudaContext context) {
        JCudaBuffer buf2 = (JCudaBuffer) copy;
        Table<String, Pair<Integer,Integer>, BaseCudaDataBuffer.DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();
        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),new Pair<>(offset,buf2.length()));
        JCuda.cudaMemcpyAsync(
                buf2.getHostPointer()
                , devicePointerInfo.getPointer()
                , devicePointerInfo.getLength()
                , cudaMemcpyKind.cudaMemcpyDeviceToHost
                , ContextHolder.getInstance().getCudaStream());


        return buf2.getHostPointer();
    }

    @Override
    public Object alloc(DataBuffer buffer,int stride,int offset,int length) {
        Pointer hostPointer = new Pointer();
        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = new BaseCudaDataBuffer.DevicePointerInfo(
                hostPointer
                , length
                ,stride
                ,offset);

        JCuda.cudaHostAlloc(
                hostPointer
                , buffer.getElementSize() * length
                , JCuda.cudaHostAllocDefault);

        return devicePointerInfo;
    }

    @Override
    public void free(DataBuffer buffer,int offset,int length) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Table<String, Pair<Integer,Integer>, BaseCudaDataBuffer.DevicePointerInfo> pointers = buf2.getPointersToContexts();
        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = pointers.get(Thread.currentThread().getName(),new Pair<>(offset,length));
        if(!devicePointerInfo.isFreed()) {
            JCuda.cudaFreeHost(devicePointerInfo.getPointer());
            devicePointerInfo.setFreed(true);
        }
    }
}
