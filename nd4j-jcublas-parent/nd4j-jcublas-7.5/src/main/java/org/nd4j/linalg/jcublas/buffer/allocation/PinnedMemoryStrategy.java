package org.nd4j.linalg.jcublas.buffer.allocation;

import com.google.common.collect.Table;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.buffer.DevicePointerInfo;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * Pinned memory:
 * http://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/
 *
 *
 * @author Adam Gibson
 */
public class PinnedMemoryStrategy implements MemoryStrategy {
    @Override
    public void getData(DataBuffer buffer, int offset, int stride, int length, DataBuffer get, CudaContext ctx, int getStride) {

    }

    @Override
    public void getData(DataBuffer buffer, int offset, DataBuffer get, CudaContext ctx) {

    }

    @Override
    public void setData(DataBuffer buffer, int offset, int stride, int length) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Table<String, Triple<Integer,Integer,Integer>, DevicePointerInfo> pointers = buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointers.get(Thread.currentThread().getName(),Triple.of(offset,length,1));
        JCuda.cudaHostGetDevicePointer(devicePointerInfo.getPointer(),buf2.getHostPointer(),0);
    }

    @Override
    public void setData(DataBuffer buffer, int offset) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Table<String, Triple<Integer,Integer,Integer>, DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,buf2.length(),1));
        JCuda.cudaHostGetDevicePointer(devicePointerInfo.getPointer(),buf2.getHostPointer(),0);

    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, CudaContext context) {
        JCudaBuffer buf2 = (JCudaBuffer) copy;
        Table<String, Triple<Integer,Integer,Integer>, DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,buf2.length(),1));
      /* JCuda.cudaMemcpyAsync(
                buf2.getHostPointer()
                , devicePointerInfo.getPointer()
                , devicePointerInfo.getLength()
                , cudaMemcpyKind.cudaMemcpyDeviceToHost
                , ContextHolder.getInstance().getCudaStream());
*/

        return buf2.getHostPointer();
    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, int stride, CudaContext context) {
        JCudaBuffer buf2 = (JCudaBuffer) copy;
        Table<String, Triple<Integer,Integer,Integer>, DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,buf2.length(),1));
      /* JCuda.cudaMemcpyAsync(
                buf2.getHostPointer()
                , devicePointerInfo.getPointer()
                , devicePointerInfo.getLength()
                , cudaMemcpyKind.cudaMemcpyDeviceToHost
                , ContextHolder.getInstance().getCudaStream());
*/

        return buf2.getHostPointer();
    }

    @Override
    public Object alloc(DataBuffer buffer,int stride,int offset,int length) {
        Pointer hostPointer = new Pointer();
        DevicePointerInfo devicePointerInfo = new DevicePointerInfo(
                hostPointer
                , length
                ,stride
                ,offset);


        JCuda.cudaHostAlloc(
                hostPointer
                , buffer.getElementSize() * length
                , JCuda.cudaHostAllocMapped);

        return devicePointerInfo;
    }

    @Override
    public void free(DataBuffer buffer,int offset,int length) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Table<String, Triple<Integer,Integer,Integer>, DevicePointerInfo> pointers = buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointers.get(Thread.currentThread().getName(),Triple.of(offset,length,1));
        if(!devicePointerInfo.isFreed()) {
            JCuda.cudaFreeHost(devicePointerInfo.getPointer());
            devicePointerInfo.setFreed(true);
        }
    }
}
