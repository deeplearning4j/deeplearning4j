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

import java.util.Map;


/**
 *
 * Direct allocation for free/destroy
 *
 * @author Adam Gibson
 */
public class PageableDirectBufferMemoryStrategy implements MemoryStrategy {
    @Override
    public Object copyToHost(DataBuffer copy,int offset) {
        JCudaBuffer buf2 = (JCudaBuffer) copy;
        Table<String, Pair<Integer,Integer>, BaseCudaDataBuffer.DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();

        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),new Pair<>(offset,buf2.length()));
        if(devicePointerInfo != null) {
            JCuda.cudaMemcpyAsync(
                    buf2.getHostPointer()
                    , devicePointerInfo.getPointer()
                    , devicePointerInfo.getLength()
                    , cudaMemcpyKind.cudaMemcpyDeviceToHost
                    , ContextHolder.getInstance().getCudaStream());
        }

        return buf2.getHostPointer();
    }

    @Override
    public Object alloc(DataBuffer buffer,int stride,int offset,int length) {
        Pointer hostData = new Pointer();
        JCuda.cudaMalloc(hostData,buffer.length() * buffer.getElementSize());
        return new BaseCudaDataBuffer.DevicePointerInfo(hostData,buffer.getElementSize() * buffer.length(),stride,offset);
    }

    @Override
    public void free(DataBuffer buffer,int offset,int length) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = buf2.getPointersToContexts().get(Thread.currentThread().getName(),new Pair<>(offset,length));
        JCuda.cudaFree(devicePointerInfo.getPointer());

    }
}
