package org.nd4j.linalg.jcublas.buffer.allocation;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.allocation.MemoryStrategy;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;


/**
 *
 * Direct allocaation for free/destroy
 *
 * @author Adam Gibson
 */
public class PageableDirectBufferMemoryStrategy implements MemoryStrategy {
    @Override
    public Object alloc(DataBuffer buffer) {
        Pointer hostData = new Pointer();
        JCuda.cudaMalloc(hostData,buffer.length() * buffer.getElementSize());
        return new BaseCudaDataBuffer.DevicePointerInfo(hostData,buffer.getElementSize() * buffer.length());
    }

    @Override
    public void free(DataBuffer buffer) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = buf2.getPointersToContexts().get(Thread.currentThread().getName());
        BaseCudaDataBuffer.checkResult(JCuda.cudaFree(devicePointerInfo.getPointer()));

    }
}
