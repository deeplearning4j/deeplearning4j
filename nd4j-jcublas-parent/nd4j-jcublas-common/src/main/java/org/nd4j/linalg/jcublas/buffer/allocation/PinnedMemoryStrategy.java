package org.nd4j.linalg.jcublas.buffer.allocation;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.allocation.MemoryStrategy;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

/**
 * @author Adam Gibson
 */
public class PinnedMemoryStrategy implements MemoryStrategy {
    @Override
    public Object alloc(DataBuffer buffer) {
        Pointer hostPointer = new Pointer();
        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = new BaseCudaDataBuffer.DevicePointerInfo(hostPointer, buffer.length());
        BaseCudaDataBuffer.checkResult(JCuda.cudaHostAlloc(hostPointer, buffer.getElementSize() * buffer.length(), JCuda.cudaHostAllocPortable));
        return devicePointerInfo;
    }

    @Override
    public void free(DataBuffer buffer) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        BaseCudaDataBuffer.DevicePointerInfo devicePointerInfo = buf2.getPointersToContexts().get(Thread.currentThread().getName());
        BaseCudaDataBuffer.checkResult(JCuda.cudaFreeHost(devicePointerInfo.getPointer()));


    }
}
