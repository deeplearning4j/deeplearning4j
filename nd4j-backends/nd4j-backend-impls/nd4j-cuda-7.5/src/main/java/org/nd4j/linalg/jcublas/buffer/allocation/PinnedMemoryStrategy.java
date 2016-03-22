package org.nd4j.linalg.jcublas.buffer.allocation;

import com.google.common.collect.Table;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.buffer.DevicePointerInfo;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.util.NioUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;


/**
 * Pinned memory:
 * http://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/
 *
 *
 * @author Adam Gibson
 */
@Deprecated
public class PinnedMemoryStrategy implements MemoryStrategy {
    public PinnedMemoryStrategy() {
    }

    @Override
    public void getData(DataBuffer buffer, int offset, int stride, int length, DataBuffer get, CudaContext ctx, int getStride, int getOffset) {
        buffer.copyAtStride(get,length,stride,getStride,offset,getOffset);
    }

    @Override
    public void getData(DataBuffer buffer, int offset, DataBuffer get, CudaContext ctx) {
        getData(buffer,offset,1,buffer.length(),get,ctx,1,0);
    }

    @Override
    public void setData(Pointer buffer, int offset, int stride, int length, Pointer hostPointer) {

    }

    @Override
    public void setData(DataBuffer buffer, int offset, int stride, int length) {

    }

    @Override
    public void setData(DataBuffer buffer, int offset) {

    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, CudaContext context) {
        JCudaBuffer buf2 = (JCudaBuffer) copy;
        DevicePointerInfo devicePointerInfo =  null; //buf2.getPointersToContexts().get(Thread.currentThread().getName(), Triple.of(offset, buf2.length(), 1));
        HostDevicePointer hostDevicePointer = devicePointerInfo.getPointers();
        Pointer hostPointer = hostDevicePointer.getHostPointer();
        ByteBuffer pointer = hostPointer.getByteBuffer(0, copy.getElementSize() * copy.length()).order(ByteOrder.nativeOrder());
        ByteBuffer bufferNio = copy.asNio();
        // Flip and read from the original.
        //pointer.flip();
        //bufferNio.put(pointer);
        NioUtil.copyAtStride(buf2.length(),getBufferType(copy),pointer,offset,1,bufferNio,offset,1);
        return devicePointerInfo;
    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, int stride, int length, CudaContext context, int hostOffset, int hostStride) {
        ByteBuffer nio = copy.asNio();
        JCudaBuffer buf2 = (JCudaBuffer) copy;
        DevicePointerInfo devicePointerInfo = null; // buf2.getPointersToContexts().get(Thread.currentThread().getName(), Triple.of(offset, length, stride));
        HostDevicePointer hostDevicePointer = devicePointerInfo.getPointers();
        Pointer hostPointer = hostDevicePointer.getHostPointer();
        ByteBuffer pointer = hostPointer.getByteBuffer(0, copy.length() * copy.getElementSize());
        //copy at zero offset because offset is already taken care of in the above line. The view will start at
        //the given offset
        NioUtil.copyAtStride(length,getBufferType(copy),pointer,offset,stride,nio,hostOffset,hostStride);
        return devicePointerInfo;
    }

    @Override
    public Object alloc(DataBuffer buffer, int stride, int offset, int length, boolean initData) {
        Pointer devicePointer = new Pointer();
        Pointer hostPointer = new Pointer();
        JCuda.cudaHostAlloc(
                hostPointer
                , buffer.getElementSize() * length
                , JCuda.cudaHostAllocMapped);
        JCuda.cudaHostGetDevicePointer(
                devicePointer
                ,hostPointer
                ,0);
        DevicePointerInfo devicePointerInfo = new DevicePointerInfo(
                new HostDevicePointer(hostPointer,devicePointer)
                , length
                ,stride
                ,offset,false);

        if(initData) {
            ByteBuffer pointer = hostPointer.getByteBuffer(0, buffer.getElementSize() * buffer.length());
            pointer.order(ByteOrder.nativeOrder());
            NioUtil.copyAtStride(buffer.length(),getBufferType(buffer),buffer.asNio(),offset,stride,pointer,0,1);
        }
        return devicePointerInfo;
    }

    private NioUtil.BufferType getBufferType(DataBuffer buffer) {
        switch(buffer.dataType()) {
            case DOUBLE: return NioUtil.BufferType.DOUBLE;
            case INT: return NioUtil.BufferType.FLOAT;
            case FLOAT: return NioUtil.BufferType.FLOAT;
            default: throw new UnsupportedOperationException("Unsupported data buffer type");
        }
    }

    @Override
    public void free(DataBuffer buffer,int offset,int length) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Table<String, Triple<Integer,Integer,Integer>, DevicePointerInfo> pointers =  null; //buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointers.get(Thread.currentThread().getName(),Triple.of(offset,length,1));
        if(!devicePointerInfo.isFreed()) {
            JCuda.cudaFreeHost(devicePointerInfo.getPointers().getDevicePointer());
            devicePointerInfo.setFreed(true);
        }
    }

    @Override
    public void validate(DataBuffer buffer, CudaContext context) throws Exception {

    }
}
