package org.nd4j.linalg.jcublas.buffer.allocation;

import com.google.common.collect.Table;
import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.buffer.DevicePointerInfo;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.NioUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;


/**
 *
 * Direct allocation for free/destroy
 *
 * @author Adam Gibson
 */
public class PageableDirectBufferMemoryStrategy implements MemoryStrategy {
    @Override
    public void getData(DataBuffer buffer, int offset, int stride, int length, DataBuffer get, CudaContext ctx, int getStride, int getOffset) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,buf2.length(),1));

        JCublas2.cublasGetVectorAsync(
                buffer.length()
                , buffer.getElementSize()
                , devicePointerInfo.getPointers().getDevicePointer()
                , stride
                , PointerUtil.getHostPointer(get)
                , getStride
                , ctx.getOldStream());
    }

    @Override
    public void getData(DataBuffer buffer, int offset, DataBuffer get, CudaContext ctx) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,buf2.length(),1));

        JCublas2.cublasGetVectorAsync(
                buffer.length()
                , buffer.getElementSize()
                , devicePointerInfo.getPointers().getDevicePointer()
                , 1
                , PointerUtil.getHostPointer(get)
                , 1
                , ctx.getOldStream());
    }

    @Override
    public void setData(Pointer buffer, int offset, int stride, int length, Pointer hostPointer) {

    }

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
        Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();

        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,buf2.length(),1));
        if(devicePointerInfo != null) {
            JCuda.cudaMemcpyAsync(
                    devicePointerInfo.getPointers().getHostPointer()
                    , devicePointerInfo.getPointers().getDevicePointer()
                    , devicePointerInfo.getLength()
                    , cudaMemcpyKind.cudaMemcpyDeviceToHost
                    , context.getOldStream());
        }

        return devicePointerInfo.getPointers().getHostPointer();
    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, int stride, int length, CudaContext context, int hostOffset, int hostStride) {
        JCudaBuffer buf2 = (JCudaBuffer) copy;
        Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> pointersToContexts = buf2.getPointersToContexts();

        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,length,1));
        if(devicePointerInfo != null) {
            JCuda.cudaMemcpyAsync(
                    devicePointerInfo.getPointers().getHostPointer()
                    , devicePointerInfo.getPointers().getDevicePointer()
                    , devicePointerInfo.getLength()
                    , cudaMemcpyKind.cudaMemcpyDeviceToHost
                    , context.getOldStream());
        };

        return devicePointerInfo.getPointers().getHostPointer();
    }

    @Override
    public Object alloc(DataBuffer buffer, int stride, int offset, int length, boolean initData) {
        JCudaBuffer buf2 = (JCudaBuffer) buffer;
        Pointer hostData = new Pointer();
        Pointer hostPointer = PointerUtil.getHostPointer(buffer);
        HostDevicePointer devicePointer = new HostDevicePointer(hostPointer,hostData);
        JCuda.cudaMalloc(hostData,buffer.length() * buffer.getElementSize());

        DevicePointerInfo devicePointerInfo = new DevicePointerInfo(devicePointer,buffer.getElementSize() * buffer.length(),stride,offset,false);
        if (initData) {
            // we'll have to use sync memcpy, to avoid passing CudaContext down here
            // FIXME: make that one cudaMemcpyAsync once again after we get nice way to pass CudaContext down here
            JCuda.cudaMemcpy(
                    devicePointerInfo.getPointers().getDevicePointer()
                    , devicePointerInfo.getPointers().getHostPointer()
                    , devicePointerInfo.getLength()
                    , cudaMemcpyKind.cudaMemcpyHostToDevice);

            // mark content as copied
            buf2.copied(Thread.currentThread().getName());
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
        DevicePointerInfo devicePointerInfo = buf2.getPointersToContexts().get(Thread.currentThread().getName(),Triple.of(offset,length, 1));
        if (devicePointerInfo != null && !devicePointerInfo.isFreed()) {
            JCuda.cudaFree(devicePointerInfo.getPointers().getDevicePointer());
        }

    }

    @Override
    public void validate(DataBuffer buffer, CudaContext context) throws Exception {
        throw new UnsupportedOperationException();
    }
}
