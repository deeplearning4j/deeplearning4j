package org.nd4j.linalg.jcublas.buffer.allocation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * @author Adam Gibson
 */
public class PageableArrayMemoryStrategy implements MemoryStrategy {
    @Override
    public void getData(DataBuffer buffer, int offset, int stride, int length, DataBuffer get, CudaContext ctx, int getStride) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void getData(DataBuffer buffer, int offset, DataBuffer get, CudaContext ctx) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void setData(DataBuffer buffer, int offset, int stride, int length) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void setData(DataBuffer buffer, int offset) {
        throw new UnsupportedOperationException();

    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, CudaContext context) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Object copyToHost(DataBuffer copy, int offset, int stride, CudaContext context) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Object alloc(DataBuffer buffer,int stride,int offset,int length) {
       throw new UnsupportedOperationException();
    }

    @Override
    public void free(DataBuffer buffer, int offset, int length) {
        throw new UnsupportedOperationException();
    }

}
