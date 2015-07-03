package org.nd4j.linalg.jcublas.buffer.allocation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.allocation.MemoryStrategy;

/**
 * @author Adam Gibson
 */
public class PageableArrayMemoryStrategy implements MemoryStrategy {
    @Override
    public Object copyToHost(DataBuffer copy,int offset) {
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
