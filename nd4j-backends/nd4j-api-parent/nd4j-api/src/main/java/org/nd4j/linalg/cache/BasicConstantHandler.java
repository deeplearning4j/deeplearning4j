package org.nd4j.linalg.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
public abstract class BasicConstantHandler implements ConstantHandler{
    @Override
    public long moveToConstantSpace(DataBuffer dataBuffer) {
        throw new UnsupportedOperationException("ConstantSpace is unavailable for x86 architecture");
    }
}
