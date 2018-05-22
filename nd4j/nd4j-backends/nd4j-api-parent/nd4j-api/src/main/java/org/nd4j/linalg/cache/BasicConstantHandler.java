package org.nd4j.linalg.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Basic No-Op abstraction for ConstantHandler
 *
 * @author raver119@gmail.com
 */
public abstract class BasicConstantHandler implements ConstantHandler {
    @Override
    public long moveToConstantSpace(DataBuffer dataBuffer) {
        // no-op
        return 0L;
    }

    @Override
    public DataBuffer relocateConstantSpace(DataBuffer dataBuffer) {
        return dataBuffer;
    }


}
