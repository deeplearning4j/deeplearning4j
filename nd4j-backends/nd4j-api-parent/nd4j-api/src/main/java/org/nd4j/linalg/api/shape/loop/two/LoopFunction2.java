package org.nd4j.linalg.api.shape.loop.two;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Used for raw iteration in loops
 *
 * @author Adam Gibson
 */
public interface LoopFunction2 {
    /**
     * Perform an operation
     * given 2 buffers
     * @param a the first buffer
     * @param aOffset the first buffer offset
     * @param b the second buffer
     * @param bOffset the second buffer offset
     */
    void perform(int i,RawArrayIterationInformation2 info,DataBuffer a,int aOffset,DataBuffer b,int bOffset);
}
