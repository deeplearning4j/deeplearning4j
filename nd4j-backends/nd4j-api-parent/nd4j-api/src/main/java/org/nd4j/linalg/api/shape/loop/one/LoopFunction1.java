package org.nd4j.linalg.api.shape.loop.one;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.loop.four.*;
import org.nd4j.linalg.api.shape.loop.four.RawArrayIterationInformation4;

/**
 * Used for raw iteration in loops
 *
 * @author Adam Gibson
 */
public interface LoopFunction1 {
    /**
     * Perform an operation
     * given 2 buffers
     * @param a the first buffer
     * @param aOffset the first buffer offset
     * @param b the second buffer
     * @param bOffset the second buffer offset
     */
    void perform(int i, RawArrayIterationInformation4 info, DataBuffer a, int aOffset, DataBuffer b, int bOffset, DataBuffer c, int cOffset, DataBuffer d, int dOffset);
}
