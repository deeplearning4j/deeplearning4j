package org.nd4j.linalg.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
public interface ConstantHandler {
    long moveToConstantSpace(DataBuffer dataBuffer);

    DataBuffer getConstantBuffer(int[] array);

    DataBuffer getConstantBuffer(float[] array);

    DataBuffer getConstantBuffer(double[] array);
}
