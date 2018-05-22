package org.nd4j.linalg.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * This interface describes
 * memory reuse strategy
 * for java-originated arrays.
 *
 * @author raver119@gmail.com
 */
public interface ConstantHandler {

    /**
     * If specific hardware supports dedicated constant memory,
     * this method forces DataBuffer passed in to be moved
     * to that constant memory.
     *
     * PLEASE NOTE: This method implementation is hardware-dependant.
     *
     * @param dataBuffer
     * @return
     */
    long moveToConstantSpace(DataBuffer dataBuffer);

    /**
     *
     * PLEASE NOTE: This method implementation is hardware-dependant.
     * PLEASE NOTE: This method does NOT allow concurrent use of any array
     *
     * @param dataBuffer
     * @return
     */
    DataBuffer relocateConstantSpace(DataBuffer dataBuffer);

    /**
     * This method returns DataBuffer with
     * constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that
     * you'll never ever change values
     * within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(int[] array);

    /**
     * This method returns DataBuffer with
     * constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that
     * you'll never ever change values
     * within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(long[] array);

    /**
     * This method returns DataBuffer
     * with constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll
     * never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(float[] array);

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(double[] array);

    /**
     * This method removes all cached constants
     */
    void purgeConstants();

    /**
     * This method returns memory used for cache, in bytes
     *
     * @return
     */
    long getCachedBytes();
}
