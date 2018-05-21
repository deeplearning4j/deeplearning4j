package org.nd4j.linalg.api.buffer.allocation;


import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 *
 * An allocation strategy handles allocating
 * and freeing memory for the gpu
 * (usually relative to the compute capabilities of the gpu)
 *
 * @author Adam Gibson
 */
public interface MemoryStrategy {


    /**
     * Set the data for the buffer
     * @param buffer the buffer to set
     * @param offset the offset to start at
     * @param stride the stride to sue
     * @param length the length to go till
     */
    void setData(DataBuffer buffer, int offset, int stride, int length);

    /**
     *
     * @param buffer
     * @param offset
     */
    void setData(DataBuffer buffer, int offset);

    /**
     * Copy data to native or gpu
     * @param copy the buffer to copy
     * @return a pointer representing
     * the copied data
     */
    Object copyToHost(DataBuffer copy, int offset);

    /**
     * Allocate memory for the given buffer
     * @param buffer the buffer to allocate for
     * @param stride the stride
     * @param offset the offset used for the buffer
     *               on allocation
     * @param length length
     */
    Object alloc(DataBuffer buffer, int stride, int offset, int length);

    /**
     * Free the buffer wrt the
     * allocation strategy
     * @param buffer the buffer to free
     * @param offset the offset to free
     * @param length the length to free
     */
    void free(DataBuffer buffer, int offset, int length);
}
