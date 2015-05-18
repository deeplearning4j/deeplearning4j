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
     * Allocate memory for the given buffer
     * @param buffer the buffer to allocate for
     */
    Object alloc(DataBuffer buffer);

    /**
     * Free the buffer wrt the
     * allocation strategy
     * @param buffer the buffer to free
     */
    void free(DataBuffer buffer);

}
