package org.nd4j.linalg.jcublas.buffer.allocation;

import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

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
    void alloc(JCudaBuffer buffer);

    /**
     * Free the buffer wrt the
     * allocation strategy
     * @param buffer the buffer to free
     */
    void free(JCudaBuffer buffer);

}
