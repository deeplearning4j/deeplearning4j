package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * A Jcuda buffer
 *
 * @author Adam Gibson
 */
public interface JCudaBuffer extends DataBuffer {

    /**
     * THe pointer for the buffer
     * @return the pointer for this buffer
     */
    public Pointer pointer();

    /**
     * Allocate the buffer
     */
    public void alloc();


    /**
     * The number of bytes for each individual element
     * @return the number of bytes for each individual element
     */
    public int elementSize();

    /**
     * Sets the data for this pointer
     * from the data in this pointer
     * @param pointer the pointer to set
     */
    void set(Pointer pointer);



}
