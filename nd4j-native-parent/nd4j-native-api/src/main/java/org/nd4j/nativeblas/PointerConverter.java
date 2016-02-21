package org.nd4j.nativeblas;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.Buffer;

/**
 * Pointer converter finds the underlying pointer
 * address relative to the data source.
 *
 * @author Adam Gibson
 */
public interface PointerConverter {
    /**
     * Get the underlying address for the array
     * @param arr the array to get the underlying address for
     * @return
     */
    long toPointer(IComplexNDArray arr);
    /**
     * Get the underlying address for the array
     * @param arr the array to get the underlying address for
     * @return
     */
    long toPointer(INDArray arr);
    /**
     * Get the underlying address for the array
     * @param buffer the array to get the underlying address for
     * @return
     */
    long toPointer(Buffer buffer);




}
