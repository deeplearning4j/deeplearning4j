package org.nd4j.linalg.memory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author raver119@gmail.com
 */
public interface MemoryManager {

    /**
     * This method returns
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     */
    Pointer allocate(long bytes, MemoryKind kind, boolean initialize);

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    void collect(INDArray... arrays);


    /**
     * This method purges all cached memory chunks
     * 
     */
    void purgeCaches();
}
