package org.nd4j.linalg.memory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public class BasicMemoryManager implements MemoryManager {
    /**
     * This method returns
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     * @param kind
     * @param initialize
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        return null;
    }

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    @Override
    public void collect(INDArray... arrays) {

    }

    /**
     * This method purges all cached memory chunks
     *
     */
    @Override
    public void purgeCaches() {

    }
}
