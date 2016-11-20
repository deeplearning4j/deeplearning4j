package org.nd4j.linalg.cpu.nativecpu;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.memory.BasicMemoryManager;
import org.nd4j.linalg.memory.MemoryKind;

/**
 * @author raver119@gmail.com
 */
public class CpuMemoryManager extends BasicMemoryManager {
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
        return super.allocate(bytes, kind, initialize);
    }

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    @Override
    public void collect(INDArray... arrays) {
        super.collect(arrays);
    }
}
