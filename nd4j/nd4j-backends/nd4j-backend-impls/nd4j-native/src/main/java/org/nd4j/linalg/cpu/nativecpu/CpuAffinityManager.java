package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public class CpuAffinityManager extends BasicAffinityManager {

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     * Has no effect on CPU backend.
     *
     * @param array
     */
    @Override
    public void touch(INDArray array) {
        // no-op
    }

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     * Has no effect on CPU backend.
     *
     * @param buffer
     */
    @Override
    public void touch(DataBuffer buffer) {
        // no-op
    }
}
