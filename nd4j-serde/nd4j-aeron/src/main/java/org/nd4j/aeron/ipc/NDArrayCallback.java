package org.nd4j.aeron.ipc;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An ndarray listener
 * @author Adam Gibson
 */
public interface NDArrayCallback {
    /**
     * Setup an ndarray
     * @param arr
     */
    void onNDArray(INDArray arr);

}
