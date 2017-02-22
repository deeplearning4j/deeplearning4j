package org.nd4j.aeron.ipc;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An ndarray listener
 * @author Adam Gibson
 */
public interface NDArrayCallback {


    /**
     * A listener for ndarray message
     * @param message the message for the callback
     */
    void onNDArrayMessage(NDArrayMessage message);

    /**
     * Used for partial updates using tensor along
     * dimension
     * @param arr the array to count as an update
     * @param idx the index for the tensor along dimension
     * @param dimensions the dimensions to act on for the tensor along dimension
     */
    void onNDArrayPartial(INDArray arr, long idx, int... dimensions);

    /**
     * Setup an ndarray
     * @param arr
     */
    void onNDArray(INDArray arr);

}
