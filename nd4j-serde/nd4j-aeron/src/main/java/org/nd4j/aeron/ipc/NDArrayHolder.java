package org.nd4j.aeron.ipc;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * A simple interface for retrieving an
 * ndarray
 *
 * @author Adam Gibson
 */
public interface NDArrayHolder extends Serializable {


    /**
     * Set the ndarray
     * @param arr the ndarray for this holder
     *            to use
     */
    void setArray(INDArray arr);


    /**
     * The number of updates
     * that have been sent to this older.
     * @return
     */
    int totalUpdates();

    /**
     * Retrieve an ndarray
     * @return
     */
    INDArray get();

    /**
     * Retrieve a partial view of the ndarray.
     * This method uses tensor along dimension internally
     * Note this will call dup()
     * @param idx the index of the tad to get
     * @param dimensions the dimensions to use
     * @return the tensor along dimension based on the index and dimensions
     * from the master array.
     */
    INDArray getTad(int idx, int... dimensions);
}
