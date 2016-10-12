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
}
