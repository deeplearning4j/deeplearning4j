package org.nd4j.linalg.api.instrumentation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Instrumentation for logging statistics
 * about the ndarrays being allocated and destroyed.
 *
 * @author Adam Gibson
 */
public interface Instrumentation {

    /**
     * Log the given ndarray
     * @param toLog the ndarray to log
     * @param status the status
     */
    void log(INDArray toLog,String status);

    /**
     * Data buffer to log
     * @param buffer the buffer to log
     * @param status the status
     */
    void log(DataBuffer buffer,String status);
    /**
     * Log the given ndarray
     * @param toLog the ndarray to log
     */
    void log(INDArray toLog);

    /**
     * Data buffer to log
     * @param buffer the buffer to log
     */
    void log(DataBuffer buffer);

}
