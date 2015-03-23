package org.nd4j.linalg.api.resources;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A resource manager is used for handling allocation of native resources
 * where applicable.
 *
 * A resource manager can be aggressive depending on the strategy
 * required by different backends.
 *
 * @author Adam Gibson
 */
public interface ResourceManager {
    /**
     * Register the ndarray with the resource manager
     * @param arr the array to register
     */
    void register(INDArray arr);


    void purge();

    /**
     * Returns true if the
     * data buffer should be collected or not
     * @param collect collect the data buffer to collect
     * @return the ndarray to connect
     */
    boolean shouldCollect(INDArray collect);



}
