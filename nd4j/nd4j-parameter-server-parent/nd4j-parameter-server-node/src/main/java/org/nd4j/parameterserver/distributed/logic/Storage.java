package org.nd4j.parameterserver.distributed.logic;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface Storage {

    INDArray getArray(Integer key);

    void setArray(Integer key, INDArray array);

    boolean arrayExists(Integer key);

    void shutdown();
}
