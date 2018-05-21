package org.nd4j.linalg.memory.stash;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This interface describe short-living storage, with pre-defined life time.
 *
 * @author raver119@gmail.com
 */
public interface Stash<T extends Object> {

    boolean checkIfExists(T key);

    void put(T key, INDArray object);

    INDArray get(T key);

    void purge();
}
