package org.nd4j.storage;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.AbstractStorage;

/**
 * AbstractStorage implementation, with Integer as key.
 * Primary goal is storage of individual rows/slices in system ram
 *
 * @author raver119@gmail.com
 */
public class RamStorage<T extends Object> implements AbstractStorage<T> {

    /**
     * Store object into storage
     *
     * @param key
     * @param object
     */
    @Override
    public void store(T key, INDArray object) {

    }

    /**
     * Store object into storage, if it doesn't exist
     *
     * @param key
     * @param object
     */
    @Override
    public void storeIfAbsent(T key, INDArray object) {

    }

    /**
     * Get object from the storage, by key
     *
     * @param key
     */
    @Override
    public void get(T key) {

    }

    /**
     * This method checks, if storage contains specified key
     *
     * @param key
     * @return
     */
    @Override
    public boolean containsKey(T key) {
        return false;
    }
}
