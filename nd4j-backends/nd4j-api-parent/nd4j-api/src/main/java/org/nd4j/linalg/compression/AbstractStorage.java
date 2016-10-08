package org.nd4j.linalg.compression;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface AbstractStorage<T extends Object> {

    /**
     * Store object into storage
     *
     * @param key
     * @param object
     */
    void store(T key, INDArray object);

    /**
     * Store object into storage, if it doesn't exist
     *
     * @param key
     * @param object
     */
    void storeIfAbsent(T key, INDArray object);

    /**
     * Get object from the storage, by key
     *
     * @param key
     */
    void get(T key);

    /**
     * This method checks, if storage contains specified key
     *
     * @param key
     * @return
     */
    boolean containsKey(T key);
}
