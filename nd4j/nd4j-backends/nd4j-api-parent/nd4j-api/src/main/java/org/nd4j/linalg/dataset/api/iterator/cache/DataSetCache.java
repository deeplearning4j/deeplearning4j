package org.nd4j.linalg.dataset.api.iterator.cache;

import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by anton on 7/16/16.
 */
public interface DataSetCache {
    /**
     * Check is given namespace has complete cache of the data set
     * @param namespace
     * @return true if namespace is fully cached
     */
    boolean isComplete(String namespace);

    /**
     * Sets the flag indicating whether given namespace is fully cached
     * @param namespace
     * @param value
     */
    void setComplete(String namespace, boolean value);

    DataSet get(String key);

    void put(String key, DataSet dataSet);

    boolean contains(String key);
}
