package org.nd4j.linalg.dataset.api.iterator.cache;

import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by anton on 7/16/16.
 */
public interface DataSetCache {
    DataSet get(String key);

    void put(String key, DataSet dataSet);

    boolean contains(String key);
}
