package org.nd4j.linalg.dataset.api.iterator.cache;

import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by anton on 7/16/16.
 */
public class InMemoryDataSetCache implements DataSetCache {

    private static final Logger log = LoggerFactory.getLogger(DataSetCache.class);

    private Map<String, byte[]> cache = new HashMap<>();

    @Override
    public DataSet get(String key) {

        if (!cache.containsKey(key)) {
            return null;
        }

        byte[] data = cache.get(key);

        ByteArrayInputStream is = new ByteArrayInputStream(data);

        DataSet ds = new DataSet();

        ds.load(is);

        return ds;
    }

    @Override
    public void put(String key, DataSet dataSet) {
        if (cache.containsKey(key)) {
            log.debug("evicting key %s from data set cache", key);
            cache.remove(key);
        }

        ByteArrayOutputStream os = new ByteArrayOutputStream();

        dataSet.save(os);

        cache.put(key, os.toByteArray());
    }

    @Override
    public boolean contains(String key) {
        return cache.containsKey(key);
    }
}
