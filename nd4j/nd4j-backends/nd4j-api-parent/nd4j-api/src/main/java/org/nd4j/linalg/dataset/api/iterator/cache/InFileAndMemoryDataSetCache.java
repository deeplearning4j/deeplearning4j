package org.nd4j.linalg.dataset.api.iterator.cache;

import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.nio.file.Path;

/**
 * Created by anton on 7/20/16.
 */
public class InFileAndMemoryDataSetCache implements DataSetCache {

    private InFileDataSetCache fileCache;
    private InMemoryDataSetCache memoryCache;

    public InFileAndMemoryDataSetCache(File cacheDirectory) {
        this.fileCache = new InFileDataSetCache(cacheDirectory);
        this.memoryCache = new InMemoryDataSetCache();
    }

    public InFileAndMemoryDataSetCache(Path cacheDirectory) {
        this(cacheDirectory.toFile());
    }

    public InFileAndMemoryDataSetCache(String cacheDirectory) {
        this(new File(cacheDirectory));
    }

    @Override
    public boolean isComplete(String namespace) {
        return fileCache.isComplete(namespace) || memoryCache.isComplete(namespace);
    }

    @Override
    public void setComplete(String namespace, boolean value) {
        fileCache.setComplete(namespace, value);
        memoryCache.setComplete(namespace, value);
    }

    @Override
    public DataSet get(String key) {
        DataSet dataSet = null;

        if (memoryCache.contains(key)) {
            dataSet = memoryCache.get(key);
            if (!fileCache.contains(key)) {
                fileCache.put(key, dataSet);
            }
        } else if (fileCache.contains(key)) {
            dataSet = fileCache.get(key);
            if (dataSet != null && !memoryCache.contains(key)) {
                memoryCache.put(key, dataSet);
            }
        }

        return dataSet;
    }

    @Override
    public void put(String key, DataSet dataSet) {
        fileCache.put(key, dataSet);
        memoryCache.put(key, dataSet);
    }

    @Override
    public boolean contains(String key) {
        return memoryCache.contains(key) || fileCache.contains(key);
    }
}
