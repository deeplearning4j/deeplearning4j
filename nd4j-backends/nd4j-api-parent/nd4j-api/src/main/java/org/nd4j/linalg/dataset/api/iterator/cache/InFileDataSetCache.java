package org.nd4j.linalg.dataset.api.iterator.cache;

import com.sun.org.apache.xpath.internal.operations.Bool;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.nio.file.Path;

/**
 * Created by anton on 7/18/16.
 */
public class InFileDataSetCache implements DataSetCache {
    private File cacheDirectory;

    public InFileDataSetCache(File cacheDirectory) {
        if (cacheDirectory.exists() && !cacheDirectory.isDirectory()) {
            throw new IllegalArgumentException("can't use path " + cacheDirectory + " as file cache directory " +
                    "because it already exists, but is not a directory");
        }
        this.cacheDirectory = cacheDirectory;
    }

    public InFileDataSetCache(Path cacheDirectory) {
        this(cacheDirectory.toFile());
    }

    public InFileDataSetCache(String cacheDirectory) {
        this(new File(cacheDirectory));
    }

    private File resolveKey(String key) {
        String filename = key.replaceAll("[^a-zA-Z0-9.-]", "_");
        return new File(cacheDirectory, filename);
    }

    @Override
    public DataSet get(String key) {
        File file = resolveKey(key);

        if (!file.exists()) {
            return null;
        } else if (!file.isFile()) {
            throw new IllegalStateException("ERROR: cannot read DataSet: cache path " + file + " is not a file");
        } else {
            DataSet ds = new DataSet();
            ds.load(file);
            return ds;
        }
    }

    @Override
    public void put(String key, DataSet dataSet) {
        File file = resolveKey(key);

        if (file.exists()) {
            throw new IllegalStateException("ERROR: cannot write DataSet: cache path " + file + " already exists");
        } else {
            File parentDir = file.getParentFile();
            if (!parentDir.exists()) {
                if (!parentDir.mkdirs()) {
                    throw new IllegalStateException("ERROR: cannot create parent directory: " + parentDir);
                }
            }

            dataSet.save(file);
        }
    }

    @Override
    public boolean contains(String key) {
        File file = resolveKey(key);

        Boolean exists = file.exists();
        if (exists && !file.isFile()) {
            throw new IllegalStateException("ERROR: DataSet cache path " + file + " exists but is not a file");
        }

        return exists;
    }
}
