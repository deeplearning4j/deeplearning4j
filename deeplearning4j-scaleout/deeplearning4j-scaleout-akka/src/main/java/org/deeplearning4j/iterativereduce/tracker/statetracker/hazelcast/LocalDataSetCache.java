package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;

import org.deeplearning4j.iterativereduce.tracker.statetracker.DataSetCache;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;

/**
 * Stores the data applyTransformToDestination on local disk
 * @author Adam Gibson
 */
public class LocalDataSetCache implements DataSetCache {


    private String dataSetPath;

    public LocalDataSetCache(String dataSetPath) {
        this.dataSetPath = dataSetPath;
    }


    public LocalDataSetCache() {
        this("cacheddataset");
    }

    @Override
    public DataSet get() {
        File f = new File(dataSetPath);
        if(f.exists())
            return SerializationUtils.readObject(f);
        return null;
    }

    @Override
    public void set(DataSet d) {
        SerializationUtils.saveObject(d,new File(dataSetPath));

    }
}
