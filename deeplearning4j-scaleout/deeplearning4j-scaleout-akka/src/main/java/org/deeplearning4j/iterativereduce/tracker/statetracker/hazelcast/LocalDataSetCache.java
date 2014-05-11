package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.iterativereduce.tracker.statetracker.DataSetCache;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;

/**
 * Stores the data set on local disk
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
        return SerializationUtils.readObject(new File(dataSetPath));
    }

    @Override
    public void set(DataSet d) {
        SerializationUtils.saveObject(d,new File(dataSetPath));

    }
}
