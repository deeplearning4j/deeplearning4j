package org.deeplearning4j.iterativereduce.tracker.statetracker.datasetcache;

import com.hazelcast.core.EntryEvent;
import com.hazelcast.core.EntryListener;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IMap;
import org.deeplearning4j.iterativereduce.tracker.statetracker.DataSetCache;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;

/**
 * Stores the data  on local disk
 * @author Adam Gibson
 */
public class LocalDataSetCache implements DataSetCache {


    private String dataSetPath;
    private IMap<String,DataSet> distributedMap;
    public final static String DATA_SET_MAP = "datasetmap";

    public LocalDataSetCache(String dataSetPath,HazelcastInstance hazelcast) {
        this.dataSetPath = dataSetPath;
        distributedMap = hazelcast.getMap(DATA_SET_MAP);
        distributedMap.addEntryListener(new EntryListener<String, DataSet>() {
            @Override
            public void entryAdded(EntryEvent<String, DataSet> event) {
                SerializationUtils.saveObject(event.getValue(),new File(event.getKey()));
            }

            @Override
            public void entryRemoved(EntryEvent<String, DataSet> event) {
                File f = new File(event.getKey());
                if(f.exists())
                    f.delete();
            }

            @Override
            public void entryUpdated(EntryEvent<String, DataSet> event) {
                SerializationUtils.saveObject(event.getValue(),new File(event.getKey()));

            }

            @Override
            public void entryEvicted(EntryEvent<String, DataSet> event) {

            }
        },true);

    }

    public LocalDataSetCache(String dataSetPath) {
        this.dataSetPath = dataSetPath;
    }

    public LocalDataSetCache() {
        this("cacheddataset.ser");
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
