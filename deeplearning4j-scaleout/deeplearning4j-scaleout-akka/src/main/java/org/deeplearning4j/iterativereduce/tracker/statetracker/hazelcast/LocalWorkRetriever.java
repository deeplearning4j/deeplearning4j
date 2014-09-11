package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;

import org.deeplearning4j.iterativereduce.tracker.statetracker.WorkRetriever;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Local worker retriever
 * @author Adam Gibson
 */
public class LocalWorkRetriever implements WorkRetriever {

    private Map<String,String> workerData = new ConcurrentHashMap<>();


    /**
     * Clears the worker
     *
     * @param worker the worker to clear
     */
    @Override
    public void clear(String worker) {
        workerData.remove(worker);
    }

    /**
     * The collection of workers that are saved
     *
     * @return the collection of workers that have data saved
     */
    @Override
    public Collection<String> workers() {
        return new HashSet<>(workerData.keySet());
    }

    /**
     * Loads the data applyTransformToDestination
     *
     * @param worker the worker to load for
     * @return the data for the given worker or null
     */
    @Override
    public DataSet load(String worker) {
        File f = workerData.get(worker) != null ? new File(workerData.get(worker)) : null;
        if(f == null || !f.exists())
            return null;
        DataSet d = SerializationUtils.readObject(f);
        workerData.remove(f);
        f.delete();
        return d;
    }

    /**
     * Saves the data applyTransformToDestination for a given worker
     *
     * @param worker the worker to save data for
     * @param data   the data to save
     */
    @Override
    public void save(String worker, DataSet data) {
        File f = new File(worker + "-work");
        SerializationUtils.saveObject(data,f);
        workerData.put(worker,f.getAbsolutePath());


    }
}
