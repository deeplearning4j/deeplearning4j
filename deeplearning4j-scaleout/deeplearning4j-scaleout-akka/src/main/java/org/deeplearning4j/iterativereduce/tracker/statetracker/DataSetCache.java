package org.deeplearning4j.iterativereduce.tracker.statetracker;

import org.deeplearning4j.datasets.DataSet;

import java.io.Serializable;

/**
 * The data applyTransformToDestination cache is used for caching data sets to a storage mechanism.
 * This is then used to load and store data sets such that the storage does not necessarily
 * have to be in memory
 */
public interface DataSetCache extends Serializable {


    DataSet get();
    void set(DataSet d);

}
