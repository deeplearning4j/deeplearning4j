package org.deeplearning4j.optimize.listeners.stats.storage;

/**
 * Created by Alex on 30/09/2016.
 */
public interface StatsStorageListener {

    void notifyNewSession(String sessionID);

    void notifyStaticInfo(String sessionID, String workerID);

    void notifyStatusUpdate(String sessionID, String workerID, long timestamp);

}
