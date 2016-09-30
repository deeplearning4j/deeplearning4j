package org.deeplearning4j.optimize.listeners.stats.storage;

import java.util.List;

/**
 * Created by Alex on 29/09/2016.
 */
public interface StatsStorage {


    /**
     * Get a list of all sessions stored by this storage backend
     */
    List<StatsSession> listSessions();

    boolean sessionExists(String sessionID);


    byte[] getStaticInfo(String sessionID, String workerID);

    List<String> listWorkerIDsForSession(String sessionID);

    int getNumStateRecordsFor(String sessionID);

    int getNumStateRecordsFor(String sessionID, String workerID);

    byte[] getLatestState(String sessionID, String workerID);

    byte[] getState(String sessionID, String workerID, long timestamp);

    List<byte[]> getLatestStateAllWorkers(String sessionID);

    List<byte[]> getAllStatesAfter(String sessionID, String workerID, long timestamp);


    // ----- Store new info -----

    void putStaticInfo(String sessionID, String workerID, byte[] staticInfo);

    void putState(String sessionID, String workerID, long timestamp, byte[] state);


    // ----- Listeners -----

    void registerStatsStorageListener(StatsStorageListener listener);

    void deregisterStatsStorageListener(StatsStorageListener listener);

    void removeAllListeners();

}
