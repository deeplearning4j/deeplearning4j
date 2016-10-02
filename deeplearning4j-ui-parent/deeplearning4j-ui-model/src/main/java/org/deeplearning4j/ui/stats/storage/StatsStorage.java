package org.deeplearning4j.ui.stats.storage;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;

import java.util.List;

/**
 * Created by Alex on 29/09/2016.
 */
public interface StatsStorage {


    /**
     * Get a list of all sessions stored by this storage backend
     */
    List<String> listSessionIDs();

    boolean sessionExists(String sessionID);


    byte[] getStaticInfo(String sessionID, String workerID);

    List<String> listWorkerIDsForSession(String sessionID);

    int getNumUpdateRecordsFor(String sessionID);

    int getNumUpdateRecordsFor(String sessionID, String workerID);

    Pair<Long, byte[]> getLatestUpdate(String sessionID, String workerID);

    byte[] getUpdate(String sessionID, String workerID, long timestamp);

    List<UpdateRecord> getLatestUpdateAllWorkers(String sessionID);

    List<UpdateRecord> getAllUpdatesAfter(String sessionID, String workerID, long timestamp);


    // ----- Store new info -----

    void putStaticInfo(String sessionID, String workerID, byte[] staticInfo);

    void putUpdate(String sessionID, String workerID, long timestamp, byte[] update);


    // ----- Listeners -----

    void registerStatsStorageListener(StatsStorageListener listener);

    void deregisterStatsStorageListener(StatsStorageListener listener);

    void removeAllListeners();

    List<StatsStorageListener> getListeners();

    @AllArgsConstructor @Data
    public static class UpdateRecord {
        private final String sessionID;
        private final String workerID;
        private final long timestamp;
        private final byte[] record;
    }
}
