package org.deeplearning4j.ui.storage;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.ui.stats.storage.StatsStorageListener;

import java.io.IOException;
import java.util.List;

/**
 * A general-purpose stats storage mechanism, for storing stats information (mainly used for iteration listeners).
 * <p>
 * Key design ideas:
 * (a) Everything is stored as byte[]
 * (b) There are 3 types of things used to identify these arrays:
 * i.   SessionID: A unique identifier for a single session
 * ii.  WorkerID: A unique identifier for workers, within a session
 * iii. Timestamp: time at
 * For example, single machine training would have 1 session ID and 1 worker ID
 * Distributed training could have 1 session ID and multiple worker IDs
 * A hyperparameter optimization job could have multiple
 * (c) Two types of things are stored:
 * i.   Static info: i.e., reported once per session ID and worker ID
 * ii.  Updates: reported multiple times (generally periodically) per session ID and worker ID
 *
 * @author Alex Black
 */
public interface StatsStorage {


    /**
     * Get a list of all sessions stored by this storage backend
     */
    List<String> listSessionIDs();

    /**
     * Check if the specified session ID exists or not
     *
     * @param sessionID Session ID to check
     * @return true if session exists, false otherwise
     */
    boolean sessionExists(String sessionID);

    /**
     * Get the static info for the given session and worker IDs, or null if no such static info has been reported
     *
     * @param sessionID Session ID
     * @param workerID  worker ID
     * @return Static info, or null if none has been reported
     */
    byte[] getStaticInfo(String sessionID, String workerID);

    /**
     * For a given session ID, list all of the known worker IDs
     *
     * @param sessionID Session ID
     * @return List of worker IDs, or possibly null if session ID is unknown
     */
    List<String> listWorkerIDsForSession(String sessionID);

    /**
     * Return the number of update records for the given session ID (all workers)
     *
     * @param sessionID Session ID
     * @return number of update records
     */
    int getNumUpdateRecordsFor(String sessionID);

    /**
     * Return the number of update records for the given session ID and worker ID
     *
     * @param sessionID Session ID
     * @param workerID  Worker ID
     * @return number of update records
     */
    int getNumUpdateRecordsFor(String sessionID, String workerID);

    /**
     * Get the latest update record (i.e., update record with the largest timestamp value) for the specified
     * session and worker IDs
     *
     * @param sessionID session ID
     * @param workerID  worker ID
     * @return UpdateRecord containing the session/worker IDs, timestamp and content for the most recent update
     */
    UpdateRecord getLatestUpdate(String sessionID, String workerID);

    /**
     * Get the specified update (or null, if none exists for the given session/worker ids and timestamp)
     *
     * @param sessionID Session ID
     * @param workerID  Worker ID
     * @param timestamp Timestamp
     * @return Update
     */
    UpdateRecord getUpdate(String sessionID, String workerID, long timestamp);

    /**
     * Get the latest update for all workers, for the given session ID
     *
     * @param sessionID Session ID
     * @return List of updates for the given Session ID
     */
    List<UpdateRecord> getLatestUpdateAllWorkers(String sessionID);

    /**
     * Get all updates for the given session and worker ID, that occur after (not including) the given timestamp.
     * Results should be sorted by time.
     *
     * @param sessionID Session ID
     * @param workerID  Worker Id
     * @param timestamp Timestamp
     * @return List of records occurring after the given timestamp
     */
    List<UpdateRecord> getAllUpdatesAfter(String sessionID, String workerID, long timestamp);


    // ----- Store new info -----

    /**
     * Put a new static info record to the stats storage instance. If static info for the given session/worker IDs
     * already exists, this will be replaced.
     *
     * @param sessionID  Session ID
     * @param workerID   Worker ID
     * @param staticInfo Bytes to put
     */
    void putStaticInfo(String sessionID, String workerID, byte[] staticInfo) throws IOException;

    /**
     * Put a new update for the given session/worker/timestamp.
     *
     * @param sessionID Session ID
     * @param workerID  Worker ID
     * @param timestamp Timestamp for the update
     * @param update    Update to store
     */
    void putUpdate(String sessionID, String workerID, long timestamp, byte[] update);


    // ----- Listeners -----

    /**
     * Add a new StatsStorageListener. The given listener will called whenever a state change occurs for the stats
     * storage instance
     *
     * @param listener Listener to add
     */
    void registerStatsStorageListener(StatsStorageListener listener);

    /**
     * Remove the specified listener, if it is present.
     *
     * @param listener Listener to remove
     */
    void deregisterStatsStorageListener(StatsStorageListener listener);

    /**
     * Remove all listeners from the StatsStorage instance
     */
    void removeAllListeners();

    /**
     * Get a list (shallow copy) of all listeners currently present
     *
     * @return List of listeners
     */
    List<StatsStorageListener> getListeners();

    /**
     * UpdateRecord is a simple object that stores a session ID, worker ID, timestamp and byte[] record
     */
    @AllArgsConstructor
    @Data
    public static class UpdateRecord {
        private final String sessionID;
        private final String workerID;
        private final long timestamp;
        private final byte[] record;
    }
}
