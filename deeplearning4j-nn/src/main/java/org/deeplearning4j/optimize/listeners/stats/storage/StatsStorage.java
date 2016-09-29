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



    Object getStaticInfo(String sessionID, String workerID);

    List<String> listWorkerIDsForSession(String sessionID);

    Object getLatestState(String sessionID, String workerID);

}
