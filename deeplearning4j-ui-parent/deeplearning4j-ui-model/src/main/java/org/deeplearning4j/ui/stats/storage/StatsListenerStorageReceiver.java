package org.deeplearning4j.ui.stats.storage;

import org.deeplearning4j.ui.stats.api.*;
import org.deeplearning4j.ui.storage.StatsStorage;

import java.io.IOException;

/**
 * A simple StatsListenerReceiver that takes the provided stats information, and passes them on to the storage layer.
 * Uses a fixed configuration for both initialization and stats collection.
 *
 * @author Alex Black
 */
public class StatsListenerStorageReceiver implements StatsListenerReceiver {

    private StatsInitializationConfiguration initConfig;
    private StatsListenerConfiguration statsConfig;
    private StatsStorage statsStorage;

    public StatsListenerStorageReceiver(StatsInitializationConfiguration initConfig, StatsListenerConfiguration statsConfig,
                                        StatsStorage statsStorage ){
        this.initConfig = initConfig;
        this.statsConfig = statsConfig;
        this.statsStorage = statsStorage;
    }

    @Override
    public StatsInitializationReport newInitializationReport() {
        //TODO
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public StatsInitializationConfiguration getInitializationConfiguration() {
        return initConfig;
    }

    @Override
    public void postInitializationReport(StatsInitializationReport initializationReport) throws IOException {
        String sessionID = null;    //TODO
        String workerID = null;     //TODO
        //TODO proper types for the init report...
        statsStorage.putStaticInfo(sessionID, workerID, initializationReport.toByteArray());
    }

    @Override
    public StatsReport newStatsReport() {
        //TODO
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void postStatsReport(StatsReport statsReport) {
        String sessionID = null;    //TODO
        String workerID = null;     //TODO
        //TODO proper types for the state report...
        long time = statsReport.getTime();
        statsStorage.putUpdate(sessionID, workerID, time, statsReport.toByteArray());
    }

    @Override
    public StatsListenerConfiguration getCurrentConfiguration() {
        return statsConfig;
    }
}
