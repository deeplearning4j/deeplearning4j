package org.deeplearning4j.optimize.listeners.stats;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsListenerReceiver {

    // --- Initialization: Done once, on first iteration ---

    StatsInitializationReport newInitializationReport();

    StatsInitConfiguration getInitializationConfiguration();

    void postInitializationReport(StatsInitializationReport initializationReport);


    // --- Stats: Collected and reported periodically (based on configuration)---
    StatsReport newStatsReport();

    void postStatsReport(StatsReport statsReport);

    StatsListenerConfiguration getCurrentConfiguration();

}
