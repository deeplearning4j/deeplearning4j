package org.deeplearning4j.ui.stats.api;

import java.io.IOException;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsListenerReceiver {

    // --- Initialization: Done once, on first iteration ---

    StatsInitializationReport newInitializationReport();

    StatsInitializationConfiguration getInitializationConfiguration();

    void postInitializationReport(StatsInitializationReport initializationReport) throws IOException ;


    // --- Stats: Collected and reported periodically (based on configuration)---
    StatsReport newStatsReport();

    void postStatsReport(StatsReport statsReport) throws IOException;

    StatsUpdateConfiguration getCurrentConfiguration();



}
