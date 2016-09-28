package org.deeplearning4j.optimize.listeners.stats;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsListenerReceiver {

    StatsReport newStatsReport();

    void postResult(StatsReport statsReport);

    StatsListenerConfiguration getCurrentConfiguration();




}
