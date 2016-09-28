package org.deeplearning4j.optimize.listeners.stats;

import java.util.Map;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsReport {

    void reportIterationCount(int iterationCount);

    //TODO: probably want to use NTP
    void reportTime(long currentTime);

    void reportScore(double currentScore);

    void reportMeanMagnitudesParameters(Map<String,Double> meanMagnitudesParameters);

    void reportMeanMagnitudesUpdates(Map<String,Double> meanMagnitudesUpdates);

    void reportMeanMagnitudesActivations(Map<String,Double> meanMagnitudesActivations);

}
