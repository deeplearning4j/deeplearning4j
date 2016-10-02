package org.deeplearning4j.optimize.listeners.stats.api;

import java.util.Map;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsReport {

    void reportIterationCount(int iterationCount);

    //TODO: probably want to use NTP
    void reportTime(long currentTime);

    long getTime();

    void reportStatsCollectionDurationMS(long statsCollectionDurationMS);

    void reportScore(double currentScore);

    //--- Performance and System Stats ---
    void reportMemoryUse(long jvmCurrentBytes, long jvmMaxBytes, long offHeapCurrentBytes, long offHeapMaxBytes,
                         long[] deviceCurrentBytes, long[] deviceMaxBytes );

    void reportPerformance(long totalRuntimeMs, long totalExamples, long totalMinibatches,
                           double examplesPerSecond, double minibatchesPerSecond);

    void reportGarbageCollection(String gcName, int deltaReportTimeMs, int deltaGCCount, int deltaGCTime);

    //--- Histograms ---
    void reportHistograms(StatsType statsType, Map<String, Histogram> histogram);


    //--- Summary Stats: Mean, Variance, Mean Magnitudes ---
    void reportMean(StatsType statsType, Map<String,Double> mean);

    void reportStdev(StatsType statsType, Map<String,Double> stdev);

    void reportMeanMagnitudes(StatsType statsType, Map<String,Double> meanMagnitudes);

    /**
     * Serialize the StatsReport to a byte[] for storage etc
     */
    byte[] toByteArray();

    void fromByteArray(byte[] bytes);
}
