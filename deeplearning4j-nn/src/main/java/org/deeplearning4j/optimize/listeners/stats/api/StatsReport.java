package org.deeplearning4j.optimize.listeners.stats.api;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.optimize.listeners.stats.StatsType;
import org.nd4j.linalg.api.ndarray.INDArray;

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
    void reportMemoryUse(Long jvmCurrentBytes, Long jvmMaxBytes, Long offHeapCurrentBytes, Long offHeapMaxBytes,
                         long[] gpuCurrentBytes, long[] gpuMaxBytes );

    void reportPerformance(long totalRuntimeMs, long totalExamples, long totalMinibatches,
                           double examplesPerSecond, double minibatchesPerSecond);

    void reportGarbageCollection(String gcName, long deltaReportTime, long deltaGCCount, long deltaGCTime);

    //--- Histograms ---
    void reportHistograms(StatsType statsType, Map<String,Pair<INDArray,int[]>> histogram);


    //--- Summary Stats: Mean, Variance, Mean Magnitudes ---
    void reportMean(StatsType statsType, Map<String,Double> mean);

    void reportStdev(StatsType statsType, Map<String,Double> stdev);

    void reportMeanMagnitudes(StatsType statsType, Map<String,Double> meanMagnitudes);

    /**
     * Serialize the StatsReport to a byte[] for storage etc
     */
    byte[] toByteArray();
}
