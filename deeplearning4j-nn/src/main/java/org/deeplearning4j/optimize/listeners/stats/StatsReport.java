package org.deeplearning4j.optimize.listeners.stats;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsReport {

    void reportIterationCount(int iterationCount);

    //TODO: probably want to use NTP
    void reportTime(long currentTime);

    void reportScore(double currentScore);

    //--- Performance and System Stats ---

    void reportMemoryUse(Long jvmCurrentBytes, Long jvmMaxBytes, Long offHeapCurrentBytes, Long offHeapMaxBytes,
                         Long gpuCurrentBytes, Long gpuMaxBytes );

    void reportPerformance(double examplesPerSecond, double minibatchesPerSecond);

    //--- Histograms ---

    void reportHistogramParameter(Map<String,Pair<INDArray,int[]>> histogram);

    void reportHistogramUpdates(Map<String,Pair<INDArray,int[]>> histogram);

    void reportHistogramActivations(Map<String,Pair<INDArray,int[]>> histogram);


    //--- Summary Stats: Mean, Variance, Mean Magnitudes ---
    void reportMean(StatsType statsType, Map<String,Double> mean);

    void reportStdev(StatsType statsType, Map<String,Double> stdev);

    void reportMeanMagnitudes(StatsType statsType, Map<String,Double> meanMagnitudes);

}
