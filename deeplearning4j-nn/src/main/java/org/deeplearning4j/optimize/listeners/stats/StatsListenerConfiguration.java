package org.deeplearning4j.optimize.listeners.stats;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsListenerConfiguration {

    int reportingFrequency();

    boolean useNTPTimeSource();

    //--- Performance and System Stats ---

    /**
     * Minibatches/second, examples/second, total time, total batches, total examples
     */
    boolean collectPerformanceStats();

    boolean collectMemoryStats();

    boolean collectGarbageCollectionStats();

    boolean collectDataSetMetaData();

    //--- General ---

    boolean collectScore();

    boolean collectLearningRates();

    //--- Histograms ---
    boolean collectHistograms(StatsType type);

    int numHistogramBins(StatsType type);

    //--- Summary Stats: Mean, Variance, Mean Magnitudes ---

    boolean collectMean(StatsType type);

    boolean collectStdev(StatsType type);

    boolean collectMeanMagnitudes(StatsType type);

}
