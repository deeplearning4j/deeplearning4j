package org.deeplearning4j.optimize.listeners.stats;

/**
 * Created by Alex on 28/09/2016.
 */
public interface StatsListenerConfiguration {

    boolean reportingFrequency();

    boolean collectScore();

    boolean useNTPTimeSource();

    boolean collectHistogramParameters();

    boolean collectHistogramUpdates();

    boolean collectHistogramActivations();

    int numHistogramBins();

    boolean collectLearningRates();

    boolean collectMeanMagnitudesParameters();

    boolean collectMeanMagnitudesUpdates();

    boolean collectMeanMagnitudesActivations();

    /**
     * Minibatches/second, examples/second, total time, total batches, total examples
     * @return
     */
    boolean collectPerformanceStats();

    boolean collectMemoryStats();

    //Machine/JVM ID, backend, hardware??
    boolean collectSystemStats();

    boolean collectDataSetMetaData();


}
