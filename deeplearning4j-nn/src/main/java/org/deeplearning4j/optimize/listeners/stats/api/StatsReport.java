package org.deeplearning4j.optimize.listeners.stats.api;

import org.deeplearning4j.berkeley.Pair;

import java.util.List;
import java.util.Map;

/**
 * StatsReport: An interface for storing and serializing update information (such as scores, parameter histograms etc) for
 * use in the {@link org.deeplearning4j.optimize.listeners.stats.StatsListener}
 *
 * @author Alex Black
 */
public interface StatsReport {

    /**
     * Report the current iteration number
     */
    void reportIterationCount(int iterationCount);

    /**
     * Report the current time for this report, in epoch (ms) format
     */
    void reportTime(long currentTime);

    /**
     * Get the report time (ms, epoch format)
     */
    long getTime();

    /**
     * Report the number of milliseconds required to calculate all of the stats. This is effectively the
     * amount of listener overhead
     */
    void reportStatsCollectionDurationMS(int statsCollectionDurationMS);

    /**
     * Report model score at the current iteration
     */
    void reportScore(double currentScore);

    /**
     * Get the score at the current iteration
     */
    double getScore();

    //--- Performance and System Stats ---

    /**
     * Report the memory stats at this iteration
     *
     * @param jvmCurrentBytes     Current bytes used by the JVM
     * @param jvmMaxBytes         Max bytes usable by the JVM (heap)
     * @param offHeapCurrentBytes Current off-heap bytes used
     * @param offHeapMaxBytes     Maximum off-heap bytes
     * @param deviceCurrentBytes  Current bytes used by each device (GPU, etc). May be null if no devices are present
     * @param deviceMaxBytes      Maximum bytes for each device (GPU, etc). May be null if no devices are present
     */
    void reportMemoryUse(long jvmCurrentBytes, long jvmMaxBytes, long offHeapCurrentBytes, long offHeapMaxBytes,
                         long[] deviceCurrentBytes, long[] deviceMaxBytes);

    /**
     * Get JVM memory - current bytes used
     */
    long getJvmCurrentBytes();

    /**
     * Get JVM memory - max available bytes
     */
    long getJvmMaxBytes();

    /**
     * Get off-heap memory - current bytes used
     */
    long getOffHeapCurrentBytes();

    /**
     * Get off-heap memory - max available bytes
     */
    long getOffHeapMaxBytes();

    /**
     * Get device (GPU, etc) current bytes - may be null if no compute devices are present in the system
     */
    long[] getDeviceCurrentBytes();

    /**
     * Get device (GPU, etc) maximum bytes - may be null if no compute devices are present in the system
     */
    long[] getDeviceMaxBytes();

    /**
     * Report the performance stats (since the last report)
     *
     * @param totalRuntimeMs       Overall runtime since initialization
     * @param totalExamples        Total examples processed since initialization
     * @param totalMinibatches     Total number of minibatches (iterations) since initialization
     * @param examplesPerSecond    Examples per second since last report
     * @param minibatchesPerSecond Minibatches per second since last report
     */
    void reportPerformance(long totalRuntimeMs, long totalExamples, long totalMinibatches,
                           double examplesPerSecond, double minibatchesPerSecond);

    /**
     * Get the total runtime since listener/model initialization
     */
    long getTotalRuntimeMs();

    /**
     * Get total number of examples that have been processed since initialization
     */
    long getTotalExamples();

    /**
     * Get the total number of minibatches that have been processed since initialization
     */
    long getTotalMinibatches();

    /**
     * Get examples per second since the last report
     */
    double getExamplesPerSecond();

    /**
     * Get the number of minibatches per second, since the last report
     */
    double getMinibatchesPerSecond();

    /**
     * Report Garbage collection stats
     *
     * @param gcName       Garbage collector name
     * @param deltaGCCount Change in the total number of garbage collections, since last report
     * @param deltaGCTime  Change in the amount of time (milliseconds) for garbage collection, since last report
     */
    void reportGarbageCollection(String gcName, int deltaGCCount, int deltaGCTime);

    /**
     * Get the garbage collection stats: Pair contains GC name and the delta count/time values
     */
    List<Pair<String, int[]>> getGarbageCollectionStats();

    //--- Histograms ---

    /**
     * Report histograms for all parameters, for a given {@link StatsType}
     *
     * @param statsType StatsType: Parameters, Updates, Activations
     * @param histogram Histogram values for all parameters
     */
    void reportHistograms(StatsType statsType, Map<String, Histogram> histogram);

    /**
     * Get the histograms for all parameters, for a given StatsType (Parameters/Updates/Activations)
     *
     * @param statsType Stats type (Params/updatse/activations) to get histograms for
     * @return Histograms by parameter name, or null if not available
     */
    Map<String, Histogram> getHistograms(StatsType statsType);

    //--- Summary Stats: Mean, Variance, Mean Magnitudes ---

    /**
     * Report the mean values for each parameter, the given StatsType (Parameters/Updates/Activations)
     *
     * @param statsType Stats type to report
     * @param mean      Map of mean values, by parameter
     */
    void reportMean(StatsType statsType, Map<String, Double> mean);

    /**
     * Get the mean values for each parameter for the given StatsType (Parameters/Updates/Activations)
     *
     * @param statsType Stats type to get mean values for
     * @return Map of mean values by parameter
     */
    Map<String, Double> getMean(StatsType statsType);

    /**
     * Report the standard deviation values for each parameter for the given StatsType (Parameters/Updates/Activations)
     *
     * @param statsType Stats type to report std. dev values for
     * @param stdev     Map of std dev values by parameter
     */
    void reportStdev(StatsType statsType, Map<String, Double> stdev);

    /**
     * Get the standard deviation values for each parameter for the given StatsType (Parameters/Updates/Activations)
     *
     * @param statsType Stats type to get std dev values for
     * @return Map of stdev values by parameter
     */
    Map<String, Double> getStdev(StatsType statsType);

    /**
     * Report the mean magnitude values for each parameter for the given StatsType (Parameters/Updates/Activations)
     *
     * @param statsType      Stats type to report mean magnitude values for
     * @param meanMagnitudes Map of mean magnitude values by parameter
     */
    void reportMeanMagnitudes(StatsType statsType, Map<String, Double> meanMagnitudes);

    /**
     * Serialize the StatsReport to a byte[] for storage etc
     */
    byte[] toByteArray();

    /**
     * Deserialize the StatsReport contents from a given byte[]
     *
     * @param bytes Bytes with content
     */
    void fromByteArray(byte[] bytes);
}
