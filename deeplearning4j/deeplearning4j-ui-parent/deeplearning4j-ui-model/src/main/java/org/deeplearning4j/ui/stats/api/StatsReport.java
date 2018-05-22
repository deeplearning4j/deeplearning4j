package org.deeplearning4j.ui.stats.api;

import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * StatsReport: An interface for storing and serializing update information (such as scores, parameter histograms etc) for
 * use in the {@link StatsListener}
 *
 * @author Alex Black
 */
public interface StatsReport extends Persistable {

    void reportIDs(String sessionID, String typeID, String workerID, long timestamp);

    /**
     * Report the current iteration number
     */
    void reportIterationCount(int iterationCount);

    /**
     * Get the current iteration number
     */
    int getIterationCount();

    /**
     * Report the number of milliseconds required to calculate all of the stats. This is effectively the
     * amount of listener overhead
     */
    void reportStatsCollectionDurationMS(int statsCollectionDurationMS);

    /**
     * Get the number of millisecons required to calculate al of the stats. This is effectively the amount of
     * listener overhead.
     */
    int getStatsCollectionDurationMs();

    /**
     * Report model score at the current iteration
     */
    void reportScore(double currentScore);

    /**
     * Get the score at the current iteration
     */
    double getScore();

    /**
     * Report the learning rates by parameter
     */
    void reportLearningRates(Map<String, Double> learningRatesByParam);

    /**
     * Get the learning rates by parameter
     */
    Map<String, Double> getLearningRates();


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
    void reportPerformance(long totalRuntimeMs, long totalExamples, long totalMinibatches, double examplesPerSecond,
                    double minibatchesPerSecond);

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
     * Report any metadata for the DataSet
     *
     * @param dataSetMetaData MetaData for the DataSet
     * @param metaDataClass   Class of the metadata. Can be later retieved using {@link #getDataSetMetaDataClassName()}
     */
    void reportDataSetMetaData(List<Serializable> dataSetMetaData, Class<?> metaDataClass);

    /**
     * Report any metadata for the DataSet
     *
     * @param dataSetMetaData MetaData for the DataSet
     * @param metaDataClass   Class of the metadata. Can be later retieved using {@link #getDataSetMetaDataClassName()}
     */
    void reportDataSetMetaData(List<Serializable> dataSetMetaData, String metaDataClass);

    /**
     * Get the mean magnitude values for each parameter for the given StatsType (Parameters/Updates/Activations)
     *
     * @param statsType Stats type to get mean magnitude values for
     * @return Map of mean magnitude values by parameter
     */
    Map<String, Double> getMeanMagnitudes(StatsType statsType);

    /**
     * Get the DataSet metadata, if any (null otherwise).
     * Note: due to serialization issues, this may in principle throw an unchecked exception related
     * to class availability, serialization etc.
     *
     * @return List of DataSet metadata, if any.
     */
    List<Serializable> getDataSetMetaData();

    /**
     * Get the class
     *
     * @return
     */
    String getDataSetMetaDataClassName();

    /**
     * Return whether the score is present (has been reported)
     */
    boolean hasScore();

    /**
     * Return whether the learning rates are present (have been reported)
     */
    boolean hasLearningRates();

    /**
     * Return whether memory use has been reported
     */
    boolean hasMemoryUse();

    /**
     * Return whether performance stats (total time, total examples etc) have been reported
     */
    boolean hasPerformance();

    /**
     * Return whether garbage collection information has been reported
     */
    boolean hasGarbageCollection();

    /**
     * Return whether histograms have been reported, for the given stats type (Parameters, Updates, Activations)
     *
     * @param statsType Stats type
     */
    boolean hasHistograms(StatsType statsType);

    /**
     * Return whether the summary stats (mean, standard deviation, mean magnitudes) have been reported for the
     * given stats type (Parameters, Updates, Activations)
     *
     * @param statsType   stats type (Parameters, Updates, Activations)
     * @param summaryType Summary statistic type (mean, stdev, mean magnitude)
     */
    boolean hasSummaryStats(StatsType statsType, SummaryType summaryType);


    /**
     * Return whether any DataSet metadata is present or not
     *
     * @return True if DataSet metadata is present
     */
    boolean hasDataSetMetaData();
}
