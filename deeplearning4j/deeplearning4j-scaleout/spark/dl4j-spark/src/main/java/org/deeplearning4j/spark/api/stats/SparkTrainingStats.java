package org.deeplearning4j.spark.api.stats;

import org.apache.spark.SparkContext;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingWorkerStats;
import org.deeplearning4j.spark.stats.EventStats;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Set;

/**
 * SparkTrainingStats is an interface that is used for accessing training statistics, for multiple {@link org.deeplearning4j.spark.api.TrainingMaster}
 * implementations.
 * <p>
 * The idea is that for debugging purposes, we want to collect a number of statistics related to the training. However, these
 * statistics will vary, depending on which the type of training we are doing. Specifically, both the keys (number/names of stats)
 * and their actual values (types/classes) can vary.
 * <p>
 * The interface here operates essentially as a {@code Map<String,Object>}. Note however that SparkTrainingStats instances
 * may be nested: for example a {@link ParameterAveragingTrainingMasterStats} may have a
 * {@link CommonSparkTrainingStats} instance which may in turn have a {@link ParameterAveragingTrainingWorkerStats}
 * instance.
 *
 * @author Alex Black
 */
public interface SparkTrainingStats extends Serializable {

    /**
     * Default indentation for {@link #statsAsString()}
     */
    int PRINT_INDENT = 55;

    /**
     * Default formatter used for {@link #statsAsString()}
     */
    String DEFAULT_PRINT_FORMAT = "%-" + PRINT_INDENT + "s";

    /**
     * @return Set of keys that can be used with {@link #getValue(String)}
     */
    Set<String> getKeySet();

    /**
     * Get the statistic value for this key
     *
     * @param key Key to get the value for
     * @return Statistic for this key, or an exception if key is invalid
     */
    List<EventStats> getValue(String key);

    /**
     * Return a short (display) name for the given key.
     *
     * @param key    Key
     * @return Short/display name for key
     */
    String getShortNameForKey(String key);

    /**
     * When plotting statistics, we don't necessarily want to plot everything.
     * For example, some statistics/measurements are made up multiple smaller components; it does not always make sense
     * to plot both the larger stat, and the components that make it up
     *
     * @param key Key to check for default plotting behaviour
     * @return Whether the specified key should be included in plots by default or not
     */
    boolean defaultIncludeInPlots(String key);

    /**
     * Combine the two training stats instances. Usually, the two objects must be of the same type
     *
     * @param other Other training stats to return
     */
    void addOtherTrainingStats(SparkTrainingStats other);

    /**
     * Return the nested training stats - if any.
     *
     * @return The nested stats, if present/applicable, or null otherwise
     */
    SparkTrainingStats getNestedTrainingStats();

    /**
     * Get a String representation of the stats. This functionality is implemented as a separate method (as opposed to toString())
     * as the resulting String can be very large.<br>
     *
     * <b>NOTE</b>: The String representation typically includes only duration information. To get full statistics (including
     * machine IDs, etc) use {@link #getValue(String)} or export full data via {@link #exportStatFiles(String, SparkContext)}
     *
     * @return A String representation of the training statistics
     */
    String statsAsString();


    /**
     * Export the stats as a collection of files. Stats are comma-delimited (CSV) with 1 header line
     *
     * @param outputPath    Base directory to write files to
     */
    void exportStatFiles(String outputPath, SparkContext sc) throws IOException;
}
