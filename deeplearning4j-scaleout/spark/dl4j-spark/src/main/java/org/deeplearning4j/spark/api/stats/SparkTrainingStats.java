package org.deeplearning4j.spark.api.stats;

import java.io.Serializable;
import java.util.Set;

/**
 * SparkTrainingStats is an interface that is used for accessing training statistics, for multiple {@link org.deeplearning4j.spark.api.TrainingMaster}
 * implementations.
 *
 * The idea is that for debugging purposes, we want to collect a number of statistics related to the training. However, these
 * statistics will vary, depending on which the type of training we are doing. Specifically, both the keys (number of stats)
 * and their actual values (types/classes) can vary.
 *
 * The interface here operates essentially as a {@code Map<String,Object>}
 *
 * @author Alex Black
 */
public interface SparkTrainingStats extends Serializable {

    Set<String> getKeySet();

    Object getValue(String key);

    /**
     * Combine the two training stats instances
     * @param other
     */
    void addOtherTrainingStats(SparkTrainingStats other);

}
