package org.deeplearning4j.spark.api;

import org.deeplearning4j.spark.api.stats.SparkTrainingStats;

/**
 * Created by Alex on 14/06/2016.
 */
public interface TrainingResult {

    void setStats(SparkTrainingStats sparkTrainingStats);
}
