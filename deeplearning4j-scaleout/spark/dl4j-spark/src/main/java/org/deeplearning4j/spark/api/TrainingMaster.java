package org.deeplearning4j.spark.api;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by Alex on 14/06/2016.
 */
public interface TrainingMaster<R extends TrainingResult, W extends TrainingWorker<R>> {

    W getWorkerInstance(SparkDl4jMultiLayer network);

    void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData);

//    void processResults(SparkDl4jMultiLayer network, JavaRDD<R> results);

    /**
     * Set whether the training statistics should be collected. Training statistics may include things like per-epoch run times,
     * time spent waiting for data, etc.
     * <p>
     * These statistics are primarily used for debugging and optimization, in order to gain some insight into what aspects
     * of network training are taking the most time.
     *
     * @param collectTrainingStats If true: collecting training statistics will be
     */
    void setCollectTrainingStats(boolean collectTrainingStats);

    /**
     * Get the current setting for collectTrainingStats
     */
    boolean getIsCollectTrainingStats();

    /**
     * Return the training statistics. Note that this may return null, unless setCollectTrainingStats has been set first
     *
     * @return Training statistics
     */
    SparkTrainingStats getTrainingStats();

}
