package org.deeplearning4j.spark.api;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by Alex on 14/06/2016.
 */
public interface TrainingMaster<R extends TrainingResult, W extends TrainingWorker<R>> {

    W getWorkerInstance(SparkDl4jMultiLayer network);

    void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData);

//    void processResults(SparkDl4jMultiLayer network, JavaRDD<R> results);

}
