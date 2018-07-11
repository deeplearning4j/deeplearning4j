package org.deeplearning4j.spark.api;

import org.apache.spark.api.java.JavaRDD;

import java.io.Serializable;

/**
 * Repartitioner interface: controls how data should be repartitioned before training.
 * Currently used only in SharedTrainingMaster
 *
 * @author Alex Black
 */
public interface Repartitioner extends Serializable {

    <T> JavaRDD<T> repartition(JavaRDD<T> input, int minObjectsPerPartition, int numExecutors);

}
