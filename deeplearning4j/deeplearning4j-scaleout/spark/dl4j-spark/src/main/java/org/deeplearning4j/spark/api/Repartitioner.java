package org.deeplearning4j.spark.api;

import org.apache.spark.api.java.JavaRDD;

import java.io.Serializable;

public interface Repartitioner extends Serializable {

    <T> JavaRDD<T> repartition(JavaRDD<T> input, int minObjectsPerPartition, int numExecutors);

}
