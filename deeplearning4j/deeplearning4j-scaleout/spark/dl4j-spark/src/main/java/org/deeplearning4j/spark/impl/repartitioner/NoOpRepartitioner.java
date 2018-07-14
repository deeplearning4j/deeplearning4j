package org.deeplearning4j.spark.impl.repartitioner;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.api.Repartitioner;

/**
 * No-op repartitioner. Returns the input un-modified
 *
 * @author Alex Black
 */
public class NoOpRepartitioner implements Repartitioner {
    @Override
    public <T> JavaRDD<T> repartition(JavaRDD<T> input, int minObjectsPerPartition, int numExecutors) {
        return input;
    }
}
