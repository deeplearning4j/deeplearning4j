package org.deeplearning4j.spark.api;

/**
 * Enumeration that is used for specifying the behaviour of repartitioning in {@link org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster}
 * (and possibly elsewhere.
 *
 * "Never" and "Always" repartition options are as expected; the "NumPartitionsExecutorsDiffers" will repartition data if and only
 * if the number of partitions is not equal to the number of executors
 *
 * @author Alex Black
 */
public enum Repartition {
    Never,
    Always,
    NumPartitionsExecutorsDiffers
}
