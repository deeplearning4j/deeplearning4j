package org.deeplearning4j.spark.api;

/**
 * Enumeration that is used for specifying the behaviour of repartitioning in {@link org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster}
 * (and possibly elsewhere.
 *
 * "Never" and "Always" repartition options are as expected; the "NumPartitionsWorkersDiffers" will repartition data if and only
 * if the number of partitions is not equal to the number of workers (total cores). Note however that even if the number of partitions
 * and number of workers differ, this does <i>not</i> guarantee that those partitions are balanced (in terms of number of
 * elements) in any way.
 *
 * @author Alex Black
 */
public enum Repartition {
    Never, Always, NumPartitionsWorkersDiffers
}
