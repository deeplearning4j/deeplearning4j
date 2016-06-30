package org.deeplearning4j.spark.api;

/**
 * Created by Alex on 30/06/2016.
 */
public enum Repartition {

    Never,
    Always,
    NumPartitionsExecutorsDiffers
}
