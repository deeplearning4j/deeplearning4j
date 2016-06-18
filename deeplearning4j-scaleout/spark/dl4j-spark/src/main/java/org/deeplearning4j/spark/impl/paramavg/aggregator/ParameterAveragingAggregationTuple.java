package org.deeplearning4j.spark.impl.paramavg.aggregator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by Alex on 15/06/2016.
 */
@AllArgsConstructor @Data
public class ParameterAveragingAggregationTuple implements Serializable {
    private final INDArray parametersSum;
    private final UpdaterAggregator updaterAggregator;
    private final double scoreSum;
    private final int aggregationsCount;
    private final SparkTrainingStats sparkTrainingStats;
}
