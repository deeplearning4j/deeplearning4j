package org.deeplearning4j.spark.impl.paramavg.aggregator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Simple helper tuple used to execute parameter averaging
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data
public class ParameterAveragingAggregationTuple implements Serializable {
    private final INDArray parametersSum;
    private final INDArray updaterStateSum;
    private final double scoreSum;
    private final int aggregationsCount;
    private final SparkTrainingStats sparkTrainingStats;
}
