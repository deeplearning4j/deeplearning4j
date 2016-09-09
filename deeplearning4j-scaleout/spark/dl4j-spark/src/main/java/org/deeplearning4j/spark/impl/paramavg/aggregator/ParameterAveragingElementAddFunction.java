package org.deeplearning4j.spark.impl.paramavg.aggregator;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Add function for parameter averaging
 *
 * @author Alex Black
 */
public class ParameterAveragingElementAddFunction implements Function2<ParameterAveragingAggregationTuple, ParameterAveragingTrainingResult, ParameterAveragingAggregationTuple> {

    @Override
    public ParameterAveragingAggregationTuple call(ParameterAveragingAggregationTuple tuple, ParameterAveragingTrainingResult result) throws Exception {
        if (tuple == null) {
            return new ParameterAveragingAggregationTuple(result.getParameters(), result.getUpdaterState(), result.getScore(), 1, result.getSparkTrainingStats());
        }

        INDArray params = tuple.getParametersSum().addi(result.getParameters());
        INDArray updaterStateSum;
        if (tuple.getUpdaterStateSum() == null) {
            updaterStateSum = result.getUpdaterState();
        } else {
            updaterStateSum = tuple.getUpdaterStateSum();
            if(result.getUpdaterState() != null) updaterStateSum.addi(result.getUpdaterState());
        }

        double scoreSum = tuple.getScoreSum() + result.getScore();
        SparkTrainingStats stats = tuple.getSparkTrainingStats();
        if (result.getSparkTrainingStats() != null) {
            if (stats == null) stats = result.getSparkTrainingStats();
            else stats.addOtherTrainingStats(result.getSparkTrainingStats());
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        return new ParameterAveragingAggregationTuple(params, updaterStateSum, scoreSum, tuple.getAggregationsCount() + 1, stats);
    }
}
