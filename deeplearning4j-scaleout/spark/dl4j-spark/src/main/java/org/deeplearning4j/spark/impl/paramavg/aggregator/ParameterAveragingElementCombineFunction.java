package org.deeplearning4j.spark.impl.paramavg.aggregator;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Function used in ParameterAveraging TrainingMaster, for doing parameter averaging, and handling updaters
 *
 * @author Alex Black
 */
public class ParameterAveragingElementCombineFunction implements Function2<ParameterAveragingAggregationTuple, ParameterAveragingAggregationTuple, ParameterAveragingAggregationTuple> {
    @Override
    public ParameterAveragingAggregationTuple call(ParameterAveragingAggregationTuple v1, ParameterAveragingAggregationTuple v2) throws Exception {
        if(v1 == null) return v2;
        else if(v2 == null) return v1;

        INDArray newParams = v1.getParametersSum().addi(v2.getParametersSum());

        UpdaterAggregator updaterAggregator = v1.getUpdaterAggregator();
        UpdaterAggregator updaterAggregator2 = v2.getUpdaterAggregator();
        UpdaterAggregator combinedAggregator;
        if(updaterAggregator == null) combinedAggregator = updaterAggregator2;
        else if(updaterAggregator2 == null) combinedAggregator = updaterAggregator;
        else{
            updaterAggregator.merge(updaterAggregator2);
            combinedAggregator = updaterAggregator;
        }

        ComputationGraphUpdater.Aggregator uAGraph1 = v1.getUpdaterAggregatorGraph();
        ComputationGraphUpdater.Aggregator uaGraph2 = v2.getUpdaterAggregatorGraph();
        ComputationGraphUpdater.Aggregator uaGraphCombined;
        if(uAGraph1 == null) uaGraphCombined = uaGraph2;
        else if(uaGraph2 == null) uaGraphCombined = uAGraph1;
        else {
            uAGraph1.merge(uaGraph2);
            uaGraphCombined = uAGraph1;
        }

        double scoreSum = v1.getScoreSum() + v2.getScoreSum();
        int aggregationCount = v1.getAggregationsCount() + v2.getAggregationsCount();

        SparkTrainingStats stats = v1.getSparkTrainingStats();
        if(v2.getSparkTrainingStats() != null){
            if(stats == null) stats = v2.getSparkTrainingStats();
            else stats.addOtherTrainingStats(v2.getSparkTrainingStats());
        }

        return new ParameterAveragingAggregationTuple(newParams, combinedAggregator, uaGraphCombined, scoreSum, aggregationCount, stats);
    }
}
