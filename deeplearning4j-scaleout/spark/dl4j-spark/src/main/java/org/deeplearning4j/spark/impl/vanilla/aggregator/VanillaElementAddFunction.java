package org.deeplearning4j.spark.impl.vanilla.aggregator;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.deeplearning4j.spark.impl.vanilla.VanillaTrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 15/06/2016.
 */
public class VanillaElementAddFunction implements Function2<VanillaAggregationTuple, VanillaTrainingResult, VanillaAggregationTuple> {

    @Override
    public VanillaAggregationTuple call(VanillaAggregationTuple tuple, VanillaTrainingResult result) throws Exception {

        if(tuple == null){
            return new VanillaAggregationTuple(result.getParameters(), result.getUpdater().getAggregator(true), result.getScore(), 1);
        }

        INDArray params = tuple.getParametersSum().addi(result.getParameters());
        UpdaterAggregator aggregator;
        if(tuple.getUpdaterAggregator() == null){
            if(result.getUpdater() == null) aggregator = null;
            else aggregator = result.getUpdater().getAggregator(true);
        } else {
            aggregator = tuple.getUpdaterAggregator();
            if(result.getUpdater() == null) aggregator.aggregate(result.getUpdater());
        }

        double scoreSum = tuple.getScoreSum() + result.getScore();
        return new VanillaAggregationTuple(params,aggregator,scoreSum,tuple.getAggregationsCount()+1);
    }
}
