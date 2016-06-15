package org.deeplearning4j.spark.impl.vanilla.aggregator;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 15/06/2016.
 */
public class VanillaElementCombineFunction implements Function2<VanillaAggregationTuple, VanillaAggregationTuple, VanillaAggregationTuple> {
    @Override
    public VanillaAggregationTuple call(VanillaAggregationTuple v1, VanillaAggregationTuple v2) throws Exception {
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

        double scoreSum = v1.getScoreSum() + v2.getScoreSum();
        int aggregationCount = v1.getAggregationsCount() + v2.getAggregationsCount();

        return new VanillaAggregationTuple(newParams, combinedAggregator, scoreSum, aggregationCount);
    }
}
