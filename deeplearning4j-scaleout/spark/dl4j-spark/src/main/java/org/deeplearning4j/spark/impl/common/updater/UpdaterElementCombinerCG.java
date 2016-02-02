package org.deeplearning4j.spark.impl.common.updater;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;

/** Simple function to add an Updater to an UpdaterAggregator */
public class UpdaterElementCombinerCG implements Function2<ComputationGraphUpdater.Aggregator,
        ComputationGraphUpdater,ComputationGraphUpdater.Aggregator> {
    @Override
    public ComputationGraphUpdater.Aggregator call(ComputationGraphUpdater.Aggregator updaterAggregator, ComputationGraphUpdater updater) throws Exception {
        if(updaterAggregator == null && updater == null) return null;

        if(updaterAggregator == null){
            //updater is not null, but updaterAggregator is
            return updater.getAggregator(true);
        }
        if(updater == null){
            //updater is null, but aggregator is not -> no op
            return updaterAggregator;
        }

        //both are non-null
        updaterAggregator.aggregate(updater);
        return updaterAggregator;
    }
}
