package org.deeplearning4j.spark.impl.common.updater;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;

/** Simple function to add ad Updater to an UpdaterAggregator */
public class UpdaterElementCombiner implements Function2<UpdaterAggregator,Updater,UpdaterAggregator> {
    @Override
    public UpdaterAggregator call(UpdaterAggregator updaterAggregator, Updater updater) throws Exception {
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
