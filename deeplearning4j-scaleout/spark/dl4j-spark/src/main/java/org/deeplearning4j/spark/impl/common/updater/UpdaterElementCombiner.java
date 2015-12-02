package org.deeplearning4j.spark.impl.common.updater;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;

/** Simple function to add ad Updater to an UpdaterAggregator */
public class UpdaterElementCombiner implements Function2<UpdaterAggregator,Updater,UpdaterAggregator> {
    @Override
    public UpdaterAggregator call(UpdaterAggregator updaterAggregator, Updater updater) throws Exception {
        updaterAggregator.aggregate(updater);
        return updaterAggregator;
    }
}
