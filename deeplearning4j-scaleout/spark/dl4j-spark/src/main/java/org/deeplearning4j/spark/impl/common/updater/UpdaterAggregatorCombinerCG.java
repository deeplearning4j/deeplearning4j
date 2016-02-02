package org.deeplearning4j.spark.impl.common.updater;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;

/**Simple function to combine ComputationGrapuhUpdater.Aggregators
 */
public class UpdaterAggregatorCombinerCG implements Function2<ComputationGraphUpdater.Aggregator,ComputationGraphUpdater.Aggregator,
            ComputationGraphUpdater.Aggregator> {
    @Override
    public ComputationGraphUpdater.Aggregator call(ComputationGraphUpdater.Aggregator updaterAggregator, ComputationGraphUpdater.Aggregator updaterAggregator2) throws Exception {
        if(updaterAggregator == null) return updaterAggregator2;
        else if(updaterAggregator2 == null) return updaterAggregator;

        updaterAggregator.merge(updaterAggregator2);
        return updaterAggregator;
    }
}
