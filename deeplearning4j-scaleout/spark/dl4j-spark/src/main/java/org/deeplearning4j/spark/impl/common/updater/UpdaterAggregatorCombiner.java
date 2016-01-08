package org.deeplearning4j.spark.impl.common.updater;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;

/**Simple function to combine UpdaterAggregators
 */
public class UpdaterAggregatorCombiner implements Function2<UpdaterAggregator,UpdaterAggregator,UpdaterAggregator> {
    @Override
    public UpdaterAggregator call(UpdaterAggregator updaterAggregator, UpdaterAggregator updaterAggregator2) throws Exception {
        if(updaterAggregator == null) return updaterAggregator2;
        else if(updaterAggregator2 == null) return updaterAggregator;

        updaterAggregator.merge(updaterAggregator2);
        return updaterAggregator;
    }
}
