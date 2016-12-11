package org.nd4j.parameterserver.distributed.messages.aggregations;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This interface describes special case for distributed environment: aggregation of partial responses received from different shards
 *
 * @author raver119@gmail.com
 */
public interface VoidAggregation {

    short getShardIndex();

    short getAggregationType();

    void accumulateAggregation(VoidAggregation aggregation);

    INDArray getPayload();

    INDArray getAccumulatedResult();
}
