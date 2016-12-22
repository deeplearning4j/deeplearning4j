package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class VectorAggregation extends BaseAggregation {

    public VectorAggregation(long taskId, short aggregationWidth, short shardIndex, INDArray array) {
        super(taskId, aggregationWidth, shardIndex);
        this.payload = array.isView() ? array.dup(array.ordering()) : array;

        addToChunks(payload);
    }

    /**
     * Vector aggregations are saved only by Shards started aggregation process. All other Shards are ignoring this meesage
     */
    @Override
    public void processMessage() {
        if (clipboard.isTracking(this.getTaskId())) {
            clipboard.pin(this);
        } else {
            log.info("Skipping vectors. Shard: {}; ", shardIndex);
        }
    }
}
