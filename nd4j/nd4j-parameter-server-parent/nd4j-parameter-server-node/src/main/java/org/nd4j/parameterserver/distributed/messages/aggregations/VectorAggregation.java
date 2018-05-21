package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.complete.VectorCompleteMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class VectorAggregation extends BaseAggregation {

    protected VectorAggregation() {
        super();
    }

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
        if (clipboard.isTracking(this.originatorId, this.getTaskId())) {
            clipboard.pin(this);

            if (clipboard.isReady(this.originatorId, taskId)) {
                VoidAggregation aggregation = clipboard.unpin(this.originatorId, taskId);

                // FIXME: probably there's better solution, then "screw-and-forget" one
                if (aggregation == null)
                    return;

                VectorCompleteMessage msg = new VectorCompleteMessage(taskId, aggregation.getAccumulatedResult());
                msg.setOriginatorId(aggregation.getOriginatorId());
                transport.sendMessage(msg);
            }
        }
    }
}
