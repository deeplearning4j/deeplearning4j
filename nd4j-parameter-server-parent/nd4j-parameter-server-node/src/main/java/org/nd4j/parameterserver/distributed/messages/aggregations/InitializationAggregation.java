package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.complete.InitializationCompleteMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class InitializationAggregation extends BaseAggregation {

    protected InitializationAggregation() {
        super();
    }

    public InitializationAggregation(int aggregationWidth, int shardIndex) {
        this((short) aggregationWidth, (short) shardIndex);
    }

    public InitializationAggregation(short aggregationWidth, short shardIndex) {
        super(-119L, aggregationWidth, shardIndex);
        this.payload = Nd4j.scalar(1.0);
    }

    @Override
    public void processMessage() {
        //log.info("sI_{} received init aggregation", transport.getShardIndex());
        if (clipboard.isTracking(this.originatorId, taskId)) {
            clipboard.pin(this);

            if (clipboard.isReady(this.originatorId, taskId)) {
                InitializationAggregation aggregation =
                                (InitializationAggregation) clipboard.unpin(this.originatorId, taskId);

                InitializationCompleteMessage icm = new InitializationCompleteMessage(taskId);
                icm.setOriginatorId(aggregation.getOriginatorId());
                transport.sendMessage(icm);
            }
        }
    }
}
