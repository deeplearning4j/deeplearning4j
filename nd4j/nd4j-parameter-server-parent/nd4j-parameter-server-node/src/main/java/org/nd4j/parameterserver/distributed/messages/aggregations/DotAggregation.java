package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class DotAggregation extends BaseAggregation {

    protected DotAggregation() {
        super();
    }

    public DotAggregation(long taskId, short aggregationWidth, short shardIndex, INDArray scalar) {
        super(taskId, aggregationWidth, shardIndex);

        this.payload = scalar;
        addToChunks(payload);
    }

    @Override
    public INDArray getAccumulatedResult() {
        INDArray stack = super.getAccumulatedResult();

        if (aggregationWidth == 1)
            return stack;

        if (stack.isRowVector()) {
            return Nd4j.scalar(stack.sumNumber().doubleValue());
        } else {
            return stack.sum(1);
        }
    }

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    @Override
    public void processMessage() {
        // since our computations are symmetric - we aggregate dot everywhere
        if (chunks == null) {
            chunks = new TreeMap<>();
            chunksCounter = new AtomicInteger(1);
            addToChunks(payload);
        }

        clipboard.pin(this);

        //log.info("sI_{} dot aggregation received", transport.getShardIndex());

        if (clipboard.isReady(this.getOriginatorId(), this.getTaskId())) {
            trainer.aggregationFinished(clipboard.unpin(this.getOriginatorId(), this.taskId));
        }
    }
}
