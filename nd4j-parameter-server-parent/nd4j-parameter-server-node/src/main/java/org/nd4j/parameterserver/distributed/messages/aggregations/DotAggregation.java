package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
@Data
public class DotAggregation extends BaseAggregation{

    public DotAggregation(long taskId, short aggregationWidth, short shardIndex, INDArray scalar) {
        super(taskId, aggregationWidth, shardIndex);

        this.payload = scalar;
        addToChunks(payload);
    }

    @Override
    public INDArray getAccumulatedResult() {
        INDArray stack = super.getAccumulatedResult();

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
        // we just pin this message, because it's tracked anyway
        clipboard.pin(this);
    }
}
