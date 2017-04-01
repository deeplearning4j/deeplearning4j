package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.logic.storage.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.VectorAggregation;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedVectorMessage;

/**
 * This message requests full weights vector for specified index
 *
 * Client -> Shard version
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class VectorRequestMessage extends BaseVoidMessage implements RequestMessage {

    protected Integer key;
    protected int rowIndex;

    protected VectorRequestMessage() {
        super(7);
    }

    public VectorRequestMessage(int rowIndex) {
        this(WordVectorStorage.SYN_0, rowIndex);
    }

    public VectorRequestMessage(@NonNull Integer key, int rowIndex) {
        this();
        this.rowIndex = rowIndex;

        // FIXME: this is temporary, should be changed
        this.taskId = rowIndex;
        this.key = key;
    }

    /**
     * This message is possible to get only as Shard
     */
    @Override
    public void processMessage() {
        VectorAggregation aggregation = new VectorAggregation(rowIndex, (short) voidConfiguration.getNumberOfShards(),
                        getShardIndex(), storage.getArray(key).getRow(rowIndex).dup());
        aggregation.setOriginatorId(this.getOriginatorId());

        clipboard.pin(aggregation);

        DistributedVectorMessage dvm = new DistributedVectorMessage(key, rowIndex);
        dvm.setOriginatorId(this.originatorId);

        if (voidConfiguration.getNumberOfShards() > 1)
            transport.sendMessageToAllShards(dvm);
        else {
            aggregation.extractContext(this);
            aggregation.processMessage();
        }
    }

    @Override
    public boolean isBlockingMessage() {
        return true;
    }
}
