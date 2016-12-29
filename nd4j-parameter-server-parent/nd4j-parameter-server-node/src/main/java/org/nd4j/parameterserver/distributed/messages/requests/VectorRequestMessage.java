package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.logic.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
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
public class VectorRequestMessage extends BaseVoidMessage {

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
        log.debug("Got request for rowIndex: {}", rowIndex);

        VectorAggregation aggregation = new VectorAggregation(rowIndex, (short) configuration.getNumberOfShards(), getShardIndex(), storage.getArray(key).getRow(rowIndex).dup());

        clipboard.pin(aggregation);

        DistributedVectorMessage dvm = new DistributedVectorMessage(key, rowIndex);
        transport.sendMessageToAllShards(dvm);
    }
}
