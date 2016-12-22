package org.nd4j.parameterserver.distributed.messages;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.logic.WordVectorStorage;
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
@NoArgsConstructor
@Slf4j
public class VectorRequestMessage extends BaseVoidMessage {

    protected int rowIndex;

    public VectorRequestMessage(int rowIndex) {
        super(7);
        this.rowIndex = rowIndex;
    }

    /**
     * This message is possible to get only as Shard
     */
    @Override
    public void processMessage() {
        log.debug("Got request for rowIndex: {}", rowIndex);

        VectorAggregation aggregation = new VectorAggregation(rowIndex, (short) configuration.getNumberOfShards(), getShardIndex(), storage.getArray(WordVectorStorage.SYN_0).getRow(rowIndex).dup());

        clipboard.pin(aggregation);

        DistributedVectorMessage dvm = new DistributedVectorMessage(WordVectorStorage.SYN_0, rowIndex);
        transport.sendMessageToAllShards(dvm);
    }
}
