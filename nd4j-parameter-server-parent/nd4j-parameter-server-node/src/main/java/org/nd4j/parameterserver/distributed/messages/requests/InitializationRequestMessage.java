package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Builder;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.InitializationAggregation;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedInitializationMessage;

/**
 * This method propagates storage/weights initialization over distributed Shards
 *
 * @author raver119@gmail.com
 */
@Builder
public class InitializationRequestMessage extends BaseVoidMessage {
    protected int vectorLength;
    protected int numWords;
    protected long seed;
    protected boolean useHs;
    protected boolean useNeg;
    protected int columnsPerShard;

    protected InitializationRequestMessage() {
        super(4);
        taskId = -119L;
    }

    public InitializationRequestMessage(int vectorLength, int numWords, long seed, boolean useHs, boolean useNeg, int columnsPerShard) {
        this();
        this.vectorLength = vectorLength;
        this.numWords = numWords;
        this.seed = seed;
        this.useHs = useHs;
        this.useNeg = useNeg;
        this.columnsPerShard = columnsPerShard;
    }


    @Override
    public void processMessage() {
        DistributedInitializationMessage dim = new DistributedInitializationMessage(vectorLength, numWords, seed, useHs, useNeg, columnsPerShard);

        dim.extractContext(this);
        dim.processMessage();

        // FIXME: i don't like this hack :(
        clipboard.pin(new InitializationAggregation((short) configuration.getNumberOfShards(), transport.getShardIndex()));

        transport.sendMessageToAllShards(dim);
    }

    @Override
    public boolean isBlockingMessage() {
        return true;
    }
}
