package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Builder;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
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
        transport.sendMessageToAllShards(dim);
    }
}
