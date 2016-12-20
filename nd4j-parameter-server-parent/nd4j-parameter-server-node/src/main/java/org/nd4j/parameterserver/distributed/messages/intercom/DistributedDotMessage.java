package org.nd4j.parameterserver.distributed.messages.intercom;

import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;

/**
 * @author raver119@gmail.com
 */
public class DistributedDotMessage extends BaseVoidMessage {

    public DistributedDotMessage() {
        messageType = 22;
    }


    public DistributedDotMessage(int rowA, int rowB) {
        this();
    }

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    @Override
    public void processMessage() {

    }
}
