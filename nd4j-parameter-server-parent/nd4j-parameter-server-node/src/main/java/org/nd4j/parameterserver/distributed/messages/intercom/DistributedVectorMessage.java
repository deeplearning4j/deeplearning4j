package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.Data;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;

/**
 * @author raver119@gmail.com
 */
@Data
public class DistributedVectorMessage extends BaseVoidMessage{
    protected int rowIndex;

    public DistributedVectorMessage() {
        messageType = 20;
    }

    public DistributedVectorMessage(int rowIndex){
        this();
        this.rowIndex = rowIndex;
    }

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    @Override
    public void processMessage() {
        // TODO: to be implemented
    }
}
