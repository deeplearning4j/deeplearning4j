package org.nd4j.parameterserver.distributed.messages.intercom;

import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;

/**
 * @author raver119@gmail.com
 */
public class DistributedShutdownMessage extends BaseVoidMessage {

    public DistributedShutdownMessage() {
        super(28);
    }

    @Override
    public void processMessage() {

        transport.shutdown();
        storage.shutdown();
    }
}
