package org.nd4j.parameterserver.distributed.messages.requests;

import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedShutdownMessage;

/**
 * This message
 * @author raver119@gmail.com
 */
public class ShutdownRequestMessage extends BaseVoidMessage implements RequestMessage {

    public ShutdownRequestMessage() {
        super(8);
    }

    @Override
    public void processMessage() {
        DistributedShutdownMessage dsm = new DistributedShutdownMessage();
        transport.sendMessage(dsm);

        try {
            Thread.sleep(1000);
        } catch (Exception e) {
        }

        transport.shutdown();
        storage.shutdown();
    }
}
