package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedIntroductionMessage;

/**
 * This message will be sent by each shard, during meeting
 *
 * @author raver119@gmail.com
 */
public class IntroductionRequestMessage extends BaseVoidMessage  implements RequestMessage {
    private String ip;
    private int port;

    protected IntroductionRequestMessage() {
        super();
    }

    protected IntroductionRequestMessage(@NonNull String ip, int port) {
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void processMessage() {
        // redistribute this message over network
        DistributedIntroductionMessage dim = new DistributedIntroductionMessage(ip, port);

        dim.extractContext(this);
        dim.processMessage();

        if (configuration.getNumberOfShards() > 1)
            transport.sendMessageToAllShards(dim);
    }
}
