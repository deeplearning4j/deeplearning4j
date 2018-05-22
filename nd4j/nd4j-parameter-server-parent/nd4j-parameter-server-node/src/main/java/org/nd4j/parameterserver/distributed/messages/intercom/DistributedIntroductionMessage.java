package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;

/**
 * @author raver119@gmail.com
 */
public class DistributedIntroductionMessage extends BaseVoidMessage implements DistributedMessage {
    private String ip;
    private int port;

    protected DistributedIntroductionMessage() {
        super();
    }

    public DistributedIntroductionMessage(@NonNull String ip, int port) {
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void processMessage() {
        transport.addClient(this.ip, this.port);
    }
}
