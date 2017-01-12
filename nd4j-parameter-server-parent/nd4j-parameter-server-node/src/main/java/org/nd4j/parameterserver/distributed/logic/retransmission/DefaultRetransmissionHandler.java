package org.nd4j.parameterserver.distributed.logic.retransmission;

import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.RetransmissionHandler;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public class DefaultRetransmissionHandler implements RetransmissionHandler {
    @Override
    public void init(VoidConfiguration configuration, Transport transport) {

    }

    @Override
    public void handleMessage(TrainingMessage message) {

    }
}
