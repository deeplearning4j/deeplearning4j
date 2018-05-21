package org.nd4j.parameterserver.distributed.logic.retransmission;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.RetransmissionHandler;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public class DefaultRetransmissionHandler implements RetransmissionHandler {
    private VoidConfiguration configuration;

    @Override
    public void init(@NonNull VoidConfiguration configuration, Transport transport) {
        this.configuration = configuration;
    }

    @Override
    public void onBackPressure() {
        try {
            Thread.sleep(2000);
        } catch (Exception e) {
        }
    }

    @Override
    public void handleMessage(TrainingMessage message) {

    }
}
