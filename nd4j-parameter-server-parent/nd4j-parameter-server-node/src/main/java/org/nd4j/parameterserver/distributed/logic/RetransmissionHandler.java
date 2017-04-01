package org.nd4j.parameterserver.distributed.logic;

import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public interface RetransmissionHandler {
    public enum TransmissionStatus {
        MESSAGE_SENT, NOT_CONNECTED, BACKPRESSURE, ADMIN_ACTION,
    }

    void init(VoidConfiguration configuration, Transport transport);

    void handleMessage(TrainingMessage message);

    void onBackPressure();

    static TransmissionStatus getTransmissionStatus(long resp) {
        if (resp >= 0) {
            return TransmissionStatus.MESSAGE_SENT;
        } else if (resp == -1) {
            return TransmissionStatus.NOT_CONNECTED;
        } else if (resp == -2) {
            return TransmissionStatus.BACKPRESSURE;
        } else if (resp == -3) {
            return TransmissionStatus.ADMIN_ACTION;
        } else {
            throw new ND4JIllegalStateException("Unknown response from Aeron received: [" + resp + "]");
        }
    }


}
