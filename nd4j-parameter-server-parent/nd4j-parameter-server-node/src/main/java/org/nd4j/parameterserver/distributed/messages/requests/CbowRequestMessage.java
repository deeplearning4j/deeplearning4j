package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class CbowRequestMessage extends BaseVoidMessage implements TrainingMessage, RequestMessage {
    protected byte counter;
    protected double alpha;

    long frameId;

    @Override
    public void processMessage() {

    }

    @Override
    public boolean isJoinSupported() {
        return true;
    }

    @Override
    public void joinMessage(VoidMessage message) {
        // TODO: apply proper join handling here
        counter++;
    }
}
