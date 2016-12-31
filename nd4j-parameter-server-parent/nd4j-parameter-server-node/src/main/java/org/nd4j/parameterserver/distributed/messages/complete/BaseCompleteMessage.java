package org.nd4j.parameterserver.distributed.messages.complete;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.MeaningfulMessage;

/**
 * This message contains information about finished computations for specific batch, being sent earlier
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public abstract class BaseCompleteMessage extends BaseVoidMessage implements MeaningfulMessage {

    protected INDArray payload;

    public BaseCompleteMessage() {
        super(10);
    }

    public BaseCompleteMessage(int messageType) {
        super(messageType);
    }


    @Override
    public void processMessage() {
        // no-op
    }
}
