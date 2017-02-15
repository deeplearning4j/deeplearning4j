package org.nd4j.parameterserver.distributed.training.chains;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.messages.Chain;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class CbowChain implements Chain {
    protected long originatorId;
    protected long taskId;
    protected long frameId;

    protected CbowRequestMessage cbowRequest;
    protected DotAggregation dotAggregation;

    public CbowChain(@NonNull CbowRequestMessage message) {
        this(message.getTaskId(), message);
    }

    public CbowChain(long taskId, @NonNull CbowRequestMessage message) {
        this.taskId = taskId;
        this.originatorId = message.getOriginatorId();
        this.frameId = message.getFrameId();
    }

    @Override
    public void addElement(VoidMessage message) {
        if (message instanceof CbowRequestMessage) {

            cbowRequest = (CbowRequestMessage) message;
        } else if (message instanceof DotAggregation) {

            dotAggregation = (DotAggregation) message;
        } else
            throw new ND4JIllegalStateException("Unknown message passed: " + message.getClass().getCanonicalName());
    }
}
