package org.nd4j.parameterserver.distributed.training.chains;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.messages.Chain;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;

/**
 * Chain implementation for SkipGram
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class SkipGramChain implements Chain {

    protected long originatorId;
    protected long taskId;
    protected long frameId;

    protected SkipGramRequestMessage requestMessage;
    protected DotAggregation dotAggregation;

    public SkipGramChain(long originatorId, long taskId, long frameId) {
        this.taskId = taskId;
        this.frameId = frameId;
        this.originatorId = originatorId;
    }

    public SkipGramChain(@NonNull SkipGramRequestMessage message) {
        this(message.getTaskId(), message);
    }

    public SkipGramChain(long taskId, @NonNull SkipGramRequestMessage message) {
        this(message.getOriginatorId(), taskId, message.getFrameId());
        addElement(message);
    }

    @Override
    public long getTaskId() {
        return taskId;
    }

    @Override
    public void addElement(VoidMessage message) {
        if (message instanceof SkipGramRequestMessage) {
            requestMessage = (SkipGramRequestMessage) message;

        } else if (message instanceof DotAggregation) {
            dotAggregation = (DotAggregation) message;

        } else
            throw new ND4JIllegalStateException(
                            "Unknown message received: [" + message.getClass().getCanonicalName() + "]");
    }
}
