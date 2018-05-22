package org.nd4j.parameterserver.distributed.messages.complete;

/**
 * @author raver119@gmail.com
 */
public class FrameCompleteMessage extends BaseCompleteMessage {
    protected FrameCompleteMessage() {
        super(19);
    }

    public FrameCompleteMessage(long taskId) {
        this();
        this.taskId = taskId;
    }
}
