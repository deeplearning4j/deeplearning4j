package org.nd4j.parameterserver.distributed.messages.complete;

/**
 * @author raver119@gmail.com
 */
public class InitializationCompleteMessage extends BaseCompleteMessage {

    protected InitializationCompleteMessage() {
        super(19);
    }

    public InitializationCompleteMessage(long taskId) {
        this();
        this.taskId = taskId;
    }
}
