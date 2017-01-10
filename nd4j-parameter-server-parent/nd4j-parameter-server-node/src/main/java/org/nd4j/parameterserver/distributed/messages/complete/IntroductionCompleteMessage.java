package org.nd4j.parameterserver.distributed.messages.complete;

/**
 * @author raver119@gmail.com
 */
public class IntroductionCompleteMessage extends BaseCompleteMessage {

    protected IntroductionCompleteMessage() {
        super(19);
    }

    public IntroductionCompleteMessage(long taskId) {
        this();
        this.taskId = taskId;
    }
}
