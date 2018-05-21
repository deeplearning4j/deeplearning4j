package org.nd4j.parameterserver.distributed.messages.complete;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public class VectorCompleteMessage extends BaseCompleteMessage {

    protected VectorCompleteMessage() {
        super();
    }

    public VectorCompleteMessage(long taskId, @NonNull INDArray vector) {
        this();
        this.taskId = taskId;
        this.payload = vector.isView() ? vector.dup(vector.ordering()) : vector;
    }
}
