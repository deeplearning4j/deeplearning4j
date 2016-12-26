package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedAssignMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class AssignRequestMessage extends BaseVoidMessage {

    protected Integer key;

    protected int rowIdx;

    // assign part
    protected INDArray payload;
    protected Number value;


    protected AssignRequestMessage() {
        super(8);
    }


    public AssignRequestMessage(@NonNull Integer key, @NonNull INDArray array) {
        this();
        this.key = key;
        this.payload = array;
    }

    public AssignRequestMessage(@NonNull Integer key, @NonNull Number value, int rowIdx) {
        this();
        this.key = key;
        this.value = value;
        this.rowIdx = rowIdx;
    }

    @Override
    public void processMessage() {
        if (payload == null) {
            DistributedAssignMessage dam = new DistributedAssignMessage(key, rowIdx, value.doubleValue());
            transport.sendMessage(dam);
        } else {
            DistributedAssignMessage dam = new DistributedAssignMessage(key, payload);
            transport.sendMessage(dam);
        }
    }
}
