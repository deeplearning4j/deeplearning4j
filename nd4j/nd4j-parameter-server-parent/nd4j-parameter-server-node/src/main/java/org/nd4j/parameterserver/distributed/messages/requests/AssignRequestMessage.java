package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedAssignMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class AssignRequestMessage extends BaseVoidMessage implements RequestMessage {

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
        this.payload = array.isView() ? array.dup(array.ordering()) : array;
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
            dam.extractContext(this);
            dam.processMessage();
            transport.sendMessageToAllShards(dam);
        } else {
            DistributedAssignMessage dam = new DistributedAssignMessage(key, payload);
            dam.extractContext(this);
            dam.processMessage();
            transport.sendMessageToAllShards(dam);
        }
    }
}
