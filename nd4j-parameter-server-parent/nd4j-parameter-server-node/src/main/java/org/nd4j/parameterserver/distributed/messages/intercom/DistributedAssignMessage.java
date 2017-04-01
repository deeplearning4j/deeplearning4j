package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;

/**
 * Assign target row to specified value
 *
 * @author raver119@gmail.com
 */
@Data
public class DistributedAssignMessage extends BaseVoidMessage implements DistributedMessage {
    /**
     * The only use of this message is negTable sharing.
     */
    private int index;
    private double value;
    private Integer key;
    private INDArray payload;

    protected DistributedAssignMessage() {
        super();
    }

    public DistributedAssignMessage(@NonNull Integer key, int index, double value) {
        super(6);
        this.index = index;
        this.value = value;
        this.key = key;
    }

    public DistributedAssignMessage(@NonNull Integer key, INDArray payload) {
        super(6);
        this.key = key;
        this.payload = payload;
    }

    /**
     * This method assigns specific value to either specific row, or whole array.
     * Array is identified by key
     */
    @Override
    public void processMessage() {
        if (payload != null) {
            // we're assigning array
            if (storage.arrayExists(key) && storage.getArray(key).length() == payload.length())
                storage.getArray(key).assign(payload);
            else
                storage.setArray(key, payload);
        } else {
            // we're assigning number to row
            if (index >= 0) {
                if (storage.getArray(key) == null)
                    throw new RuntimeException("Init wasn't called before for key [" + key + "]");
                storage.getArray(key).getRow(index).assign(value);
            } else
                storage.getArray(key).assign(value);
        }
    }
}
