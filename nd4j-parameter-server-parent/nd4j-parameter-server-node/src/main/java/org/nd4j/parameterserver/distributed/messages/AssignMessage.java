package org.nd4j.parameterserver.distributed.messages;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Assign target row to specified value
 *
 * @author raver119@gmail.com
 */
@Data
@Builder
@NoArgsConstructor
public class AssignMessage extends BaseVoidMessage {
    /**
     * The only use of this message is negTable sharing.
     */
    private int index;
    private double value;
    private String key;

    public AssignMessage(@NonNull String key, int index, double value) {
        super(6);
        this.index = index;
        this.value = value;
        this.key = key;
    }

    /**
     * This method assigns
     */
    @Override
    public void processMessage() {
        INDArray array = storage.getArray(key);
        if (array != null) {
            if (index < 0)
                array.assign(value);
            else
                array.getRow(index).assign(value);
        }

    }
}
