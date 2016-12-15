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

    public AssignMessage(int index, double value) {
        super(6);
        this.index = index;
        this.value = value;
    }
}
