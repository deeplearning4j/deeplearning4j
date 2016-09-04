package org.nd4j.linalg.api.ops.grid;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ops.Op;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class OpDescriptor {
    private Op op;
    private int[] dimensions;

    public OpDescriptor(Op op) {
        this(op, null);
    }
}
