package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;

/**
 * MetaOp is special op, that contains multiple ops
 *
 * @author raver119@gmail.com
 */
public interface MetaOp extends GridOp {
    /**
     *
     * @return
     */
    Op getFirstOp();

    Op getSecondOp();

    OpDescriptor getFirstOpDescriptor();

    OpDescriptor getSecondOpDescriptor();

    void setFirstPointers(GridPointers pointers);

    void setSecondPointers(GridPointers pointers);
}
