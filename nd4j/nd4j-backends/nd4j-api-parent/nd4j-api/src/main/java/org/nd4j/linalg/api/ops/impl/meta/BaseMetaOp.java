package org.nd4j.linalg.api.ops.impl.meta;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.MetaOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;
import org.nd4j.linalg.api.ops.impl.grid.BaseGridOp;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseMetaOp extends BaseGridOp implements MetaOp {

    public BaseMetaOp() {

    }

    public BaseMetaOp(INDArray x, INDArray y) {
        super(x, y);
    }

    protected BaseMetaOp(Op opA, Op opB) {
        super(opA, opB);
    }

    @Override
    public OpDescriptor getFirstOpDescriptor() {
        return queuedOps.get(0);
    }

    @Override
    public OpDescriptor getSecondOpDescriptor() {
        return queuedOps.get(1);
    }

    protected BaseMetaOp(OpDescriptor opA, OpDescriptor opB) {
        super(opA, opB);
    }

    protected BaseMetaOp(GridPointers opA, GridPointers opB) {
        super(opA, opB);
    }

    public Op getFirstOp() {
        return getFirstOpDescriptor().getOp();
    }

    public Op getSecondOp() {
        return getSecondOpDescriptor().getOp();
    }

    @Override
    public void setFirstPointers(GridPointers pointers) {
        grid.set(0, pointers);
    }

    @Override
    public void setSecondPointers(GridPointers pointers) {
        grid.set(1, pointers);
    }
}
