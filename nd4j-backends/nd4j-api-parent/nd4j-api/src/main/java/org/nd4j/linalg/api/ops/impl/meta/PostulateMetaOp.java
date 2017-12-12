package org.nd4j.linalg.api.ops.impl.meta;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;

import java.util.List;

/**
 * You're NOT supposed to directly call this op. Do it on your own risk, only if you're absolutely have to.
 *
 * @author raver119@gmail.com
 */
public class PostulateMetaOp extends BaseMetaOp {

    public PostulateMetaOp() {

    }

    public PostulateMetaOp(INDArray x, INDArray y) {
        super(x, y);
    }

    public PostulateMetaOp(ScalarOp opA, Accumulation opB) {
        super(opA, opB);
    }

    public PostulateMetaOp(TransformOp opA, Accumulation opB) {
        super(opA, opB);
    }

    public PostulateMetaOp(OpDescriptor opA, OpDescriptor opB) {
        super(opA, opB);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "meta_postulate";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
