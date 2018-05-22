package org.nd4j.linalg.api.ops.impl.grid;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;

import java.util.List;

/**
 * Simple GridOp that operates on arbitrary number of Ops, that have no relations between them.
 *
 * @author raver119@gmail.com
 */
public class FreeGridOp extends BaseGridOp {

    public FreeGridOp() {

    }

    public FreeGridOp(INDArray x, INDArray y) {
        super(x, y);
    }

    public FreeGridOp(Op... ops) {
        super(ops);
    }

    public FreeGridOp(List<Op> ops) {
        super(ops);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "grid_free";
    }

     @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
