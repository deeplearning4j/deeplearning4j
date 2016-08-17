package org.nd4j.linalg.api.ops.impl.meta;

import org.nd4j.linalg.api.ops.*;

/**
 * This MetaOp covers case, when Op A and Op B are both using linear memory access
 *
 * @author raver119@gmail.com
 */
public class LinearMetaOp extends BaseMetaOp {

    public LinearMetaOp(ScalarOp opA, TransformOp opB) {
        super(opA, opB);
    }

    public LinearMetaOp(TransformOp opA, TransformOp opB) {
        super(opA, opB);
    }

    public LinearMetaOp(TransformOp opA, ScalarOp opB) {
        super(opA, opB);
    }

    public LinearMetaOp(ScalarOp opA, ScalarOp opB) {
        super(opA, opB);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "meta_linear";
    }
}
