package org.nd4j.linalg.api.ops.impl.meta;

import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;

/**
 * @author raver119@gmail.com
 */
public class ReducedMetaOp extends BaseMetaOp {

    public ReducedMetaOp(ScalarOp opA, Accumulation opB) {
        super(opA, opB);
    }

    public ReducedMetaOp(TransformOp opA, Accumulation opB) {
        super(opA, opB);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String name() {
        return "meta_reduce";
    }
}
