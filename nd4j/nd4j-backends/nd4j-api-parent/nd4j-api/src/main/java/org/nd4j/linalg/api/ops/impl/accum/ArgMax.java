package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class ArgMax extends DynamicCustomOp {
    @Override
    public String opName() {
        return "argmax";
    }

    @Override
    public String tensorflowName() {
        return "ArgMax";
    }
}
