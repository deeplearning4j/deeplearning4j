package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class Cross extends DynamicCustomOp {

    @Override
    public String opName() {
        return "cross";
    }


    @Override
    public String tensorflowName() {
        return "Cross";
    }


}
