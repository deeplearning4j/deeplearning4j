package org.nd4j.linalg.api.ops.impl.controlflow.compat;

import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;


public class StopGradient extends BaseDynamicTransformOp {
    @Override
    public String opName() {
        return "stop_gradient";
    }

    @Override
    public String tensorflowName() {
        return "StopGradient";
    }
}
