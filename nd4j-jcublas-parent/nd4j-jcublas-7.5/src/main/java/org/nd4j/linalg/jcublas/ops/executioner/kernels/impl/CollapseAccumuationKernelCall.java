package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;

/**
 * Created by agibsonccc on 12/15/15.
 */
public class CollapseAccumuationKernelCall extends BaseGpuKernelCall {
    public CollapseAccumuationKernelCall(Op op) {
        super(op);
    }

    @Override
    public void createArgs() {

    }

    @Override
    public void createCudaConext() {

    }

    @Override
    public void createMetrics() {

    }
}
