package org.nd4j.linalg.jcublas.ops.executioner.kernels.factory.impl;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCallFactory;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.impl.ScalarKernelCall;

/**
 * Creates a scalar kernel call
 * @author Adam Gibson
 */
public class ScalarKernelCallFactory implements GpuKernelCallFactory {
    @Override
    public GpuKernelCall create(Op op, Object... otherArgs) {
        return new ScalarKernelCall(op);
    }
}
