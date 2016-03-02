package org.nd4j.linalg.jcublas.ops.executioner.kernels.factory.impl;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCallFactory;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.impl.TransformKernelCall;

/**
 * Created by agibsonccc on 12/11/15.
 */
public class TransformKernelCallFactory implements GpuKernelCallFactory {
    @Override
    public GpuKernelCall create(Op op, Object... otherArgs) {
        return new TransformKernelCall(op);
    }
}
