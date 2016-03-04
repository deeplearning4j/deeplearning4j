package org.nd4j.linalg.jcublas.ops.executioner.kernels.factory.impl;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCallFactory;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.impl.AccumulationKernelCall;

/**
 * Accumulation kernel call factory
 *
 * @author Adam Gibson
 */
public class AccumulationKernelCallFactory implements GpuKernelCallFactory {
    @Override
    public GpuKernelCall create(Op op, Object... otherArgs) {
        return new AccumulationKernelCall(op,(int[]) otherArgs[0]);
    }
}
