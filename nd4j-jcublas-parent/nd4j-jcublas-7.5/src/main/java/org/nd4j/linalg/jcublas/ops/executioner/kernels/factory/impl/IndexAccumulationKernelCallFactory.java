package org.nd4j.linalg.jcublas.ops.executioner.kernels.factory.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCallFactory;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.impl.IndexAccumulationKernelCall;

/**
 * Index accumulation
 * kernel calls
 *
 * @author Adam Gibson
 */
public class IndexAccumulationKernelCallFactory implements GpuKernelCallFactory {
    @Override
    public GpuKernelCall create(Op op, Object... otherArgs) {
        return new IndexAccumulationKernelCall(op,(int[]) otherArgs[0],(INDArray) otherArgs[1]);
    }
}
