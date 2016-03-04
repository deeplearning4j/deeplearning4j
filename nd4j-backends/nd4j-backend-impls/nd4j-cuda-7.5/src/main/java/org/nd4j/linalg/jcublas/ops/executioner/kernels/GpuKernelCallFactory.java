package org.nd4j.linalg.jcublas.ops.executioner.kernels;

import org.nd4j.linalg.api.ops.Op;

/**
 * Created by agibsonccc on 12/11/15.
 */
public interface GpuKernelCallFactory {

    /**
     * Creates a gpu kernel call from the given ops
     * and extra arguments
     * @param op the op to create from
     * @param otherArgs the other arguments
     * @return a gpu  kernel call
     */
    GpuKernelCall create(Op op, Object... otherArgs);
}
