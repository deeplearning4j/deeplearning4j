package org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.BaseKernelCallPointerArgs;

/**
 * Created by agibsonccc on 12/21/15.
 */
public class PostProcessKernelCallPointerArgs extends BaseKernelCallPointerArgs {
    /**
     * @param op
     * @param extraArgs
     */
    public PostProcessKernelCallPointerArgs(Op op, Object[] extraArgs) {
        super(op, extraArgs);
    }

    @Override
    protected void initPointers(Op op, Object[] extraArgs) {
        /**
         *     args = new Object[] {
         op.x().tensorAlongDimension(0, dimension).length(),
         op.x().offset(),
         (Pointer) this.originalArgs[resultIndex],
         op.x().tensorAlongDimension(0, dimension).elementWiseStride(),
         (Pointer) this.originalArgs[extraParamsIndex],
         (Pointer) this.originalArgs[resultIndex],
         };
          */
        this.x = (Pointer) extraArgs[2];
        this.extraArgsPointer = (Pointer) extraArgs[4];
        this.z = (Pointer) extraArgs[5];
    }
}
