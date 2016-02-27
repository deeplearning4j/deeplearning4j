package org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.BaseKernelCallPointerArgs;

/**
 * Kernel args for
 * transform operations
 * @author Adam Gibson
 */
public class TransformKernelCallPointerArgs extends BaseKernelCallPointerArgs {

    /**
     * @param op
     * @param extraArgs
     */
    public TransformKernelCallPointerArgs(Op op, Object[] extraArgs) {
        super(op, extraArgs);
    }

    @Override
    protected void initPointers(Op op, Object[] extraArgs) {
        /**
         * op.y() != null
         *      args = new Object[] {
         op.n(),
         op.x().offset(),
         op.y().offset(),
         op.z().offset(),
         op.x(),
         op.y(),
         BlasBufferUtil.getBlasStride(op.x()),
         BlasBufferUtil.getBlasStride(op.y()),
         toArgs(op.extraArgs(), getType(op)),
         op.z(),
         BlasBufferUtil.getBlasStride(op.z())
         ,metrics.getBlockSize()
         };

         op.y() == null
         args = new Object[] {
         op.n(),
         op.x().offset(),
         op.x(),
         BlasBufferUtil.getBlasStride(op.x()),
         toArgs(op.extraArgs(), getType(op)),
         op.z()
         ,metrics.getBlockSize()
         };

         */
        if(op.y() != null) {
            this.x = (Pointer) extraArgs[4];
            this.y = (Pointer) extraArgs[5];
            this.extraArgsPointer = (Pointer) extraArgs[8];
            this.z = (Pointer) extraArgs[9];
        }
        else {
            this.x = (Pointer) extraArgs[2];
            this.extraArgsPointer = (Pointer) extraArgs[4];
            this.z = (Pointer) extraArgs[5];
        }


    }
}
