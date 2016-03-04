package org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.BaseKernelCallPointerArgs;

/**
 * Created by agibsonccc on 12/21/15.
 */
public class ScalarKernelCallPointerArgs extends BaseKernelCallPointerArgs {
    /**
     * @param op
     * @param extraArgs
     */
    public ScalarKernelCallPointerArgs(Op op, Object[] extraArgs) {
        super(op, extraArgs);
    }

    @Override
    protected void initPointers(Op op, Object[] extraArgs) {
        /**
         * op.y() != null
         *   args = new Object[]{
         op.n(),
         op.x().offset(),
         op.y().offset(),
         op.x(),
         op.y(),
         BlasBufferUtil.getBlasStride(op.x()),
         BlasBufferUtil.getBlasStride(op.y()),
         toArgs(op.(extraArgs(), getType(op)),
         op.z()
         ,metrics.getBlockSize()
         };


         op.y() == null
         args = new Object[]{
         op.n(),
         op.x().offset(),
         PointerUtil.getPointer(scalarOp),
         op.x(),
         BlasBufferUtil.getBlasStride(op.x()),
         toArgs(op.extraArgs(), getType(op)),
         op.z(),metrics.getBlockSize()
         };

         */
        if(op.y() != null) {
            this.x = (Pointer) extraArgs[3];
            this.y = (Pointer) extraArgs[4];
            this.extraArgsPointer = (Pointer) extraArgs[7];
            this.z = (Pointer) extraArgs[8];
        }
        else {
            this.x = (Pointer) extraArgs[3];
            this.extraArgsPointer = (Pointer) extraArgs[6];
            this.z = (Pointer) extraArgs[7];
        }
    }
}
