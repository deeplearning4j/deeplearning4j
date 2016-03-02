package org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.BaseKernelCallPointerArgs;

/**
 * Pointer arguments for accumulation
 *
 * @author Adam Gibson
 */
public class AccumulationKernelCallPointerArgs extends BaseKernelCallPointerArgs {
    /**
     * @param op
     * @param extraArgs
     */
    public AccumulationKernelCallPointerArgs(Op op, Object[] extraArgs) {
        super(op, extraArgs);
    }

    @Override
    protected void initPointers(Op op, Object[] extraArgs) {
        /**
         * op.y() != null
         *  args = new Object[] {
         op.n(),
         op.x(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(), dimension)),
         op.y(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.y(), dimension)),
         toArgs(op.extraArgs(),
         getType(op)),
         op.z(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.z())),
         KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
         KernelFunctions.alloc(dimension == null ? new int[]{Integer.MAX_VALUE} : dimension),
         dimension == null ? 1 : dimension.length,
         //if the whole buffer is to be used don't do final aggregation this happens
         //by aggregating blocks on cpu first
         toInt((dimension == null || dimension[0] == Integer.MAX_VALUE))

         };

         op.y() == null

         args = new Object[] {
         length,
         op.x(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(), dimension)),
         toArgs(op.extraArgs(), getType(op)),
         op.z(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.z())),
         KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
         KernelFunctions.alloc(scalarResult ? new int[]{Integer.MAX_VALUE} : dimension),
         scalarResult ? 1 : dimension.length,
         //if the whole buffer is to be used don't do final aggregation this happens
         //by aggregating blocks on cpu first
         toInt(scalarResult)
         };

         */

        if(op.y() != null) {
            this.x = (Pointer) extraArgs[2];
            this.xShapeInfoPointer = (Pointer) extraArgs[3];
            this.y = (Pointer) extraArgs[4];
            this.yShapeInfoPointer = (Pointer) extraArgs[5];
            this.extraArgsPointer = (Pointer) extraArgs[6];
            this.z = (Pointer) extraArgs[7];
            this.zShapeInfoPointer = (Pointer) extraArgs[8];
            this.gpuInfoPointer = (Pointer) extraArgs[9];
            this.dimensionArrPointer = (Pointer) extraArgs[10];

        }
        else {
            this.x = (Pointer) extraArgs[2];
            this.xShapeInfoPointer = (Pointer) extraArgs[3];
            this.extraArgsPointer = (Pointer) extraArgs[4];
            this.z = (Pointer) extraArgs[5];
            this.zShapeInfoPointer = (Pointer) extraArgs[6];
            this.gpuInfoPointer = (Pointer) extraArgs[7];
            this.dimensionArrPointer = (Pointer) extraArgs[8];
        }
    }
}
