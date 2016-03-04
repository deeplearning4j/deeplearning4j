package org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.BaseKernelCallPointerArgs;

/**
 * Created by agibsonccc on 12/21/15.
 */
public class BroadcastKernelCallPointerArgs extends BaseKernelCallPointerArgs {
    /**
     * @param op
     * @param extraArgs
     */
    public BroadcastKernelCallPointerArgs(Op op, Object[] extraArgs) {
        super(op, extraArgs);
    }

    @Override
    protected void initPointers(Op op, Object[] extraArgs) {
        /**
         * this.args = new Object[] {
         op.x(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(), dimensions)),
         op.y(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.y())),
         op.z(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.z(),dimensions)),
         KernelFunctions.alloc(dimensions),
         dimensions.length,
         KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
         };

         */
        this.x = (Pointer) extraArgs[0];
        this.xShapeInfoPointer = (Pointer) extraArgs[1];
        this.y = (Pointer) extraArgs[2];
        this.yShapeInfoPointer = (Pointer) extraArgs[3];
        this.z = (Pointer) extraArgs[4];
        this.zShapeInfoPointer = (Pointer) extraArgs[5];
        this.dimensionArrPointer = (Pointer) extraArgs[6];
        this.gpuInfoPointer = (Pointer) extraArgs[8];
    }
}
