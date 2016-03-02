package org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.BaseKernelCallPointerArgs;

/**
 * Created by agibsonccc on 12/21/15.
 */
public class CollapseAccumulationKernelCallPointerArgs extends BaseKernelCallPointerArgs {
    /**
     * @param op
     * @param extraArgs
     */
    public CollapseAccumulationKernelCallPointerArgs(Op op, Object[] extraArgs) {
        super(op, extraArgs);
    }

    @Override
    protected void initPointers(Op op, Object[] extraArgs) {
        /**
         * op.y() != null
         *    args = new Object[] {
         op.x(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(), accumulation.getSmallerDimension())),
         op.y(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.y(), accumulation.getSmallerDimension())),
         toArgs(op.extraArgs(),
         getType(op)),
         op.z(),
         KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.z())),
         KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
         KernelFunctions.alloc(scalarResult ? new int[]{Integer.MAX_VALUE} : accumulation.getSmallerDimension()),
         //reason here: we only work with smaller dimensions
         1,
         //if the whole buffer is to be used don't do final aggregation this happens
         //by aggregating blocks on cpu first
         };

         op.y() == null
         args = new Object[] {
         devicePointers[0],
         devicePointers[1],
         toArgs(op.extraArgs(), getType(op)),
         op.x().tensorAlongDimension(0,accumulation.getSmallerDimension()).length(),
         op.x().tensorssAlongDimension(accumulation.getSmallerDimension()),
         op.n(),
         op.x().tensorAlongDimension(0,accumulation.getSmallerDimension()).elementWiseStride(),
         op.x().tensorssAlongDimension(accumulation.getOriginalDimension()),
         metrics.getSharedMemory(),
         devicePointers[2],
         KernelFunctions.alloc(scalarResult ? new int[]{Integer.MAX_VALUE} : accumulation.getSmallerDimension()),
         1
         };
         */

        if(op.y() != null) {
            this.x = (Pointer) extraArgs[0];
            this.xShapeInfoPointer = (Pointer) extraArgs[1];
            this.y = (Pointer) extraArgs[2];
            this.yShapeInfoPointer = (Pointer) extraArgs[3];
            this.extraArgsPointer = (Pointer) extraArgs[4];
            this.z = (Pointer) extraArgs[5];
            this.zShapeInfoPointer = (Pointer) extraArgs[6];
            this.gpuInfoPointer = (Pointer) extraArgs[7];
            this.dimensionArrPointer = (Pointer) extraArgs[8];
        }
        else {
            this.x = (Pointer) extraArgs[0];
            this.xShapeInfoPointer = (Pointer) extraArgs[1];
            this.extraArgsPointer = (Pointer) extraArgs[2];
            this.z = (Pointer)  extraArgs[9];
            this.dimensionArrPointer = (Pointer) extraArgs[10];
        }
    }
}
