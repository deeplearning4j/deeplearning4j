package org.nd4j.linalg.jcublas.ops.executioner.kernels.args;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;

/**
 * Base class for kernel call pointer
 * arguments
 *
 * @author Adam Gibson
 */
public abstract  class BaseKernelCallPointerArgs implements KernelCallPointerArgs {
   protected Pointer gpuInfoPointer;
    protected Pointer xShapeInfoPointer;
    protected Pointer yShapeInfoPointer;
    protected Pointer zShapeInfoPointer;
    protected Pointer dimensionArrPointer;
    protected Pointer x,y,z;
    protected Pointer extraArgsPointer;

    /**
     *
     * @param op
     * @param extraArgs
     */
    public BaseKernelCallPointerArgs(Op op,Object[] extraArgs) {
        initPointers(op,extraArgs);
    }

    /**
     * Initialize the pointers
     * wrt the op and extra arguments
     * @param op the op to initialized based on
     * @param extraArgs the extra arguments
     */
    protected abstract  void initPointers(Op op,Object[] extraArgs);



    @Override
    public Pointer getGpuInfoPointer() {
        return gpuInfoPointer;
    }

    @Override
    public Pointer getXShapeInfoPointer() {
        return xShapeInfoPointer;
    }

    @Override
    public Pointer getYShapeInfoPointer() {
        return yShapeInfoPointer;
    }

    @Override
    public Pointer getZShapeInfoPointer() {
        return zShapeInfoPointer;
    }

    @Override
    public Pointer getDimensionArrPointer() {
        return dimensionArrPointer;
    }

    @Override
    public Pointer getX() {
        return x;
    }

    @Override
    public Pointer getY() {
        return y;
    }

    @Override
    public Pointer getZ() {
        return z;
    }

    @Override
    public Pointer getExtraArgs() {
        return extraArgsPointer;
    }
}
