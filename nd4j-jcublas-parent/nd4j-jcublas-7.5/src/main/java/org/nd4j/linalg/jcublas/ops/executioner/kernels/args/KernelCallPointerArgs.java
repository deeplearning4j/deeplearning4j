package org.nd4j.linalg.jcublas.ops.executioner.kernels.args;

import jcuda.Pointer;

/**
 * Wrapper class for different kinds of device pointers
 *  and operations.
 *
 *  This is meant for the easy translation of and
 *  reuse of pointers
 *  across composing operations.
 *
 *  @author Adam Gibson
 */
public interface KernelCallPointerArgs {
    /**
     *
     * @return
     */
    Pointer getGpuInfoPointer();

    /**
     *
     * @return
     */
    Pointer getXShapeInfoPointer();

    /**
     *
     * @return
     */
    Pointer getYShapeInfoPointer();

    /**
     *
     * @return
     */
    Pointer getZShapeInfoPointer();

    /**
     *
     * @return
     */
    Pointer getDimensionArrPointer();

    /**
     *
     * @return
     */
    Pointer getX();

    /**
     *
     * @return
     */
    Pointer getY();

    /**
     *
     * @return
     */
    Pointer getZ();

    /**
     *
     * @return
     */
    Pointer getExtraArgs();
}
