package org.nd4j.linalg.jcublas.ops.executioner.kernels;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.KernelCallPointerArgs;

/**
 * Represents a cuda call
 * used by the executioner.
 *
 * @author Adam Gibson
 */
public interface GpuKernelCall {
    /**
     * Arguments for the kernel call
     * @return
     */
    Object[] getArgs();

    /**
     * Cuda context for the
     * kernel call
     * @return
     */
    CudaContext cudaContext();

    /**
     * The op the kernel
     * call was created from
     * @return
     */
    Op op();

    /**
     * The gpu metrics
     * for the kernel launch
     * @return
     */
    GpuMetrics metrics();

    /**
     * Create the arguments for
     * tne kernel call
     * based on the arguments
     */
    void createArgs();

    /**
     * Creates the metrics for this call
     */
    void createMetrics();

    /**
     * Invoke the kernel
     */
    void invoke();
    /**
     * Invoke the kernel
     * @param functionName  the name of the function to invoke
     */
    void invoke(String functionName);

    /**
     * Function name for kernel invocation
     * @return the function name for kernel invocation
     */
    String functionName();

    /**
     * Module name for kernel invocation
     * @return the module name for kernel invocation
     */
    String moduleName();

    /**
     * Returns the pointers
     * for the given arguments
     * for this kernel call
     * @return the pointers for the kernel call
     *
     */
    KernelCallPointerArgs getPointers();

}
