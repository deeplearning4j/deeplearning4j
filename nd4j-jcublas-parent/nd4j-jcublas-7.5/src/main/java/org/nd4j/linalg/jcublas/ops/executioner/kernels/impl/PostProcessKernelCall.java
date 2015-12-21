package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import jcuda.Pointer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.KernelCallPointerArgs;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl.PostProcessKernelCallPointerArgs;
import org.nd4j.linalg.jcublas.util.CudaArgs;

/**
 * Kernel call for post processing a
 * reduce operation.
 *
 * @author Adam Gibson
 */
public class PostProcessKernelCall extends BaseGpuKernelCall {
    public final static String POST_PROCESS_NAME = "postProcessLoop";
    protected int[] dimension;
    protected int resultIndex;
    protected int extraParamsIndex;
    protected Object[] originalArgs;


    public PostProcessKernelCall(Op op,int[] dimension,int resultIndex,int extraParamsIndex,Object[] originalArgs,CudaContext ctx,GpuMetrics metrics) {
        super(op);
        this.dimension = dimension;
        this.resultIndex = resultIndex;
        this.extraParamsIndex = extraParamsIndex;
        this.originalArgs = originalArgs;
        this.cudaContext = ctx;
        this.metrics = metrics;
        createArgs();
    }

    @Override
    public void createArgs() {
        args = new Object[] {
                op.x().tensorAlongDimension(0, dimension).length(),
                op.x().offset(),
                (Pointer) this.originalArgs[resultIndex],
                op.x().tensorAlongDimension(0, dimension).elementWiseStride(),
                (Pointer) this.originalArgs[extraParamsIndex],
                (Pointer) this.originalArgs[resultIndex],
        };
    }

    @Override
    public void createCudaConext() {

    }

    @Override
    public void createMetrics() {

    }

    @Override
    public void invoke() {
        KernelFunctions.invoke(
                metrics
                , true
                , CudaArgs.getModuleNameFor(op)
                , POST_PROCESS_NAME + "_" + getType(op)
                , getType(op)
                , cudaContext, args);
    }

    @Override
    public String functionName() {
        return POST_PROCESS_NAME + "_" + getType(op);
    }

    @Override
    public KernelCallPointerArgs getPointers() {
        return new PostProcessKernelCallPointerArgs(op,args);
    }
}
