package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

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
    protected Object[] originalArgs;
    protected KernelCallPointerArgs kernelCallPointerArgs;

    public PostProcessKernelCall(Op op,int[] dimension,KernelCallPointerArgs kernelCallPointerArgs,Object[] originalArgs,CudaContext ctx) {
        super(op);
        this.kernelCallPointerArgs = kernelCallPointerArgs;
        this.dimension = dimension;
        this.originalArgs = originalArgs;
        this.cudaContext = ctx;
        createArgs();
    }

    @Override
    public void createArgs() {
        args = new Object[] {
                op.x().tensorAlongDimension(0, dimension).length(),
                op.x().offset(),
                kernelCallPointerArgs.getX(),
                op.x().tensorAlongDimension(0, dimension).elementWiseStride(),
                kernelCallPointerArgs.getExtraArgs(),
                kernelCallPointerArgs.getZ(),
        };
    }

    @Override
    public void createMetrics() {
        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());
        metrics.setSharedMemoryNotOverMax(metrics.getBlockSize() * op.x().data().getElementSize());
        this.metrics = metrics;
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
