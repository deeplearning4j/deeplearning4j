package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.KernelCallPointerArgs;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl.ScalarKernelCallPointerArgs;
import org.nd4j.linalg.jcublas.util.CudaArgs;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * Kernel call for scalars
 *
 * @author Adam Gibson
 */
public class ScalarKernelCall extends BaseGpuKernelCall {
    public ScalarKernelCall(Op op) {
        super(op);
        createArgs();
    }

    @Override
    public void createArgs() {
        ScalarOp scalarOp = (ScalarOp) op;
        if (op.y() != null) {
            metrics.setSharedMemory(metrics.getSharedMemory() * 2);

            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int yStride = BlasBufferUtil.getBlasStride(op.y());
            if(yStride < 0) {
                op.setY(op.y().dup());
            }

            args = new Object[]{
                    CudaArgs.getOpCode(op),
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z()
                    ,metrics.getBlockSize()
            };




        } else {
            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }


            args = new Object[]{
                    CudaArgs.getOpCode(op),
                    op.n(),
                    op.x().offset(),
                    PointerUtil.getPointer(scalarOp),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),metrics.getBlockSize()
            };

        }


    }

    @Override
    public void createMetrics() {
        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());
        metrics.setGridSize(op.n());
        metrics.setBlockSize(1024);
        metrics.setSharedMemory(metrics.getBlockSize() * op.x().data().getElementSize());
        this.metrics = metrics;
    }

    @Override
    public void invoke() {
        try(KernelParamsWrapper kParams = new KernelParamsWrapper(true,args).setResultArray(op.z())) {
            this.args = kParams.getKernelParameters();
            cudaContext = kParams.getContext();
            super.invoke();
        } catch(Exception e) {
            throw new RuntimeException("Could not execute kernel", e);
        }

    }

    @Override
    public KernelCallPointerArgs getPointers() {
        return new ScalarKernelCallPointerArgs(op,args);
    }

}
