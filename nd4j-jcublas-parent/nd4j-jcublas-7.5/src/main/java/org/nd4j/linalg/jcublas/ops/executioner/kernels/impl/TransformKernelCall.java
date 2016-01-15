package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.KernelCallPointerArgs;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl.TransformKernelCallPointerArgs;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;

/**
 * Kernel call for transform
 * @author Adam Gibson
 *
 */
public class TransformKernelCall extends BaseGpuKernelCall {
    public TransformKernelCall(Op op) {
        super(op);
        createArgs();
    }

    @Override
    public void createArgs() {
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
            else if(op.y().ordering() != op.x().ordering()) {
                op.setY(op.y().dup(op.x().ordering()));
            }

            /**
             * Construct pointer arguments in the following order:
             * n
             * offset,
             * pointer to buffer
             * increment,
             * extraArgs,
             * result
             */
            System.out.println("TKC op.y() != null");

            args = new Object[] {
                    getOpCode(op),
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.z().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),
                    BlasBufferUtil.getBlasStride(op.z())
                    ,metrics.getBlockSize()
            };



        } else {
            System.out.println("TKC op.y() == null");
            args = new Object[] {
                    getOpCode(op),
                    op.n(),
                    op.x().offset(),
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
        metrics.setSharedMemoryNotOverMax(metrics.getBlockSize() * op.x().data().getElementSize());
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
        return new TransformKernelCallPointerArgs(op,args);
    }


    private int getOpCode(Op op) {
        String name = op.name();
        int code = -1;
        if (name.equals("abs")) {
            code = 0;
        } else if (name.equals("ceil")) {
            code = 1;
        } else if (name.equals("cos")) {
            code = 2;
        } else if (name.equals("exp")) {
            code = 3;
        } else if (name.equals("floor")) {
            code = 4;
        } else if (name.equals("log")) {
            code = 5;
        } else if (name.equals("neg")) {
            code = 6;
        } else if (name.equals("pow")) {
            code = 7;
        } else if (name.equals("round")) {
            code = 8;
        } else if (name.equals("setrange")) {
            code = 9;
        } else if (name.equals("sigmoid")) {
            code = 10;
        } else if (name.equals("sign")) {
            code = 11;
        } else if (name.equals("sin")) {
            code = 12;
        } else if (name.equals("softplus")) {
            code = 13;
        } else if (name.equals("sqrt")) {
            code = 14;
        } else if (name.equals("tanh")) {
            code = 15;
        } else if (name.equals("acos")) {
            code = 16;
        } else if (name.equals("asin")) {
            code = 17;
        } else if (name.equals("atan")) {
            code = 18;
        }


        System.out.println("Looking for op.name: [" + name + "] -> [" + code+"]");
        return code;
    }
}
