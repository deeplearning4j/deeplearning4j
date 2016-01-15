package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.KernelCallPointerArgs;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl.BroadcastKernelCallPointerArgs;
import org.nd4j.linalg.jcublas.util.CudaArgs;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * Broad cast kernel call
 *
 * @author Adam Gibson
 */
public class BroadcastKernelCall extends BaseGpuKernelCall {
    protected int[] dimensions;

    public BroadcastKernelCall(Op op,int[] dimensions) {
        super(op);
        this.dimensions = dimensions;
        createArgs();
    }

    @Override
    public void createArgs() {
        this.args = new Object[] {
                CudaArgs.getOpCode(op),
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

    }

    @Override
    public void createMetrics() {
        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());
        metrics.setGridSizeNotOverMax(512);
        int blocksPerGrid =(op.n() + metrics.getGridSize() - 1) / metrics.getGridSize();
        metrics.setBlockSizeNotOverMax(blocksPerGrid);
        metrics.setSharedMemoryNotOverMax(1);
        this.metrics = metrics;
    }

    @Override
    public void invoke() {

        /**
         *
         * Will need to get an element wise stride
         * along a broadcast dimension wrt the shape
         * This will allow us to setup a linear
         * operator along a subset of the original array
         * repeating along the desired dimensions.
         *
         * Will also need to figure out how to split
         * the bigger array wrt the original input
         * computing broadcast slices wrt the
         * specified y being broadcast.
         *
         * Will also need an element wise stride
         * for the bigger array
         * and a way to compute the offsets
         * (likely related to the major stride of the bigger array?)
         *
         * This will be very similar to how TAD is designed.
         *
         * There will likely be times when we need to compute a dup()
         * in order to force alignment of the data.
         */
        try(KernelParamsWrapper kParams = new KernelParamsWrapper(true,args).setResultArray(op.z())) {
            this.args = kParams.getKernelParameters();
            this.cudaContext = kParams.getContext();
            super.invoke();

        } catch(Exception e) {
            throw new RuntimeException("Could not execute kernel: Kernel launch was: " + metrics, e);
        }



    }

    @Override
    public KernelCallPointerArgs getPointers() {
        return new BroadcastKernelCallPointerArgs(op,args);
    }
}
