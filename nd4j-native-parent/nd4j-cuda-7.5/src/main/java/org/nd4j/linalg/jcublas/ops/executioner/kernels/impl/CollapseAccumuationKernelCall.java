package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TadCollapseAccumulation;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.KernelCallPointerArgs;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl.CollapseAccumulationKernelCallPointerArgs;
import org.nd4j.linalg.jcublas.util.CudaArgs;

/**
 * Kernel call
 * for running the kernel function
 * collapseTad on a given
 * reduction function
 *
 * @author Adam Gibson
 */
public class CollapseAccumuationKernelCall extends BaseGpuKernelCall {
    protected  int xStride;
    protected int yStride;
    protected boolean scalarResult;
    protected TadCollapseAccumulation accumulation;
    protected KernelCallPointerArgs kernelCallPointerArgs;
    public CollapseAccumuationKernelCall(KernelCallPointerArgs kernelCallPointerArgs,Op op,GpuMetrics metrics,CudaContext cudaContext) {
        super(op);
        if(!(op instanceof TadCollapseAccumulation))
            throw new IllegalArgumentException("Op must be of type " + TadCollapseAccumulation.class.getName());
        this.accumulation = (TadCollapseAccumulation) op;
        this.kernelCallPointerArgs = kernelCallPointerArgs;
        this.metrics = metrics;
        this.cudaContext = cudaContext;
        createArgs();
    }

    @Override
    public String functionName() {
        return "collapseTad_" + getType(op);
    }

    @Override
    public KernelCallPointerArgs getPointers() {
        return new CollapseAccumulationKernelCallPointerArgs(op,args);
    }

    @Override
    public void createArgs() {
        if (op.y() != null) {
            metrics.setSharedMemoryNotOverMax(metrics.getSharedMemory() * 2);
            xStride = BlasBufferUtil.getBlasStride(scalarResult ? op.x() : op.x().tensorAlongDimension(0, accumulation.getSmallerDimension()));
            if (xStride < 0) {
                op.setX(op.x().dup());
                xStride = BlasBufferUtil.getBlasStride(scalarResult ? op.x() : op.x().tensorAlongDimension(0, accumulation.getSmallerDimension()));
                if (xStride < 0)
                    throw new IllegalStateException("Unable to compute element wise stride");

            }

            yStride = BlasBufferUtil.getBlasStride(scalarResult ? op.y() : op.y().tensorAlongDimension(0, accumulation.getSmallerDimension()));
            if (op.y().ordering() != op.x().ordering()) {
                op.setY(op.y().dup(op.x().ordering()));
                yStride = BlasBufferUtil.getBlasStride(scalarResult ? op.y() : op.y().tensorAlongDimension(0, accumulation.getSmallerDimension()));
                if (yStride < 0)
                    throw new IllegalStateException("Unable to compute element wise stride");

            }

            //result index for the pointer to use when invoking the post process method
            args = new Object[] {
                    kernelCallPointerArgs.getX(),
                    kernelCallPointerArgs.getXShapeInfoPointer(),
                    kernelCallPointerArgs.getY(),
                    kernelCallPointerArgs.getYShapeInfoPointer(),
                    kernelCallPointerArgs.getExtraArgs(),
                    kernelCallPointerArgs.getZ(),
                    kernelCallPointerArgs.getZShapeInfoPointer(),
                    kernelCallPointerArgs.getGpuInfoPointer(),
                    kernelCallPointerArgs.getDimensionArrPointer(),
                    //reason here: we only work with smaller dimensions
                    1,
                    //if the whole buffer is to be used don't do final aggregation this happens
                    //by aggregating blocks on cpu first
            };


        } else {
            INDArray firstTad = null;
            //handle case where the tad is actually the whole array
            if (!scalarResult) {
                firstTad = op.x().tensorAlongDimension(0, accumulation.getSmallerDimension());
            }

            xStride = BlasBufferUtil.getBlasStride(scalarResult ? op.x() : firstTad);
            if (xStride < 0) {
                op.setX(op.x().dup());
                xStride = BlasBufferUtil.getBlasStride(scalarResult ? op.x() : firstTad);
                //dup didn't handle it
                if (xStride < 0) {
                    throw new IllegalStateException("Unable to compute element wise stride for x");}
            }

            int sharedMemBasedOnBlockSize = op.n() * op.x().data().getElementSize();
            if (sharedMemBasedOnBlockSize < 1024)
                sharedMemBasedOnBlockSize = 1024;
            metrics.setSharedMemoryNotOverMax(sharedMemBasedOnBlockSize);


            int length = op.x().data().length();
            if (scalarResult && xStride == 1 && op.x().offset() == 0)
                length = op.n();
            //result index for the pointer to use when invoking the post process method

            /**
             * 		T *data
             ,T *result
             ,T *extraParams
             ,int elementsPerTad
             ,int numTads
             ,int n
             ,int elementWiseStride
             ,int numOriginalTads,int sharedMemorySize,
             int *xShapeInfo
             ,int *dimension,int dimensionLength
             */

            args = new Object[] {
                    kernelCallPointerArgs.getZ(),
                    kernelCallPointerArgs.getZ(),
                    kernelCallPointerArgs.getExtraArgs(),
                    op.x().tensorAlongDimension(0,accumulation.getOriginalDimension()).length(),
                    op.x().tensorssAlongDimension(accumulation.getOriginalDimension()),
                    length,
                    op.x().tensorAlongDimension(0, accumulation.getSmallerDimension()).elementWiseStride(),
                    op.x().tensorssAlongDimension(accumulation.getSmallerDimension()),
                    metrics.getSharedMemory(),
                    kernelCallPointerArgs.getXShapeInfoPointer(),
                    kernelCallPointerArgs.getZShapeInfoPointer(),
                    kernelCallPointerArgs.getDimensionArrPointer(),
                    1,
                    //iterate along the last dimension wrt the solution
                    accumulation.getOriginalDimension()[accumulation.getOriginalDimension().length - 2]
            };
        }
    }

    @Override
    public void invoke() {
        String moduleName = CudaArgs.getModuleNameFor(op);
        /**
         * Invoke a cuda kernel by name. This will be wrt the function name.
         * Functions that are accumulations or transforms have names that end with _strided.
         *
         */

        metrics.validate();
        //module name is the op, function name is transform
        KernelFunctions.invoke(
                metrics,
                true
                , moduleName,
                functionName()
                , getType(op), cudaContext
                , args);
    }

    @Override
    public void createMetrics() {

    }



}
