package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * Kernel call for accumulation
 *
 * @author Adam Gibson
 */
public class AccumulationKernelCall extends BaseGpuKernelCall {
    protected int[] dimension;
    protected INDArray result;

    /**
     * Accumulation kernel call
     * @param op the op to use
     * @param dimension the dimensions for reduction
     * @param result the resulting operation
     */
    public AccumulationKernelCall(Op op,int[] dimension,INDArray result) {
        super(op);
        this.dimension = dimension;
        Accumulation acc = (Accumulation) op;
        if(result == null)
            this.result = Nd4j.scalar(acc.zeroDouble());
        else
            this.result = result;
    }

    @Override
    public void createMetrics() {
        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op), op.n());
        if (dimension != null && dimension.length >= 1 && dimension[0] != Integer.MAX_VALUE) {
            int length = op.x().tensorssAlongDimension(dimension);
            if (length > 1000)
                length = 1000;
            //of note here: THIS IS REVERSE OF WHAT IT SHOULD BE, THIS IS INTENDED.
            metrics.setGridSizeNotOverMax(length);
            metrics.setBlockSizeNotOverMax(op.x().tensorAlongDimension(0, dimension).length());
            int sharedMemBasedOnBlockSize = op.x().tensorAlongDimension(0, dimension).length() * 10 * op.x().data().getElementSize();
            if (sharedMemBasedOnBlockSize < 1024)
                sharedMemBasedOnBlockSize = 1024;
            metrics.setSharedMemoryNotOverMax(sharedMemBasedOnBlockSize);
        } else {
            int sharedMemBasedOnBlockSize = op.n() * op.x().data().getElementSize();
            if (sharedMemBasedOnBlockSize < 1024)
                sharedMemBasedOnBlockSize = 1024;
            metrics.setSharedMemoryNotOverMax(sharedMemBasedOnBlockSize);
            //setup a number of threads = the number of blocks being launched
            result = Nd4j.create(metrics.getGridSize());
        }

        this.metrics = metrics;


    }

    @Override
    public void createCudaConext() {

    }

    @Override
    public void createArgs() {
        if (op.y() != null) {
            metrics.setSharedMemoryNotOverMax(metrics.getSharedMemory() * 2);
            int xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : op.x().tensorAlongDimension(0, dimension));
            if (xStride < 0) {
                op.setX(op.x().dup());
                xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : op.x().tensorAlongDimension(0, dimension));
                if (xStride < 0)
                    throw new IllegalStateException("Unable to compute element wise stride");

            }

            int yStride = BlasBufferUtil.getBlasStride(dimension == null ? op.y() : op.y().tensorAlongDimension(0, dimension));
            if (op.y().ordering() != op.x().ordering()) {
                op.setY(op.y().dup(op.x().ordering()));
                yStride = BlasBufferUtil.getBlasStride(dimension == null ? op.y() : op.y().tensorAlongDimension(0, dimension));
                if (yStride < 0)
                    throw new IllegalStateException("Unable to compute element wise stride");

            }


            args = new Object[]{
                    op.n(),
                    op.x(),
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(), dimension)),
                    op.y(),
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.y(), dimension)),
                    toArgs(op.extraArgs(),
                            getType(op)),
                    result,
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(result)),
                    KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
                    KernelFunctions.alloc(dimension == null ? new int[]{Integer.MAX_VALUE} : dimension),
                    dimension == null ? 1 : dimension.length,
                    //if the whole buffer is to be used don't do final aggregation this happens
                    //by aggregating blocks on cpu first
                    toInt((dimension == null || dimension[0] == Integer.MAX_VALUE))

            };


        } else {
            INDArray firstTad = null;
            //handle case where the tad is actually the whole array
            if (dimension != null) {
                firstTad = op.x().tensorAlongDimension(0, dimension);
                if (firstTad.length() == op.x().length())
                    dimension = null;
            }

            int xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : firstTad);
            if (xStride < 0) {
                op.setX(op.x().dup());
                xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : firstTad);
                //dup didn't handle it
                if (xStride < 0) {
                    int[] squashed = Shape.squeeze(op.x().shape());
                    int lengthDiff = Math.abs(op.x().shape().length - squashed.length);
                    if (lengthDiff < 1) {
                        throw new IllegalStateException("Unable to compute element wise stride for x");
                    }
                    for (int i = 0; i < dimension.length; i++)
                        dimension[i] -= lengthDiff;
                    INDArray reshapedX = op.x().reshape(squashed).dup();
                    xStride = reshapedX.tensorAlongDimension(0, dimension).elementWiseStride();
                    op.setX(reshapedX);
                }
            }

            int sharedMemBasedOnBlockSize = op.n() * op.x().data().getElementSize();
            if (sharedMemBasedOnBlockSize < 1024)
                sharedMemBasedOnBlockSize = 1024;
            metrics.setSharedMemoryNotOverMax(sharedMemBasedOnBlockSize);


            int length = op.x().data().length();
            if (dimension == null && xStride == 1 && op.x().offset() == 0)
                length = op.n();

            args = new Object[]{
                    length,
                    op.x(),
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(), dimension)),
                    toArgs(op.extraArgs(), getType(op)),
                    result,
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(result)),
                    KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
                    KernelFunctions.alloc(dimension == null ? new int[]{Integer.MAX_VALUE} : dimension),
                    dimension == null ? 1 : dimension.length,
                    //if the whole buffer is to be used don't do final aggregation this happens
                    //by aggregating blocks on cpu first
                    toInt((dimension == null || dimension[0] == Integer.MAX_VALUE))
            };


        }
    }

    /**
     * Calculates a reduction across blocks
     * @param op
     * @param resultAcrossBlocks
     */
    public static  void calculateBlockResult(Accumulation op,INDArray resultAcrossBlocks) {
        int oldN = op.n();
        op.setX(resultAcrossBlocks);
        op.setApplyFinalTransform(false);
        double result = op.zeroDouble();
        for(int i = 0; i < resultAcrossBlocks.length(); i++) {
            double firstVal = resultAcrossBlocks.data().getDouble(resultAcrossBlocks.offset() + i * resultAcrossBlocks.elementWiseStride());
            result = op.combineSubResults(firstVal,result);
        }

        if(resultAcrossBlocks.length() == 1)
            result = resultAcrossBlocks.getDouble(0);

        op.setFinalResult(result);
        op.setApplyFinalTransform(true);
        op.setN(oldN);
        op.getAndSetFinalResult(op.getFinalResult().doubleValue());
    }



    @Override
    public void invoke() {
        Accumulation acc = (Accumulation) op;
        try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,true,args).setResultOp(acc, result,dimension)) {
            this.args = kParams.getKernelParameters();
            super.invoke();
        } catch(Exception e) {
            throw new RuntimeException("Could not execute kernel", e);
        }

    }

    private int toInt(boolean val) {
        return val ? 1 : 0;
    }

}