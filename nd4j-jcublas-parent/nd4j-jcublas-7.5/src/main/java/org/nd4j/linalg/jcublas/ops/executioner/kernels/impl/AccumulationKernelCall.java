package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * Kernel call
 *  for accumulation
 *
 * @author Adam Gibson
 */
public class AccumulationKernelCall extends BaseGpuKernelCall {
    protected int[] dimension;
    protected INDArray result;
    protected int xStride;
    protected int yStride;
    protected int[] multiDimension;

    /**
     * Accumulation kernel call
     * @param op the op to use
     * @param dimension the dimensions for reduction
     * @param result the resulting operation
     */
    public AccumulationKernelCall(Op op,int[] dimension,INDArray result) {
        super(op);
        if(dimension == null)
            dimension = new int[] {Integer.MAX_VALUE};
        this.dimension = dimension;

        if(dimension.length > 1) {
            if(dimension.length == op.x().rank()) {
                this.dimension = new int[] {Integer.MAX_VALUE};
            }
            else {
                //the dimensions need to be in order
                Arrays.sort(dimension);
                this.multiDimension = dimension;
                //switch it to be being only the last dimension
                //the tad will be the prod of the previous dimensions
                this.dimension = new int[] {dimension[dimension.length - 1]};
            }

        }

        Accumulation acc = (Accumulation) op;
        if(result == null)
            this.result = Nd4j.scalar(acc.zeroDouble());
        else
            this.result = result;
    }


    public void multiReduce() {
        int lengthDelta = op.x().tensorssAlongDimension(dimension) / op.x().tensorssAlongDimension(multiDimension);
        //the shape of the new collapsed result
        INDArray collapsedResult = Nd4j.create(ArrayUtil.removeIndex(op.x().shape(),multiDimension));
        Accumulation acc = (Accumulation) op;
        for(int i = 0; i < result.length(); i++) {
            collapsedResult.putScalar(i % lengthDelta,acc.combineSubResults(collapsedResult.getDouble(i % lengthDelta),result.getDouble(i)));
        }

        for(int i = 0; i < collapsedResult.length(); i++) {
            ((Accumulation) op).setFinalResult(collapsedResult.getDouble(i));
            collapsedResult.putScalar(i,acc.getAndSetFinalResult(collapsedResult.getDouble(i)));
        }

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
            xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : op.x().tensorAlongDimension(0, dimension));
            if (xStride < 0) {
                op.setX(op.x().dup());
                xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : op.x().tensorAlongDimension(0, dimension));
                if (xStride < 0)
                    throw new IllegalStateException("Unable to compute element wise stride");

            }

            yStride = BlasBufferUtil.getBlasStride(dimension == null ? op.y() : op.y().tensorAlongDimension(0, dimension));
            if (op.y().ordering() != op.x().ordering()) {
                op.setY(op.y().dup(op.x().ordering()));
                yStride = BlasBufferUtil.getBlasStride(dimension == null ? op.y() : op.y().tensorAlongDimension(0, dimension));
                if (yStride < 0)
                    throw new IllegalStateException("Unable to compute element wise stride");

            }


            args = new Object[] {
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

            xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : firstTad);
            if (xStride < 0) {
                op.setX(op.x().dup());
                xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : firstTad);
                //dup didn't handle it
                if (xStride < 0) {
                    throw new IllegalStateException("Unable to compute element wise stride for x");}
            }

            int sharedMemBasedOnBlockSize = op.n() * op.x().data().getElementSize();
            if (sharedMemBasedOnBlockSize < 1024)
                sharedMemBasedOnBlockSize = 1024;
            metrics.setSharedMemoryNotOverMax(sharedMemBasedOnBlockSize);


            int length = op.x().data().length();
            if (dimension == null && xStride == 1 && op.x().offset() == 0)
                length = op.n();

            args = new Object[] {
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
        try(KernelParamsWrapper kParams = new KernelParamsWrapper(true,args).setResultOp(acc, result,dimension)) {
            //setup the kernel parameters such that super.invoke() will call the kernel with the given parameters
            this.args = kParams.getKernelParameters();
            this.cudaContext = kParams.getContext();
            super.invoke();
        } catch(Exception e) {
            throw new RuntimeException("Could not execute kernel", e);
        }

    }

    private int toInt(boolean val) {
        return val ? 1 : 0;
    }

}