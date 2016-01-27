package org.nd4j.linalg.jcublas.ops.executioner.kernels.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.BaseGpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.KernelCallPointerArgs;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.args.impl.IndexAccumulationKernelCallPointerArgs;
import org.nd4j.linalg.jcublas.util.CudaArgs;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;

import java.util.Arrays;

/**
 * Index accumulation kernel call
 * @author Adam Gibson
 */
public class IndexAccumulationKernelCall extends BaseGpuKernelCall {
    protected int[] dimension;
    protected INDArray result;
    protected int xStride;
    protected int yStride;

    /**
     * Accumulation kernel call
     * @param op the op to use
     * @param dimension the dimensions for reduction
     * @param result the resulting operation
     */
    public IndexAccumulationKernelCall(Op op,int[] dimension,INDArray result) {
        super(op);
        if(dimension == null)
            dimension = new int[] {Integer.MAX_VALUE};

        this.dimension = dimension;
        //ensure dimensions are sorted
        Arrays.sort(dimension);
        IndexAccumulation acc = (IndexAccumulation) op;
        if(result == null)
            this.result = Nd4j.scalar(acc.zeroDouble());
        else
            this.result = result;

        createArgs();
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


            args = new Object[]{
                    CudaArgs.getOpCode(op),
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
                // for 1D array we don't need TAD
                if (dimension.length == 1 && dimension[0] == Integer.MAX_VALUE) {
                    firstTad = op.x();
                    dimension = null;
                } else {
                    firstTad = op.x().tensorAlongDimension(0, dimension);
                    if (firstTad.length() == op.x().length())
                        dimension = null;
                }
            }

            xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : firstTad);
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
                    CudaArgs.getOpCode(op),
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
    public static  void calculateBlockResult(IndexAccumulation op,INDArray resultAcrossBlocks) {
        int oldN = op.n();
        Pair<Double,Integer> pair = op.zeroPair();
        for(int i = 0; i < resultAcrossBlocks.length(); i++) {
            int firstVal = (int) resultAcrossBlocks.data().getDouble(resultAcrossBlocks.offset() + i * resultAcrossBlocks.elementWiseStride());
            pair = op.combineSubResults(Pair.create(op.x().getDouble(firstVal),firstVal),pair);
        }

        if(resultAcrossBlocks.length() == 1)
            op.setFinalResult(resultAcrossBlocks.getInt(0));

        op.setFinalResult(pair.getSecond());
        op.setN(oldN);

    }


    @Override
    public void invoke() {
        IndexAccumulation acc = (IndexAccumulation) op;
        try(KernelParamsWrapper kParams = new KernelParamsWrapper(true,args).setResultOp(acc, result,dimension)) {
            this.args = kParams.getKernelParameters();
            this.cudaContext = kParams.getContext();
            //setup the kernel parameters such that super.invoke() will call the kernel with the given parameters
            super.invoke();
        } catch(Exception e) {
            throw new RuntimeException("Could not execute kernel", e);
        }

    }

    @Override
    public KernelCallPointerArgs getPointers() {
        return new IndexAccumulationKernelCallPointerArgs(op,args);
    }

    private int toInt(boolean val) {
        return val ? 1 : 0;
    }
}
