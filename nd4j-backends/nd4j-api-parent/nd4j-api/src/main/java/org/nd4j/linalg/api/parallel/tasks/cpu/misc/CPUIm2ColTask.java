package org.nd4j.linalg.api.parallel.tasks.cpu.misc;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveTask;

/**
 *
 * Parallel Im2Col implementation
 * @author Alex Black
 */
public class CPUIm2ColTask extends RecursiveTask<INDArray> implements Task<INDArray> {
    protected Future<INDArray> future;
    protected List<CPUIm2ColTask> subTasks;    //For callable execution

    protected final INDArray img;
    protected INDArray out;
    protected final int kernelHeight;
    protected final int kernelWidth;
    protected final int strideY;
    protected final int strideX;
    protected final int padHeight;
    protected final int padWidth;
    protected final boolean coverAll;
    protected final int parallelThreshold;

    protected final int exampleFrom;
    protected final int exampleTo;
    protected final int depthFrom;
    protected final int depthTo;
    protected final int xOutFrom;
    protected final int xOutTo;
    protected final int yOutFrom;
    protected final int yOutTo;

    public CPUIm2ColTask(INDArray img, int kernelHeight, int kernelWidth, int strideY, int strideX, int padHeight,
                         int padWidth, boolean coverAll, int parallelThreshold) {
        this(img, getNewOutputArray(img, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth, coverAll),
                kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                0, img.size(0), //example ranges
                0, img.size(1), //depth ranges
                0, Convolution.outSize(img.size(2), kernelHeight, strideY, padHeight, coverAll), //yOut ranges (along height)
                0, Convolution.outSize(img.size(3), kernelWidth, strideX, padWidth, coverAll), //xOut ranges (along width)
                coverAll, parallelThreshold);
        //NOTE: Ranges above are [from,to) i.e., exclusive of to, inclusive of from
    }

    public CPUIm2ColTask(INDArray img, INDArray out, int kernelHeight, int kernelWidth, int strideY, int strideX, int padHeight, int padWidth,
                         int exampleFrom, int exampleTo, int depthFrom, int depthTo, int yOutFrom, int yOutTo, int xOutFrom, int xOutTo,
                         boolean coverAll, int parallelThreshold) {
        this.img = img;
        this.out = out;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.strideY = strideY;
        this.strideX = strideX;
        this.padHeight = padHeight;
        this.padWidth = padWidth;
        this.coverAll = coverAll;
        this.parallelThreshold = parallelThreshold;

        this.exampleFrom = exampleFrom;
        this.exampleTo = exampleTo;
        this.depthFrom = depthFrom;
        this.depthTo = depthTo;
        this.xOutFrom = xOutFrom;
        this.xOutTo = xOutTo;
        this.yOutFrom = yOutFrom;
        this.yOutTo = yOutTo;
    }


    private static INDArray getNewOutputArray(INDArray img, int kernelHeight, int kernelWidth, int strideY, int strideX,
                                              int padHeight, int padWidth, boolean coverAll) {
        //number of images
        int n = img.size(0);
        //number of channels (depth)
        int c = img.size(1);
        //image height
        int h = img.size(2);
        //image width
        int w = img.size(3);
        int outHeight = Convolution.outSize(h, kernelHeight, strideY, padHeight, coverAll);
        int outWidth = Convolution.outSize(w, kernelWidth, strideX, padWidth, coverAll);

        return Nd4j.create(n, c, kernelHeight, kernelWidth, outHeight, outWidth);
    }


    @Override
    protected INDArray compute() {
        //Fork join
        splitOrExecute(true);
        return out;
    }

    @Override
    public INDArray call() {
        //Callable / ExecutorService
        splitOrExecute(true);
        return null;
    }

    private void splitOrExecute(final boolean forkJoin){
        if(!forkJoin) subTasks = new ArrayList<>();

        if (parallelThreshold != Integer.MAX_VALUE && opSize() > parallelThreshold) {
            //Split. First on examples, then on depth, then on xOut, then on yOut

            CPUIm2ColTask first;
            CPUIm2ColTask second;
            int temp;
            if ((temp = exampleTo - exampleFrom) > 1) { //exampleTo is exclusive -> single example has to-from=1
                int countFirst = temp / 2;
                first = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom, exampleFrom + countFirst,   //If countFirst=1, then want want to=from+1 exclusive, i.e., to=from inclusive
                        depthFrom, depthTo, yOutFrom, yOutTo, xOutFrom, xOutTo, coverAll, parallelThreshold);
                if(forkJoin) first.fork();
                else{
                    first.invokeAsync();
                    subTasks.add(first);
                }

                second = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom + countFirst, exampleTo,
                        depthFrom, depthTo, yOutFrom, yOutTo, xOutFrom, xOutTo, coverAll, parallelThreshold);
                if( forkJoin ) second.fork();
                else{
                    second.invokeAsync();
                    subTasks.add(second);
                }

            }
            else if ((temp = depthTo - depthFrom) > 1) {
                //Split on depth
                int countFirst = temp / 2;
                first = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom, exampleTo, depthFrom, depthFrom + countFirst,
                        yOutFrom, yOutTo, xOutFrom, xOutTo, coverAll, parallelThreshold);
                first.fork();

                second = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom, exampleTo, depthFrom + countFirst, depthTo,
                        yOutFrom, yOutTo, xOutFrom, xOutTo, coverAll, parallelThreshold);
                second.fork();

            } else if ((temp = yOutTo - yOutFrom) > 1) {
                //split on output:
                int countFirst = temp / 2;
                first = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom, exampleTo, depthFrom, depthTo,
                        yOutFrom, yOutFrom + countFirst, xOutFrom, xOutTo, coverAll, parallelThreshold);
                if( forkJoin ) first.fork();
                else{
                    first.invokeAsync();
                    subTasks.add(first);
                }

                second = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom, exampleTo, depthFrom, depthTo,
                        yOutFrom + countFirst, yOutTo, xOutFrom, xOutTo, coverAll, parallelThreshold);
                if( forkJoin ) second.fork();
                else{
                    second.invokeAsync();
                    subTasks.add(second);
                }

            } else if ((temp = xOutTo - xOutFrom) > 1) {
                int countFirst = temp / 2;
                first = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom, exampleTo, depthFrom, depthTo,
                        yOutFrom, yOutTo, xOutFrom, xOutFrom+countFirst, coverAll, parallelThreshold);
                if( forkJoin ) first.fork();
                else{
                    first.invokeAsync();
                    subTasks.add(first);
                }

                second = new CPUIm2ColTask(img, out, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth,
                        exampleFrom, exampleTo, depthFrom, depthTo,
                        yOutFrom, yOutTo, xOutFrom+countFirst, xOutTo, coverAll, parallelThreshold);
                if( forkJoin ) second.fork();
                else{
                    second.invokeAsync();
                    subTasks.add(second);
                }

            } else {
                //single image patch on one channel exceeds parallel threshold
                execute();
                return;
            }

            if(forkJoin) {
                first.join();
                second.join();
            }
        } else {
            //Execute directly
            execute();
        }
    }

    private int opSize() {
        return (exampleTo - exampleFrom) * (depthTo - depthFrom) * (xOutTo - xOutFrom) * (yOutTo - yOutFrom) * kernelHeight * kernelWidth;
    }

    private void execute() {
        DataBuffer dbIn = img.data();

        if (dbIn.allocationMode() == DataBuffer.AllocationMode.HEAP) {
            if (dbIn.dataType() == DataBuffer.Type.FLOAT) {
                doHeapFloat();
            } else {
                doHeapDouble();
            }
        } else {
            if(dbIn.dataType() == DataBuffer.Type.FLOAT) {
                doDirectFloat();
            } else {
                doDirectDouble();
            }
        }
    }

    private void doHeapFloat() {
        DataBuffer dbIn = img.data();
        DataBuffer dbOut = out.data();

        int outArrayOffset = 0;
        int[] outShape = out.shape();
        int[] outStride = out.stride();

        int inArrayOffset = 0;
        int[] inShape = img.shape();
        int[] inStride = img.stride();

        int[] outIndices = new int[6];
        int[] inIndices = new int[4];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int inShape2 = inShape[2];
        final int inShape3 = inShape[3];

        final boolean padding = padHeight > 0 || padWidth > 0;

        float[] fIn = (float[]) dbIn.array();
        float[] fOut = (float[]) dbOut.array();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
                    for (int y = yOutFrom; y < yOutTo; y++) {  //along height
                        outIndices[4] = y;
                        outIndices[5] = x;
                        int baseOffsetOut = getOffsetUnsafe6(outArrayOffset, outShape, outStride, outIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 || j + patchX >= inShape3)
                                            fOut[outBufferIdxX + patchY * outStride2] = 0f; //padding
                                        else {
                                            fOut[outBufferIdxX + patchY * outStride2] = fIn[inBufferIdxX + patchY * inStride2];
                                        }
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 || j + patchX >= inShape3)
                                            fOut[outBufferIdxY + patchX * outStride3] = 0f; //padding
                                        else {
                                            fOut[outBufferIdxY + patchX * outStride3] = fIn[inBufferIdxY + patchX * inStride3];
                                        }
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in original img
                            int j = x * strideX;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        fOut[outBufferIdxX + patchY * outStride2] = fIn[inBufferIdxX + patchY * inStride2];
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        fOut[outBufferIdxY + patchX*outStride3] = fIn[inBufferIdxY + patchX*inStride3];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void doHeapDouble(){
        DataBuffer dbIn = img.data();
        DataBuffer dbOut = out.data();

        int outArrayOffset = out.offset();
        int[] outShape = out.shape();
        int[] outStride = out.stride();

        int inArrayOffset = img.offset();
        int[] inShape = img.shape();
        int[] inStride = img.stride();

        int[] outIndices = new int[6];
        int[] inIndices = new int[4];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int inShape2 = inShape[2];
        final int inShape3 = inShape[3];

        final boolean padding = padHeight > 0 || padWidth > 0;

        double[] dIn = (double[]) dbIn.array();
        double[] dOut = (double[]) dbOut.array();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
                    for (int y = yOutFrom; y < yOutTo; y++) {  //along height
                        outIndices[4] = y;
                        outIndices[5] = x;
                        int baseOffsetOut = getOffsetUnsafe6(outArrayOffset, outShape, outStride, outIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 || j + patchX >= inShape3)
                                            dOut[outBufferIdxX + patchY * outStride2] = 0f; //padding
                                        else {
                                            dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX + patchY * inStride2];
                                        }
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape[2] || j + patchX >= inShape[3])
                                            dOut[outBufferIdxY + patchX * outStride3] = 0f; //padding
                                        else {
                                            dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY + patchX * inStride3];
                                        }
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in original img
                            int j = x * strideX;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX + patchY * inStride2];
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        dOut[outBufferIdxY + patchX*outStride3] = dIn[inBufferIdxY + patchX*inStride3];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void doDirectFloat() {
        DataBuffer dbIn = img.data();
        DataBuffer dbOut = out.data();

        int outArrayOffset = out.offset();
        int[] outShape = out.shape();
        int[] outStride = out.stride();

        int inArrayOffset = img.offset();
        int[] inShape = img.shape();
        int[] inStride = img.stride();

        int[] outIndices = new int[6];
        int[] inIndices = new int[4];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int inShape2 = inShape[2];
        final int inShape3 = inShape[3];

        final boolean padding = padHeight > 0 || padWidth > 0;

        FloatBuffer dIn = dbIn.asNioFloat();
        FloatBuffer dOut = dbOut.asNioFloat();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
                    for (int y = yOutFrom; y < yOutTo; y++) {  //along height
                        outIndices[4] = y;
                        outIndices[5] = x;
                        int baseOffsetOut = getOffsetUnsafe6(outArrayOffset, outShape, outStride, outIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 || j + patchX >= inShape3)
                                            dOut.put(outBufferIdxX + patchY * outStride2,0); //padding
                                        else {
                                            dOut.put(outBufferIdxX + patchY * outStride2,dIn.get(inBufferIdxX + patchY * inStride2));
                                        }
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape[2] || j + patchX >= inShape[3])
                                            dOut.put(outBufferIdxY + patchX * outStride3,0f); //padding
                                        else {
                                            dOut.put(outBufferIdxY + patchX * outStride3,dIn.get(inBufferIdxY + patchX * inStride3));
                                        }
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in original img
                            int j = x * strideX;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        dOut.put(outBufferIdxX + patchY * outStride2,dIn.get(inBufferIdxX + patchY * inStride2));
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        dOut.put(outBufferIdxY + patchX*outStride3,dIn.get(inBufferIdxY + patchX*inStride3));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void doDirectDouble() {
        DataBuffer dbIn = img.data();
        DataBuffer dbOut = out.data();

        int outArrayOffset = out.offset();
        int[] outShape = out.shape();
        int[] outStride = out.stride();

        int inArrayOffset = img.offset();
        int[] inShape = img.shape();
        int[] inStride = img.stride();

        int[] outIndices = new int[6];
        int[] inIndices = new int[4];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int inShape2 = inShape[2];
        final int inShape3 = inShape[3];

        final boolean padding = padHeight > 0 || padWidth > 0;

        DoubleBuffer dIn = dbIn.asNioDouble();
        DoubleBuffer dOut = dbOut.asNioDouble();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
                    for (int y = yOutFrom; y < yOutTo; y++) {  //along height
                        outIndices[4] = y;
                        outIndices[5] = x;
                        int baseOffsetOut = getOffsetUnsafe6(outArrayOffset, outShape, outStride, outIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 || j + patchX >= inShape3)
                                            dOut.put(outBufferIdxX + patchY * outStride2,0); //padding
                                        else {
                                            dOut.put(outBufferIdxX + patchY * outStride2,dIn.get(inBufferIdxX + patchY * inStride2));
                                        }
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape[2] || j + patchX >= inShape[3])
                                            dOut.put(outBufferIdxY + patchX * outStride3,0f); //padding
                                        else {
                                            dOut.put(outBufferIdxY + patchX * outStride3,dIn.get(inBufferIdxY + patchX * inStride3));
                                        }
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in original img
                            int j = x * strideX;     //index along width of first element in patch in original img
                            inIndices[2] = i;   //along height
                            inIndices[3] = j;   //along width

                            int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                            if (outStride2 <= outStride3) {
                                //Want dimension 2 (along height) in inner loop for cache reasons
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                    int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        dOut.put(outBufferIdxX + patchY * outStride2,dIn.get(inBufferIdxX + patchY * inStride2));
                                    }
                                }
                            } else {
                                //Want dimension 3 in inner loop for cache reasons
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                    int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        dOut.put(outBufferIdxY + patchX*outStride3,dIn.get(inBufferIdxY + patchX*inStride3));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
     *  normally negative indices are bad, OK here because of other checks on input indices
     *  Uses unrolled loop specifically for length 4
     */
    private static int getOffsetUnsafe4(int baseOffset, int[] shape, int[] stride, int[] indices) {
        int offset = baseOffset;
        if(shape[0] != 1) offset += indices[0] * stride[0];
        if(shape[1] != 1) offset += indices[1] * stride[1];
        if(shape[2] != 1) offset += indices[2] * stride[2];
        if(shape[3] != 1) offset += indices[3] * stride[3];
        return offset;
    }

    /**
     * A version of Shape.getOffset without checking on input for negative indices etc
     * normally negative indices are bad, OK here because of other checks on input indices
     * Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
     */
    private static int getOffsetUnsafe6(int baseOffset, int[] shape, int[] stride, int[] indices) {
        int offset = baseOffset;
        if(shape[0] != 1) offset += indices[0] * stride[0];
        if(shape[1] != 1) offset += indices[1] * stride[1];
        if(shape[4] != 1) offset += indices[4] * stride[4];
        if(shape[5] != 1) offset += indices[5] * stride[5];
        return offset;
    }

    @Override
    public INDArray invokeBlocking() {
        invokeAsync();
        return blockUntilComplete();
    }

    @Override
    public void invokeAsync() {
        future = TaskExecutorProvider.getTaskExecutor().executeAsync(this);
    }

    @Override
    public INDArray blockUntilComplete() {
        try{
            future.get();
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
        if(subTasks != null){
            //Callable execution
            for(CPUIm2ColTask task : subTasks) {
                task.blockUntilComplete();
            }
        }
        return out;
    }
}
