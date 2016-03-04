package org.nd4j.linalg.api.parallel.tasks.cpu.misc;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveTask;

/**
 *
 * Parallel Col2Im implementation
 * @author Alex Black
 */
public class CPUCol2ImTask extends RecursiveTask<INDArray> implements Task<INDArray> {
    protected Future<INDArray> future;
    protected List<CPUCol2ImTask> subTasks;    //For callable execution

    protected final INDArray col;
    protected INDArray imgOut;
    protected final int kernelHeight;
    protected final int kernelWidth;
    protected final int strideY;
    protected final int strideX;
    protected final int padHeight;
    protected final int padWidth;
    protected final int imgHeight;
    protected final int imgWidth;
    protected final int parallelThreshold;

    protected final int exampleFrom;
    protected final int exampleTo;
    protected final int depthFrom;
    protected final int depthTo;

    public CPUCol2ImTask(INDArray col, int strideY, int strideX, int padHeight, int padWidth, int imgHeight, int imgWidth, int parallelThreshold) {
        this(col, getNewOutputArray(col, imgHeight, imgWidth),
                strideY, strideX, padHeight, padWidth,
                imgHeight, imgWidth,
                0, col.size(0), //example ranges
                0, col.size(1), //depth ranges
                parallelThreshold);
        //NOTE: Ranges above are [from,to) i.e., exclusive of to, inclusive of from
    }

    public CPUCol2ImTask(INDArray col, INDArray imgOut, int strideY, int strideX, int padHeight, int padWidth, int imgHeight, int imgWidth,
                         int exampleFrom, int exampleTo, int depthFrom, int depthTo, int parallelThreshold) {
        this.col = col;
        this.imgOut = imgOut;
        this.kernelHeight = col.size(2);
        this.kernelWidth = col.size(3);
        this.strideY = strideY;
        this.strideX = strideX;
        this.padHeight = padHeight;
        this.padWidth = padWidth;
        this.imgHeight = imgHeight;
        this.imgWidth = imgWidth;
        this.parallelThreshold = parallelThreshold;

        this.exampleFrom = exampleFrom;
        this.exampleTo = exampleTo;
        this.depthFrom = depthFrom;
        this.depthTo = depthTo;
    }


    private static INDArray getNewOutputArray(INDArray col, int imgHeight, int imgWidth) {
        //number of images
        int n = col.size(0);
        //number of columns
        int c = col.size(1);

        return Nd4j.create(n, c, imgHeight, imgWidth);
    }


    @Override
    protected INDArray compute() {
        //Fork join
        splitOrExecute(true);
        return imgOut;
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

            CPUCol2ImTask first;
            CPUCol2ImTask second;
            int temp;
            if ((temp = exampleTo - exampleFrom) > 1) { //exampleTo is exclusive -> single example has to-from=1
                int countFirst = temp / 2;

                first = new CPUCol2ImTask(col, imgOut, strideY, strideX, padHeight, padWidth, imgHeight, imgWidth,
                        exampleFrom, exampleFrom + countFirst,   //If countFirst=1, then want want to=from+1 exclusive, i.e., to=from inclusive
                        depthFrom, depthTo, parallelThreshold);
                if( forkJoin ) first.fork();
                else{
                    first.invokeAsync();
                    subTasks.add(first);
                }

                second = new CPUCol2ImTask(col, imgOut, strideY, strideX, padHeight, padWidth, imgHeight, imgWidth,
                        exampleFrom + countFirst, exampleTo,
                        depthFrom, depthTo, parallelThreshold);
                if( forkJoin ) second.fork();
                else{
                    second.invokeAsync();
                    subTasks.add(second);
                }

            } else if ((temp = depthTo - depthFrom) > 1) {
                //Split on depth
                int countFirst = temp / 2;
                first = new CPUCol2ImTask(col, imgOut, strideY, strideX, padHeight, padWidth, imgHeight, imgWidth,
                        exampleFrom, exampleTo, depthFrom, depthFrom + countFirst, parallelThreshold);
                first.fork();

                second = new CPUCol2ImTask(col, imgOut, strideY, strideX, padHeight, padWidth, imgHeight, imgWidth,
                        exampleFrom, exampleTo, depthFrom + countFirst, depthTo, parallelThreshold);
                second.fork();

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
        return (exampleTo - exampleFrom) * (depthTo - depthFrom) * col.size(4) * col.size(5) * kernelHeight * kernelWidth;
    }

    private void execute() {
        DataBuffer dbIn = col.data();

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

    private void doHeapFloat(){
        DataBuffer dbCol = col.data();
        DataBuffer dbOut = imgOut.data();

        int outArrayOffset = 0;
        int[] outShape = imgOut.shape();
        int[] outStride = imgOut.stride();

        int inOffset = 0;
        int[] inShape = col.shape();
        int[] inStride = col.stride();

        int[] outIndices = new int[4];
        int[] inIndices = new int[6];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int outShape2 = outShape[2];
        final int outShape3 = outShape[3];

        final int yOutTo = inShape[4];
        final int xOutTo = inShape[5];


        final boolean padding = padHeight > 0 || padWidth > 0;

        float[] fIn = (float[]) dbCol.array();
        float[] fOut = (float[]) dbOut.array();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = 0; x < xOutTo; x++) {  //Patch number along width
                    for (int y = 0; y < yOutTo; y++) {  //Patch number along height
                        inIndices[4] = y;   //patch number (along height)
                        inIndices[5] = x;   //patch number (along width)
                        int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            outIndices[2] = i;  //along height
                            outIndices[3] = j;  //along width

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    if( j + patchX < 0 || j + patchX >= outShape3 ) continue;

                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || i + patchY >= outShape2 ) continue;
                                        fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    if(i + patchY < 0 || i + patchY >= outShape2) continue;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (j + patchX < 0 || j + patchX >= outShape3) continue;
                                        fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in output img
                            int j = x * strideX;     //index along width of first element in patch in output img

                            outIndices[2] = i;
                            outIndices[3] = j;

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
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
        DataBuffer dbCol = col.data();
        DataBuffer dbOut = imgOut.data();

        int outArrayOffset = imgOut.offset();
        int[] outShape = imgOut.shape();
        int[] outStride = imgOut.stride();

        int inOffset = col.offset();
        int[] inShape = col.shape();
        int[] inStride = col.stride();

        int[] outIndices = new int[4];
        int[] inIndices = new int[6];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int outShape2 = outShape[2];
        final int outShape3 = outShape[3];

        final int yOutTo = inShape[4];
        final int xOutTo = inShape[5];


        final boolean padding = padHeight > 0 || padWidth > 0;

        double[] dIn = (double[]) dbCol.array();
        double[] dOut = (double[]) dbOut.array();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = 0; x < xOutTo; x++) {  //Patch number along width
                    for (int y = 0; y < yOutTo; y++) {  //Patch number along height
                        inIndices[4] = y;   //patch number (along height)
                        inIndices[5] = x;   //patch number (along width)
                        int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            outIndices[2] = i;  //along height
                            outIndices[3] = j;  //along width

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    if( j + patchX < 0 || j + patchX >= outShape3 ) continue;
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || i + patchY >= outShape2 ) continue;
                                        dOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                dIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    if(i + patchY < 0 || i + patchY >= outShape2) continue;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (j + patchX < 0 || j + patchX >= outShape3) continue;
                                        dOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                dIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in output img
                            int j = x * strideX;     //index along width of first element in patch in output img

                            outIndices[2] = i;
                            outIndices[3] = j;

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        dOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                dIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        dOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                dIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
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
        DataBuffer dbCol = col.data();
        DataBuffer dbOut = imgOut.data();

        int outArrayOffset = 0;
        int[] outShape = imgOut.shape();
        int[] outStride = imgOut.stride();

        int inOffset = 0;
        int[] inShape = col.shape();
        int[] inStride = col.stride();

        int[] outIndices = new int[4];
        int[] inIndices = new int[6];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int outShape2 = outShape[2];
        final int outShape3 = outShape[3];

        final int yOutTo = inShape[4];
        final int xOutTo = inShape[5];


        final boolean padding = padHeight > 0 || padWidth > 0;

        FloatBuffer fIn = dbCol.asNioFloat();
        FloatBuffer fOut = dbOut.asNioFloat();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = 0; x < xOutTo; x++) {  //Patch number along width
                    for (int y = 0; y < yOutTo; y++) {  //Patch number along height
                        inIndices[4] = y;   //patch number (along height)
                        inIndices[5] = x;   //patch number (along width)
                        int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            outIndices[2] = i;  //along height
                            outIndices[3] = j;  //along width

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    if( j + patchX < 0 || j + patchX >= outShape3 ) continue;

                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || i + patchY >= outShape2 ) continue;
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                                fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3)
                                                        +
                                                        fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    if(i + patchY < 0 || i + patchY >= outShape2) continue;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (j + patchX < 0 || j + patchX >= outShape3) continue;
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                                fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3) + fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in output img
                            int j = x * strideX;     //index along width of first element in patch in output img

                            outIndices[2] = i;
                            outIndices[3] = j;

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                               fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3) + fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                                fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3) + fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void doDirectDouble(){
        DataBuffer dbCol = col.data();
        DataBuffer dbOut = imgOut.data();

        int outArrayOffset = 0;
        int[] outShape = imgOut.shape();
        int[] outStride = imgOut.stride();

        int inOffset = 0;
        int[] inShape = col.shape();
        int[] inStride = col.stride();

        int[] outIndices = new int[4];
        int[] inIndices = new int[6];

        final int inStride2 = inStride[2];
        final int inStride3 = inStride[3];
        final int outStride2 = outStride[2];
        final int outStride3 = outStride[3];
        final int outShape2 = outShape[2];
        final int outShape3 = outShape[3];

        final int yOutTo = inShape[4];
        final int xOutTo = inShape[5];


        final boolean padding = padHeight > 0 || padWidth > 0;

        DoubleBuffer fIn = dbCol.asNioDouble();
        DoubleBuffer fOut = dbOut.asNioDouble();

        for (int ex = exampleFrom; ex < exampleTo; ex++) {
            for (int d = depthFrom; d < depthTo; d++) {
                inIndices[0] = ex;
                inIndices[1] = d;
                outIndices[0] = ex;
                outIndices[1] = d;

                for (int x = 0; x < xOutTo; x++) {  //Patch number along width
                    for (int y = 0; y < yOutTo; y++) {  //Patch number along height
                        inIndices[4] = y;   //patch number (along height)
                        inIndices[5] = x;   //patch number (along width)
                        int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

                        if(padding){
                            int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                            int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                            outIndices[2] = i;  //along height
                            outIndices[3] = j;  //along width

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    if( j + patchX < 0 || j + patchX >= outShape3 ) continue;

                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if (i + patchY < 0 || i + patchY >= outShape2 ) continue;
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                                fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3)
                                                        +
                                                        fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    if(i + patchY < 0 || i + patchY >= outShape2) continue;
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if (j + patchX < 0 || j + patchX >= outShape3) continue;
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                                fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3) + fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
                                    }
                                }
                            }
                        } else {
                            //No padding
                            int i = y * strideY;    //index along height of first element of patch in output img
                            int j = x * strideX;     //index along width of first element in patch in output img

                            outIndices[2] = i;
                            outIndices[3] = j;

                            int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                            if (inStride2 <= inStride3) {
                                //Want dimension 2 (along height) in inner loop for cache efficiency
                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                                fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3) + fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
                                    }
                                }
                            } else {
                                //Want dimension 3 (along width) in inner loop for cache efficiency
                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        fOut.put(baseOffsetOut + patchY * outStride2 + patchX * outStride3,
                                                fOut.get(baseOffsetOut + patchY * outStride2 + patchX * outStride3) + fIn.get(baseOffsetIn + patchY * inStride2 + patchX * inStride3));
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

    /** A version of Shape.getOffset without checking on input for negative indices etc
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
        }catch(Exception e){
            throw new RuntimeException(e);
        }
        if(subTasks != null){
            //Callable execution
            for(CPUCol2ImTask task : subTasks){
                task.blockUntilComplete();
            }
        }
        return imgOut;
    }
}
