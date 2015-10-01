package org.nd4j.linalg.api.parallel.bufferops;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;

import java.util.concurrent.RecursiveTask;

public abstract class IndexAccumulationDataBufferTask extends RecursiveTask<Pair<Double,Integer>> {
    protected final IndexAccumulation op;
    protected final int threshold;
    protected int n;
    protected final DataBuffer x;
    protected final DataBuffer y;
    protected int offsetX;    //Data buffer offset
    protected int offsetY;
    protected int incrX;
    protected int incrY;
    protected int elementOffset;  //Starting index of the first element
    protected final boolean outerTask;

    protected final boolean doTensorFirst;
    protected INDArray ndx;
    protected INDArray ndy;
    protected int tadIdx;
    protected int tadDim;

    public IndexAccumulationDataBufferTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                           int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask){
        this.op = op;
        this.threshold = threshold;
        this.n = n;
        this.x = x;
        this.y = y;
        this.offsetX = offsetX;
        this.offsetY = offsetY;
        this.incrX = incrX;
        this.incrY = incrY;
        this.elementOffset = elementOffset;
        this.outerTask = outerTask;
        this.doTensorFirst = false;
    }

    public IndexAccumulationDataBufferTask( IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
        this.op = op;
        this.threshold = threshold;
        this.x = x.data();
        this.y = (y != null ? y.data() : null);
        this.outerTask = outerTask;
        this.ndx = x;
        this.ndy = y;
        this.tadIdx = tadIdx;
        this.tadDim = tadDim;
        this.doTensorFirst = true;
    }

    @Override
    protected Pair<Double,Integer> compute() {
        if(doTensorFirst){
            INDArray tadX = ndx.tensorAlongDimension(tadIdx,tadDim);
            INDArray tadY = (ndy!=null ? ndy.tensorAlongDimension(tadIdx,tadDim) : null);
            this.offsetX = tadX.offset();
            this.offsetY = (tadY != null ? tadY.offset() : 0);
            this.incrX = tadX.elementWiseStride();
            this.incrY = (tadY != null ? tadY.elementWiseStride() : 0);
            this.n = tadX.length();

            this.elementOffset = tadIdx*tadX.length();  //First element of this tensor has index of elementOffset in original NDArray
        }

        if(n>threshold){
            //Split task
            int nFirst = n / 2;
            IndexAccumulationDataBufferTask t1 = getSubTask(op,threshold,nFirst,x,y,offsetX,offsetY,incrX,incrY,elementOffset,false);
            t1.fork();

            int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
            int elementOffset2 = elementOffset + nFirst;
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            IndexAccumulationDataBufferTask t2 = getSubTask(op,threshold,nSecond,x,y,offsetX2,offsetY2,incrX,incrY,elementOffset2,false);
            t2.fork();

            Pair<Double,Integer> p1 = t1.join();
            Pair<Double,Integer> p2 = t2.join();

            Pair<Double,Integer> out = op.combineSubResults(p1,p2);
            if(outerTask) op.setFinalResult(out.getSecond());
            return out;
        } else {
            Pair<Double,Integer> out = doTask();
            if(outerTask) op.setFinalResult(out.getSecond());
            return out;
        }
    }

    public abstract Pair<Double,Integer> doTask();

    public abstract IndexAccumulationDataBufferTask getSubTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                                                   int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask);
}
