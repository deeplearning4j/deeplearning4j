package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;

import java.util.concurrent.RecursiveTask;

/**
 * Created by Alex on 1/10/2015.
 */
@AllArgsConstructor
public abstract class IndexAccumulationDataBufferTask extends RecursiveTask<Pair<Double,Integer>> {
    protected final IndexAccumulation op;
    protected final int threshold;
    protected final int n;
    protected final DataBuffer x;
    protected final DataBuffer y;
    protected final int offsetX;    //Data buffer offset
    protected final int offsetY;
    protected final int incrX;
    protected final int incrY;
    protected final int elementOffset;  //Starting index of the first element
    protected final boolean outerTask;

    public IndexAccumulationDataBufferTask( IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
        this.op = op;
        this.threshold = threshold;
        this.outerTask = outerTask;
        INDArray tadX = x.tensorAlongDimension(tadIdx,tadDim);
        INDArray tadY = (y!=null ? y.tensorAlongDimension(tadIdx,tadDim) : null);
        this.x = x.data();
        this.y = (y != null ? y.data() : null);
        this.offsetX = tadX.offset();
        this.offsetY = (tadY != null ? tadY.offset() : 0);
        this.incrX = tadX.elementWiseStride();
        this.incrY = (tadY != null ? tadY.elementWiseStride() : 0);
        this.n = tadX.length();

        this.elementOffset = tadIdx*tadX.length();  //First element of this tensor has index of elementOffset in original NDArray
    }

    @Override
    protected Pair<Double,Integer> compute() {
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
