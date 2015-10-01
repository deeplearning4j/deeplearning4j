package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;

import java.util.concurrent.RecursiveTask;

@AllArgsConstructor
public abstract class AccumulationDataBufferTask extends RecursiveTask<Double> {
    protected final Accumulation op;
    protected final int threshold;
    protected final int n;
    protected final DataBuffer x;
    protected final DataBuffer y;
    protected final int offsetX;
    protected final int offsetY;
    protected final int incrX;
    protected final int incrY;
    protected final boolean outerTask;

    public AccumulationDataBufferTask( Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
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
    }

    @Override
    protected Double compute() {
        if (n > threshold) {
            //Split task
            int nFirst = n / 2;
            AccumulationDataBufferTask t1 = getSubTask(threshold, nFirst, x, y, offsetX, offsetY, incrX, incrY, false);
            t1.fork();

            int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            AccumulationDataBufferTask t2 = getSubTask(threshold, nSecond, x, y, offsetX2, offsetY2, incrX, incrY, false);
            t2.fork();

            double first = t1.join();
            double second = t2.join();
            double preFinalResult = op.combineSubResults(first, second);
            if (outerTask) return op.getAndSetFinalResult(preFinalResult);
            else return preFinalResult;
        } else {
            return doTask();
        }
    }

    public abstract double doTask();

    public abstract AccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y,
                                                              int offsetX, int offsetY, int incrX, int incrY, boolean outerTask);
}
