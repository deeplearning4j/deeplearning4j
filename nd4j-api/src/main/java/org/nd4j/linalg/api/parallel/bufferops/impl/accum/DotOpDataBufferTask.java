package org.nd4j.linalg.api.parallel.bufferops.impl.accum;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;

public class DotOpDataBufferTask extends AccumulationDataBufferTask {

    public DotOpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                               int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

    public DotOpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
        super(op,tadIdx,tadDim,threshold,x,y,outerTask);
    }

    @Override
    public double doTask() {
        //Task: dotProduct(x,y)
        return Nd4j.getBlasWrapper().level1().dot(n,x,offsetX,incrX,y,offsetY,incrY);
    }

    @Override
    public AccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                     int incrX, int incrY, boolean outerTask) {
        return new DotOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }
}
