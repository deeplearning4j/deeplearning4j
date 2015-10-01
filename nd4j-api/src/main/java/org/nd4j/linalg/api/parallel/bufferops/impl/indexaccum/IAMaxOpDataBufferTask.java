package org.nd4j.linalg.api.parallel.bufferops.impl.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.parallel.bufferops.IndexAccumulationDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;

public class IAMaxOpDataBufferTask extends IndexAccumulationDataBufferTask {

    public IAMaxOpDataBufferTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                             int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
    }

    public IAMaxOpDataBufferTask(IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
        super(op,tadIdx,tadDim,threshold,x,y,outerTask);
    }

    @Override
    public Pair<Double, Integer> doTask() {
        int idxAccum = Nd4j.getBlasWrapper().level1().iamax(n, x, offsetX, incrX);
        double value = x.getDouble(offsetX + idxAccum * incrX);
        idxAccum += this.elementOffset; //idxAccum gives relative to start of segment, not relative to start of NDArray
        if(outerTask) op.setFinalResult(idxAccum);
        return new Pair<>(value,idxAccum);
    }

    @Override
    public IndexAccumulationDataBufferTask getSubTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask) {
        return new IAMaxOpDataBufferTask(op,threshold,n,x,y,offsetX,offsetY,incrX,incrY,elementOffset,outerTask);
    }
}
