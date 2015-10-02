package org.nd4j.linalg.api.parallel.bufferops.impl.accum;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;

public class Norm1OpDataBufferTask extends AccumulationDataBufferTask {

    public Norm1OpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                 int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

    public Norm1OpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask) {
        super(op, tadIdx, tadDim, threshold, x, y, outerTask);
    }

    @Override
    public double doTask() {
        //Task: sum_i |x_i|
        return Nd4j.getBlasWrapper().level1().asum(n, x, offsetX, incrX);
    }

    @Override
    public AccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                 int incrX, int incrY, boolean outerTask) {
        return new Norm1OpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }
}
