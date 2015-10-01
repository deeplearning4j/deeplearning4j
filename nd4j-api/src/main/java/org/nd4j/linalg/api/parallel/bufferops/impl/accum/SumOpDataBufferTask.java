package org.nd4j.linalg.api.parallel.bufferops.impl.accum;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;

public class SumOpDataBufferTask extends AccumulationDataBufferTask {

    public SumOpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                        int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

    public SumOpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
        super(op,tadIdx,tadDim,threshold,x,y,outerTask);
    }

    @Override
    public double doTask() {
        //Task: accum = sum_i x_i
        double sum = op.zeroDouble();
        if (x.dataType() == DataBuffer.Type.FLOAT) {
            float[] xf = (float[]) x.array();
            if (incrX == 1) {
                for (int i = 0; i < n; i++) {
                    sum += xf[offsetX+i];
                }
            } else {
                for (int i = 0; i < n; i++) {
                    sum += xf[offsetX + i * incrX];
                }
            }
        } else {
            double[] xd = (double[]) x.array();
            if (incrX == 1) {
                for (int i = 0; i < n; i++) {
                    sum += xd[offsetX + i];
                }
            } else {
                for (int i = 0; i < n; i++) {
                    sum += xd[offsetX + i * incrX];
                }
            }
        }
        return (outerTask ? op.getAndSetFinalResult(sum) : sum);
    }

    @Override
    public AccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                 int incrX, int incrY, boolean outerTask) {
        return new SumOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

}
