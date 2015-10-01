package org.nd4j.linalg.api.parallel.bufferops.impl.accum;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;

public class AccumulationOpDataBufferTask extends AccumulationDataBufferTask {

    public AccumulationOpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                        int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

    public AccumulationOpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
        super(op,tadIdx,tadDim,threshold,x,y,outerTask);
    }

    @Override
    public double doTask() {
        if (y != null) {
            //Task: accum = update(accum,X,Y)
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float[] xf = (float[]) x.array();
                float[] yf = (float[]) y.array();
                float accum = op.zeroFloat();
                if (incrX == 1 && incrY == 1) {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xf[offsetX + i], yf[offsetY + i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xf[offsetX + i * incrX], yf[offsetY + i * incrY]);
                    }
                }
                return (outerTask ? op.getAndSetFinalResult(accum) : accum);
            } else {
                double[] xd = (double[]) x.array();
                double[] yd = (double[]) y.array();
                double accum = op.zeroDouble();
                if (incrX == 1 && incrY == 1) {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xd[offsetX + i], yd[offsetY + i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xd[offsetX + i * incrX], yd[offsetY + i * incrY]);
                    }
                }
                return (outerTask ? op.getAndSetFinalResult(accum) : accum);
            }
        } else {
            //Task: accum = update(accum,X)
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float[] xf = (float[]) x.array();
                float accum = op.zeroFloat();
                if (incrX == 1) {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xf[offsetX + i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xf[offsetX + i * incrX]);
                    }
                }
                return (outerTask ? op.getAndSetFinalResult(accum) : accum);
            } else {
                double[] xd = (double[]) x.array();
                double accum = op.zeroDouble();
                if (incrX == 1) {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xd[offsetX + i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        accum = op.update(accum, xd[offsetX + i * incrX]);
                    }
                }
                return (outerTask ? op.getAndSetFinalResult(accum) : accum);
            }
        }
    }

    @Override
    public AccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                     int incrX, int incrY, boolean outerTask) {
        return new AccumulationOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }
}
