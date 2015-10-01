package org.nd4j.linalg.api.parallel.bufferops.impl.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.parallel.bufferops.IndexAccumulationDataBufferTask;

public class IndexAccumulationOpDataBufferTask extends IndexAccumulationDataBufferTask {

    public IndexAccumulationOpDataBufferTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                             int offsetX, int offsetY, int incrX, int incrY, int elementOffset, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
    }

    public IndexAccumulationOpDataBufferTask(IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask){
        super(op,tadIdx,tadDim,threshold,x,y,outerTask);
    }

    @Override
    public Pair<Double,Integer> doTask() {
        if (y != null) {
            //Task: accum = update(accum,X,Y)
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float[] xf = (float[]) x.array();
                float[] yf = (float[]) y.array();
                float accum = op.zeroFloat();
                int idxAccum = -1;
                if (incrX == 1 && incrY == 1) {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xf[offsetX+i],yf[offsetY+i],i);
                        if(idxAccum==i) accum = op.op(xf[offsetX+i],yf[offsetY+i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xf[offsetX+i*incrX],yf[offsetY+i*incrY],i);
                        if(idxAccum==i) accum = op.op(xf[offsetX+i*incrX],yf[offsetY+i*incrY]);
                    }
                }
                int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                if(outerTask) op.setFinalResult(finalIdx);
                return new Pair<>((double)accum,finalIdx);
            } else {
                double[] xd = (double[]) x.array();
                double[] yd = (double[]) y.array();
                double accum = op.zeroDouble();
                int idxAccum = -1;
                if (incrX == 1 && incrY == 1) {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xd[offsetX+i],yd[offsetY+i],i);
                        if(idxAccum==i) accum = op.op(xd[offsetX+i],yd[offsetY+i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xd[offsetX+i*incrX],yd[offsetY+i*incrY],i);
                        if(idxAccum==i) accum = op.op(xd[offsetX+i*incrX],yd[offsetY+i*incrY]);
                    }
                }
                int finalIdx = idxAccum + elementOffset;
                if(outerTask) op.setFinalResult(finalIdx);
                return new Pair<>(accum,finalIdx);
            }
        } else {
            //Task: accum = update(accum,X)
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float[] xf = (float[]) x.array();
                float accum = op.zeroFloat();
                int idxAccum = -1;
                if (incrX == 1) {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xf[offsetX+i],i);
                        if(idxAccum==i) accum = op.op(xf[offsetX+i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xf[offsetX+i*incrX],i);
                        if(idxAccum==i) accum = op.op(xf[offsetX+i*incrX]);
                    }
                }
                int finalIdx = idxAccum + elementOffset;
                if(outerTask) op.setFinalResult(finalIdx);
                return new Pair<>((double)accum,finalIdx);
            } else {
                double[] xd = (double[]) x.array();
                double accum = op.zeroDouble();
                int idxAccum = -1;
                if (incrX == 1 && incrY == 1) {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xd[offsetX+i],i);
                        if(idxAccum==i) accum = op.op(xd[offsetX+i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        idxAccum = op.update(accum,idxAccum,xd[offsetX+i*incrX],i);
                        if(idxAccum==i) accum = op.op(xd[offsetX+i*incrX]);
                    }
                }
                int finalIdx = idxAccum + elementOffset;
                if(outerTask) op.setFinalResult(finalIdx);
                return new Pair<>(accum,finalIdx);
            }
        }
    }

    @Override
    public IndexAccumulationOpDataBufferTask getSubTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                          int incrX, int incrY, int elementOffset, boolean outerTask) {
        return new IndexAccumulationOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
    }
}
