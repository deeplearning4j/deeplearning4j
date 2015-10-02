package org.nd4j.linalg.api.parallel.bufferops.impl.indexaccum;

import io.netty.buffer.ByteBuf;
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

    public IndexAccumulationOpDataBufferTask(IndexAccumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask) {
        super(op, tadIdx, tadDim, threshold, x, y, outerTask);
    }

    @Override
    public Pair<Double, Integer> doTask() {
        if (y != null) {
            //Task: accum = update(accum,X,Y)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    float accum = op.zeroFloat();
                    int idxAccum = -1;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xf[offsetX + i], yf[offsetY + i], i);
                            if (idxAccum == i) accum = op.op(xf[offsetX + i], yf[offsetY + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xf[offsetX + i * incrX], yf[offsetY + i * incrY], i);
                            if (idxAccum == i) accum = op.op(xf[offsetX + i * incrX], yf[offsetY + i * incrY]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    double accum = op.zeroDouble();
                    int idxAccum = -1;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xd[offsetX + i], yd[offsetY + i], i);
                            if (idxAccum == i) accum = op.op(xd[offsetX + i], yd[offsetY + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xd[offsetX + i * incrX], yd[offsetY + i * incrY], i);
                            if (idxAccum == i) accum = op.op(xd[offsetX + i * incrX], yd[offsetY + i * incrY]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
                }
            } else {
                ByteBuf nbbx = x.asNetty();
                ByteBuf nbby = y.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetY = 4 * offsetY;
                    float accum = op.zeroFloat();
                    int idxAccum = -1;
                    int idx = 0;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i += 4, idx++) {
                            float fx = nbbx.getFloat(byteOffsetX + i);
                            float fy = nbby.getFloat(byteOffsetY + i);
                            idxAccum = op.update(accum, idxAccum, fx, fy, idx);
                            if (idxAccum == idx) accum = op.op(fx, fy);
                        }
                    } else {
                        for (int i = 0; i < n; i += 4, idx++) {
                            float fx = nbbx.getFloat(byteOffsetX + i * incrX);
                            float fy = nbby.getFloat(byteOffsetY + i * incrY);
                            idxAccum = op.update(accum, idxAccum, fx, fy, idx);
                            if (idxAccum == idx) accum = op.op(fx, fy);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
                } else {
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetY = 8 * offsetY;
                    double accum = op.zeroDouble();
                    int idxAccum = -1;
                    int idx = 0;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i += 8, idx++) {
                            double dx = nbbx.getDouble(byteOffsetX + i);
                            double dy = nbby.getDouble(byteOffsetY + i);
                            idxAccum = op.update(accum, idxAccum, dx, dy, idx);
                            if (idxAccum == idx) accum = op.op(dx, dy);
                        }
                    } else {
                        for (int i = 0; i < n; i += 8, idx++) {
                            double dx = nbbx.getDouble(byteOffsetX + i * incrX);
                            double dy = nbby.getDouble(byteOffsetY + i * incrY);
                            idxAccum = op.update(accum, idxAccum, dx, dy, idx);
                            if (idxAccum == idx) accum = op.op(dx, dy);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
                }
            }
        } else {
            //Task: accum = update(accum,X)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float accum = op.zeroFloat();
                    int idxAccum = -1;
                    if (incrX == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xf[offsetX + i], i);
                            if (idxAccum == i) accum = op.op(xf[offsetX + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xf[offsetX + i * incrX], i);
                            if (idxAccum == i) accum = op.op(xf[offsetX + i * incrX]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
                } else {
                    double[] xd = (double[]) x.array();
                    double accum = op.zeroDouble();
                    int idxAccum = -1;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xd[offsetX + i], i);
                            if (idxAccum == i) accum = op.op(xd[offsetX + i]);
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            idxAccum = op.update(accum, idxAccum, xd[offsetX + i * incrX], i);
                            if (idxAccum == i) accum = op.op(xd[offsetX + i * incrX]);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
                }
            } else {
                ByteBuf nbbx = x.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    float accum = op.zeroFloat();
                    int idxAccum = -1;
                    int idx = 0;
                    if (incrX == 1) {
                        for (int i = 0; i < n; i += 4, idx++) {
                            float fx = nbbx.getFloat(byteOffsetX + i);
                            idxAccum = op.update(accum, idxAccum, fx, idx);
                            if (idxAccum == idx) accum = op.op(fx);
                        }
                    } else {
                        for (int i = 0; i < n; i += 4, idx++) {
                            float fx = nbbx.getFloat(byteOffsetX + i * incrX);
                            idxAccum = op.update(accum, idxAccum, fx, idx);
                            if (idxAccum == idx) accum = op.op(fx);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
                } else {
                    int byteOffsetX = 8 * offsetX;
                    double accum = op.zeroDouble();
                    int idxAccum = -1;
                    int idx = 0;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < n; i += 8, idx++) {
                            double dx = nbbx.getDouble(byteOffsetX + i);
                            idxAccum = op.update(accum, idxAccum, dx, idx);
                            if (idxAccum == idx) accum = op.op(dx);
                        }
                    } else {
                        for (int i = 0; i < n; i += 8, idx++) {
                            double dx = nbbx.getDouble(byteOffsetX + i * incrX);
                            idxAccum = op.update(accum, idxAccum, dx, idx);
                            if (idxAccum == idx) accum = op.op(dx);
                        }
                    }
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
                }
            }
        }
    }

    @Override
    public IndexAccumulationOpDataBufferTask getSubTask(IndexAccumulation op, int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                        int incrX, int incrY, int elementOffset, boolean outerTask) {
        return new IndexAccumulationOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
    }
}
