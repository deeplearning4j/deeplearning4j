package org.nd4j.linalg.api.parallel.bufferops.impl.accum;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;

public class AccumulationOpDataBufferTask extends AccumulationDataBufferTask {

    public AccumulationOpDataBufferTask(Accumulation op, int threshold, int n, DataBuffer x, DataBuffer y,
                                        int offsetX, int offsetY, int incrX, int incrY, boolean outerTask) {
        super(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }

    public AccumulationOpDataBufferTask(Accumulation op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, boolean outerTask) {
        super(op, tadIdx, tadDim, threshold, x, y, outerTask);
    }

    @Override
    public double doTask() {
        if (y != null) {
            //Task: accum = update(accum,X,Y)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                //Heap allocation: float[] or double[]
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
                //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
                ByteBuf nbbx = x.asNetty();
                ByteBuf nbby = y.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetY = 4 * offsetY;
                    float accum = op.zeroFloat();
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < 4 * n; i += 4) {
                            accum = op.update(accum, nbbx.getFloat(byteOffsetX + i), nbby.getFloat(byteOffsetY + i));
                        }
                    } else {
                        for (int i = 0; i < 4 * n; i += 4) {
                            accum = op.update(accum, nbbx.getFloat(byteOffsetX + i * incrX), nbby.getFloat(byteOffsetY + i * incrY));
                        }
                    }
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
                } else {
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetY = 8 * offsetY;
                    double accum = op.zeroDouble();
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < 8 * n; i += 8) {
                            accum = op.update(accum, nbbx.getDouble(byteOffsetX + i), nbby.getDouble(byteOffsetY + i));
                        }
                    } else {
                        for (int i = 0; i < 8 * n; i += 8) {
                            accum = op.update(accum, nbbx.getDouble(byteOffsetX + i * incrX), nbby.getDouble(byteOffsetY + i * incrY));
                        }
                    }
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
                }
            }
        } else {
            //Task: accum = update(accum,X)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
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
            } else {
                //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
                ByteBuf nbbx = x.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    float accum = op.zeroFloat();
                    if (incrX == 1) {
                        for (int i = 0; i < 4 * n; i += 4) {
                            accum = op.update(accum, nbbx.getFloat(byteOffsetX + i));
                        }
                    } else {
                        for (int i = 0; i < 4 * n; i += 4) {
                            accum = op.update(accum, nbbx.getFloat(byteOffsetX + i * incrX));
                        }
                    }
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
                } else {
                    int byteOffsetX = 8 * offsetX;
                    double accum = op.zeroDouble();
                    if (incrX == 1) {
                        for (int i = 0; i < 8 * n; i += 8) {
                            accum = op.update(accum, nbbx.getDouble(byteOffsetX + i));
                        }
                    } else {
                        for (int i = 0; i < 8 * n; i += 8) {
                            accum = op.update(accum, nbbx.getDouble(byteOffsetX + i * incrX));
                        }
                    }
                    return (outerTask ? op.getAndSetFinalResult(accum) : accum);
                }
            }
        }
    }

    @Override
    public AccumulationDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, int offsetX, int offsetY,
                                                 int incrX, int incrY, boolean outerTask) {
        return new AccumulationOpDataBufferTask(op, threshold, n, x, y, offsetX, offsetY, incrX, incrY, outerTask);
    }
}
