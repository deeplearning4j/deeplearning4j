package org.nd4j.linalg.api.parallel.bufferops.impl.transform;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.bufferops.TransformDataBufferAction;
import org.nd4j.linalg.factory.Nd4j;

public class AddOpDataBufferAction extends TransformDataBufferAction {

    public AddOpDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z,
                                 int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        super(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public AddOpDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
        super(op, tadIdx, tadDim, threshold, x, y, z);
    }

    @Override
    public void doTask() {
        //Task: Z = X+Y
        if (x == z) {
            //X += Y
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                //can use axpy (but only for heap allocated -> need double[] or float[])
                Nd4j.getBlasWrapper().level1().axpy(n, 1.0, y, offsetY, incrY, x, offsetX, incrX);
            } else {
                //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
                ByteBuf nbbx = x.asNetty();
                ByteBuf nbby = y.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetY = 4 * offsetY;
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < 4 * n; i += 4) {
                            int ox = byteOffsetX + i;
                            nbbx.setFloat(ox, nbbx.getFloat(ox) + nbby.getFloat(byteOffsetY + i));
                        }
                    } else {
                        for (int i = 0; i < 4 * n; i += 4) {
                            int ox = byteOffsetX + i * incrX;
                            nbbx.setFloat(ox, nbbx.getFloat(ox) + nbby.getFloat(byteOffsetY + i * incrY));
                        }
                    }
                } else {
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetY = 8 * offsetY;
                    if (incrX == 1 && incrY == 1) {
                        for (int i = 0; i < 8 * n; i += 8) {
                            int ox = byteOffsetX + i;
                            nbbx.setDouble(ox, nbbx.getDouble(ox) + nbby.getDouble(byteOffsetY + i));
                        }
                    } else {
                        for (int i = 0; i < 8 * n; i += 8) {
                            int ox = byteOffsetX + i * incrX;
                            nbbx.setDouble(ox, nbbx.getDouble(ox) + nbby.getDouble(byteOffsetY + i * incrY));
                        }
                    }
                }
            }
        } else {
            //Z = X+Y
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                //Heap allocation (float[] or double[])
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    float[] zf = (float[]) z.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i] = xf[offsetX + i] + yf[offsetY + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i * incrZ] = xf[offsetX + i * incrX] + yf[offsetY + i * incrY];
                        }
                    }
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    double[] zd = (double[]) z.array();
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i] = xd[offsetX + i] + yd[offsetY + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i * incrZ] = xd[offsetX + i * incrX] + yd[offsetY + i * incrY];
                        }
                    }
                }
            } else {
                //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
                ByteBuf nbbx = x.asNetty();
                ByteBuf nbby = y.asNetty();
                ByteBuf nbbz = z.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetY = 4 * offsetY;
                    int byteOffsetZ = 4 * offsetZ;
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < 4 * n; i += 4) {
                            nbbz.setFloat(byteOffsetZ + i, nbbx.getFloat(byteOffsetX + i) + nbby.getFloat(byteOffsetY + i));
                        }
                    } else {
                        for (int i = 0; i < 4 * n; i += 4) {
                            nbbz.setFloat(byteOffsetZ + i * incrZ, nbbx.getFloat(byteOffsetX + i * incrX) + nbby.getFloat(byteOffsetY + i * incrY));
                        }
                    }
                } else {
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetY = 8 * offsetY;
                    int byteOffsetZ = 8 * offsetZ;
                    if (incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < 8 * n; i += 8) {
                            nbbz.setDouble(byteOffsetZ + i, nbbx.getDouble(byteOffsetX + i) + nbby.getDouble(byteOffsetY + i));
                        }
                    } else {
                        for (int i = 0; i < 8 * n; i += 8) {
                            nbbz.setDouble(byteOffsetZ + i * incrZ, nbbx.getDouble(byteOffsetX + i * incrX) + nbby.getDouble(byteOffsetY + i * incrY));
                        }
                    }
                }
            }
        }
    }

    @Override
    public TransformDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        return new AddOpDataBufferAction(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }
}
