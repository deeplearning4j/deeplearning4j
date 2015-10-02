package org.nd4j.linalg.api.parallel.bufferops.impl.transform;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.bufferops.TransformDataBufferAction;


public class TransformOpDataBufferAction extends TransformDataBufferAction {
    public TransformOpDataBufferAction(TransformOp op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        super(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public TransformOpDataBufferAction(TransformOp op, int tadIdx, int tadDim, int threshold, INDArray x, INDArray y, INDArray z) {
        super(op, tadIdx, tadDim, threshold, x, y, z);
    }

    @Override
    public void doTask() {
        if (y != null) {
            //Task: Z = Op(X,Y)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                //Heap allocation: float[] or double[]
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i;
                                xf[xIdx] = op.op(xf[xIdx], yf[offsetY + i]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zf[offsetZ + i] = op.op(xf[offsetX + i], yf[offsetY + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i * incrX;
                                xf[xIdx] = op.op(xf[xIdx], yf[offsetY + i * incrY]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zf[offsetZ + i * incrZ] = op.op(xf[offsetX + i * incrX], yf[offsetY + i * incrY]);
                            }
                        }
                    }
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i;
                                xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zd[offsetZ + i] = op.op(xd[offsetX + i], yd[offsetY + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i * incrX;
                                xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i * incrY]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zd[offsetZ + i * incrZ] = op.op(xd[offsetX + i * incrX], yd[offsetY + i * incrY]);
                            }
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
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i += 4) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < n; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i), nbby.getFloat(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i += 4) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < n; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i * incrZ, op.op(nbbx.getFloat(byteOffsetX + i * incrX), nbby.getFloat(byteOffsetY + i * incrY)));
                            }
                        }
                    }
                } else {
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetY = 8 * offsetY;
                    int byteOffsetZ = 8 * offsetZ;
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i += 8) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i), nbby.getDouble(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i += 8) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i * incrZ, op.op(nbbx.getDouble(byteOffsetX + i * incrX), nbby.getDouble(byteOffsetY + i * incrY)));
                            }
                        }
                    }
                }
            }
        } else {
            //Task: Z = Op(X)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                //Heap allocation: float[] or double[]
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i;
                                xf[xIdx] = op.op(xf[xIdx]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zf[offsetZ + i] = op.op(xf[offsetX + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i * incrX;
                                xf[xIdx] = op.op(xf[xIdx]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zf[offsetZ + i * incrZ] = op.op(xf[offsetX + i * incrX]);
                            }
                        }
                    }
                } else {
                    double[] xd = (double[]) x.array();
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i;
                                xd[xIdx] = op.op(xd[xIdx]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zd[offsetZ + i] = op.op(xd[offsetX + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xIdx = offsetX + i * incrX;
                                xd[xIdx] = op.op(xd[xIdx]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = 0; i < n; i++) {
                                zd[offsetZ + i * incrZ] = op.op(xd[offsetX + i * incrX]);
                            }
                        }
                    }
                }
            } else {
                //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
                ByteBuf nbbx = x.asNetty();
                ByteBuf nbbz = z.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetZ = 4 * offsetZ;
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i += 4) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < n; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i += 4) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < n; i++) {
                                nbbz.setFloat(byteOffsetZ + i * incrZ, op.op(nbbx.getFloat(byteOffsetX + i * incrX)));
                            }
                        }
                    }
                } else {
                    //Double
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetZ = 8 * offsetZ;
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i += 8) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i += 8) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i * incrZ, op.op(nbbx.getDouble(byteOffsetX + i * incrX)));
                            }
                        }
                    }
                }
            }
        }
    }

    @Override
    public TransformDataBufferAction getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        return new TransformOpDataBufferAction(op, threshold, n, x, y, z, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }
}
