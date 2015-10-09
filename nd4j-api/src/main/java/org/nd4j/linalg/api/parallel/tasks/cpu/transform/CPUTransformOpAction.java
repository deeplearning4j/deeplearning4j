package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;

import java.util.ArrayList;

public class CPUTransformOpAction extends BaseCPUTransformOpAction {

    public CPUTransformOpAction(TransformOp op, int threshold, int n, int offsetX, int offsetY, int offsetZ,
                                int incrX, int incrY, int incrZ){
        super(op, threshold, n, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public CPUTransformOpAction(TransformOp op, int threshold){
        super(op,threshold);
    }

    /** Constructor for doing a 1d TAD first */
    public CPUTransformOpAction(TransformOp op, int threshold, int tadIdx, int tadDim){
        super(op,threshold,tadIdx,tadDim);
    }

    @Override
    public void invokeAsync() {
        if(n > threshold){
            //Break into subtasks
            int nSubTasks = 1 + n / threshold;  //(round up)
            subTasks = new ArrayList<>(nSubTasks);
            //break into equal sized tasks:

            int taskSize = n / nSubTasks;
            int soFar = 0;
            for( int i=0; i<nSubTasks; i++ ){
                int nInTask;
                if(i==nSubTasks-1){
                    //All remaining tasks (due to integer division)
                    nInTask = n - soFar;
                } else {
                    nInTask = taskSize;
                }
                int offsetXNew = offsetX + soFar*incrX;
                int offsetYNew = offsetY + soFar*incrY;
                int offsetZNew = offsetZ + soFar*incrZ;

                Task t = new CPUTransformOpAction(op,threshold,nInTask,offsetXNew,offsetYNew,offsetZNew,incrX,incrY,incrZ);
                t.invokeAsync();
                subTasks.add(t);

                soFar += nInTask;
            }
        } else {
            //Execute directly
            future = TaskExecutorProvider.getTaskExecutor().executeAsync(this);
        }
    }

    @Override
    public Void call() {
        DataBuffer x = op.x().data();
        DataBuffer y = (op.y() != null ? op.y().data() : null);
        DataBuffer z = op.z().data();


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
                            for (int i = 0; i < 4*n; i += 4) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < 4*n; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i), nbby.getFloat(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < 4*n; i += 4) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < 4*n; i += 4) {
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
                            for (int i = 0; i < 8*n; i += 8) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < 8*n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i), nbby.getDouble(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < 8*n; i += 8) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < 8*n; i += 8) {
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
                            for (int i = 0; i < 4*n; i += 4) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < 4*n; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < 4*n; i += 4) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < 4*n; i++) {
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
                            for (int i = 0; i < 8*n; i += 8) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < 8*n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < 8*n; i += 8) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < 8*n; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i * incrZ, op.op(nbbx.getDouble(byteOffsetX + i * incrX)));
                            }
                        }
                    }
                }
            }
        }

        return null;
    }
}
