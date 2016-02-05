package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.tasks.Task;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.concurrent.RecursiveAction;

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
    public Void call() {
        //Callable / ExecutorService
        if(doTensorFirst) doTensorFirst(op);

        if (n > threshold) {
            //Break into subtasks
            int nSubTasks = 1 + n / threshold;  //(round up)
            subTasks = new ArrayList<>(nSubTasks);
            //break into equal sized tasks:

            int taskSize = n / nSubTasks;
            int soFar = 0;
            for (int i = 0; i < nSubTasks; i++) {
                int nInTask;
                if (i == nSubTasks - 1) {
                    //All remaining tasks (due to integer division)
                    nInTask = n - soFar;
                } else {
                    nInTask = taskSize;
                }
                int offsetXNew = offsetX + soFar * incrX;
                int offsetYNew = offsetY + soFar * incrY;
                int offsetZNew = offsetZ + soFar * incrZ;

                Task t = new CPUTransformOpAction(op, threshold, nInTask, offsetXNew, offsetYNew, offsetZNew, incrX, incrY, incrZ);
                t.invokeAsync();
                subTasks.add(t);

                soFar += nInTask;
            }
        } else {
            //Execute directly
            execute();
        }
        return null;
    }


    @Override
    protected void compute() {
        //Fork join
        if(doTensorFirst) doTensorFirst(op);

        if (n > threshold) {
            //Break into subtasks
            int nFirst = n / 2;
            RecursiveAction first = new CPUTransformOpAction(op, threshold, nFirst, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
            first.fork();

            int nSecond = n - nFirst;
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            int offsetZ2 = offsetZ + nFirst * incrZ;
            RecursiveAction second = new CPUTransformOpAction(op, threshold, nSecond, offsetX2, offsetY2, offsetZ2, incrX, incrY, incrZ);
            second.fork();

            first.join();
            second.join();
        } else {
            //Execute directly
            execute();
        }
    }

    private void execute(){
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
                ByteBuffer nbbx = x.asNio();
                ByteBuffer nbby = y.asNio();
                ByteBuffer nbbz = z.asNio();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = offsetX;
                    int byteOffsetY = offsetY;
                    int byteOffsetZ = offsetZ;
                    FloatBuffer floatBufferX = nbbx.asFloatBuffer();
                    FloatBuffer floatBufferY = nbby.asFloatBuffer();
                    FloatBuffer floatBufferZ = nbbz.asFloatBuffer();
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {

                            for (int i = 0; i < n; i ++) {
                                int xbOffset = byteOffsetX + i;
                                floatBufferX.put(xbOffset, op.op(floatBufferX.get(xbOffset), floatBufferY.get(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < n; i ++) {
                                floatBufferZ.put(byteOffsetZ + i, op.op(floatBufferX.get(byteOffsetX + i), floatBufferY.get(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i++) {
                                int xbOffset = byteOffsetX + i * incrX;
                                floatBufferX.put(xbOffset, op.op(floatBufferX.get(xbOffset), floatBufferY.get(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < 4*n; i ++) {
                                floatBufferZ.put(byteOffsetZ + i * incrZ, op.op(floatBufferX.get(byteOffsetX + i * incrX), floatBufferY.get(byteOffsetY + i * incrY)));
                            }
                        }
                    }
                } else {
                    int byteOffsetX = offsetX;
                    int byteOffsetY = offsetY;
                    int byteOffsetZ = offsetZ;
                    DoubleBuffer doubleBufferX = nbbx.asDoubleBuffer();
                    DoubleBuffer doubleBufferY = nbby.asDoubleBuffer();
                    DoubleBuffer doubleBufferZ = nbbz.asDoubleBuffer();
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i ++) {
                                int xbOffset = byteOffsetX + i;
                                doubleBufferX.put(xbOffset, op.op(doubleBufferX.get(xbOffset), doubleBufferY.get(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = 0; i < n; i ++) {
                                doubleBufferZ.put(byteOffsetZ + i, op.op(doubleBufferX.get(byteOffsetX + i), doubleBufferY.get(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i ++) {
                                int xbOffset = byteOffsetX + i * incrX;
                                doubleBufferX.put(xbOffset, op.op(doubleBufferX.get(xbOffset), doubleBufferY.get(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = 0; i < n; i++) {
                                doubleBufferZ.put(byteOffsetZ + i * incrZ, op.op(doubleBufferX.get(byteOffsetX + i * incrX), doubleBufferY.get(byteOffsetY + i * incrY)));
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
                ByteBuffer nbbx = x.asNio();
                ByteBuffer nbbz = z.asNio();
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    int byteOffsetX = offsetX;
                    int byteOffsetZ = offsetZ;
                    FloatBuffer floatBufferX = nbbx.asFloatBuffer();
                    FloatBuffer floatBufferZ = nbbz.asFloatBuffer();
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i ++) {
                                int xbOffset = byteOffsetX + i;
                                floatBufferX.put(xbOffset, op.op(floatBufferX.get(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < n; i ++) {
                                floatBufferZ.put(byteOffsetZ + i, op.op(floatBufferX.get(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i ++) {
                                int xbOffset = byteOffsetX + i * incrX;
                                floatBufferX.put(xbOffset, op.op(nbbx.getFloat(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i <  n; i++) {
                                floatBufferZ.put(byteOffsetZ + i * incrZ, op.op(floatBufferX.get(byteOffsetX + i * incrX)));
                            }
                        }
                    }
                } else {
                    //Double
                    int byteOffsetX = offsetX;
                    int byteOffsetZ = offsetZ;
                    DoubleBuffer doubleBufferX = nbbx.asDoubleBuffer();
                    DoubleBuffer doubleBufferZ = nbbz.asDoubleBuffer();
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = 0; i < n; i ++) {
                                int xbOffset = byteOffsetX + i;
                                doubleBufferX.put(xbOffset, op.op(doubleBufferX.get(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < n; i ++) {
                                doubleBufferZ.put(byteOffsetZ + i, op.op(doubleBufferX.get(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = 0; i < n; i ++) {
                                int xbOffset = byteOffsetX + i * incrX;
                                doubleBufferX.put(xbOffset, op.op(doubleBufferX.get(xbOffset)));
                            }
                        } else {
                            for (int i = 0; i < n; i ++) {
                                doubleBufferZ.put(byteOffsetZ + i * incrZ, op.op(doubleBufferX.get(byteOffsetX + i * incrX)));
                            }
                        }
                    }
                }
            }
        }
    }
}
