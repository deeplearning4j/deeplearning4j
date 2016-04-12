package org.nd4j.linalg.api.parallel.tasks.cpu.scalar;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.parallel.tasks.Task;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.concurrent.RecursiveAction;

public class CPUScalarOpAction extends BaseCPUScalarOpAction {

    /**
     * Constructor for operating on subset of NDArray
     */
    public CPUScalarOpAction(ScalarOp op, int threshold, int n, int offsetX, int offsetZ, int incrX, int incrZ) {
        super(op, threshold, n, offsetX, offsetZ, incrX, incrZ);
    }

    /**
     * Constructor for doing task on entire NDArray
     */
    public CPUScalarOpAction(ScalarOp op, int threshold) {
        super(op, threshold);
    }

    /**
     * Constructor for doing a 1d tensor first
     */
    public CPUScalarOpAction(ScalarOp op, int threshold, int tadIdx, int tadDim) {
        super(op, threshold, tadIdx, tadDim);
    }

    @Override
    public Void call() {
        //Callable / ExecutorService
        if (doTensorFirst) doTensorFirst(op);

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
                int offsetZNew = offsetZ + soFar * incrZ;

                Task t = new CPUScalarOpAction(op, threshold, nInTask, offsetXNew, offsetZNew, incrX, incrZ);
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
        if (doTensorFirst)
            doTensorFirst(op);

        if (n > threshold) {
            //Break into subtasks:
            int nFirst = n / 2;
            RecursiveAction first = new CPUScalarOpAction(op, threshold, nFirst, offsetX, offsetZ, incrX, incrZ);
            first.fork();

            int nSecond = n - nFirst;
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetZ2 = offsetZ + nFirst * incrZ;
            RecursiveAction second = new CPUScalarOpAction(op, threshold, nSecond, offsetX2, offsetZ2, incrX, incrZ);
            second.fork();

            first.join();
            second.join();
        } else {
            //Execute directly
            execute();
        }
    }

    private void execute() {
        if (doTensorFirst)
            doTensorFirst(op);

        DataBuffer x = op.x().data();
        DataBuffer z = op.z().data();

        //Task: Z = Op(X)
        if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
            //Heap allocation (float[] or double[])
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
                if (incrX == 1 && incrZ == 1) {
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
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                java.nio.FloatBuffer xFloatBuffer = x.asNioFloat();
                java.nio.FloatBuffer zFloatBuffer = z.asNioFloat();
                int byteOffsetX = 0;
                int byteOffsetZ = 0;
                if (incrX == 1 && (x == z || incrZ == 1)) {
                    if (x == z) {
                        for (int i = 0; i < n; i ++) {
                            int xbIdx = byteOffsetX + i;
                            xFloatBuffer.put(i,op.op(xFloatBuffer.get(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            zFloatBuffer.put(byteOffsetZ + i,op.op(xFloatBuffer.get(byteOffsetX + i)));
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = 0; i < n; i ++) {
                            int xbIdx = byteOffsetX + i * incrX;
                            xFloatBuffer.put(xbIdx,op.op(xFloatBuffer.get(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < n; i ++) {
                            zFloatBuffer.put(byteOffsetZ + i * incrZ,op.op(xFloatBuffer.get(byteOffsetX + i * incrX)));
                        }
                    }
                }
            }


            else {
                int byteOffsetX = offsetX;
                int byteOffsetZ = offsetZ;
                DoubleBuffer xDoubleBuffer = x.asNioDouble();
                DoubleBuffer zDoubleBuffer = z.asNioDouble();
                if (incrX == 1 && (x == z || incrZ == 1)) {
                    if (x == z) {
                        for (int i = 0; i < n; i ++) {
                            int xbIdx = i;
                            xDoubleBuffer.put(xbIdx,op.op(xDoubleBuffer.get(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < n; i ++) {
                            zDoubleBuffer.put(i,op.op(xDoubleBuffer.get(i)));
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = 0; i < n; i++) {
                            int xbIdx =  i * incrX;
                            xDoubleBuffer.put(xbIdx,op.op(xDoubleBuffer.get(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < n; i ++) {
                            zDoubleBuffer.put(i * incrZ,op.op(xDoubleBuffer.get(i * incrX)));
                        }
                    }
                }
            }
        }
    }
}
