package org.nd4j.linalg.api.parallel.tasks.cpu.scalar;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;

import java.util.ArrayList;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

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
        if (doTensorFirst) doTensorFirst(op);

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
        if (doTensorFirst) doTensorFirst(op);

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
            ByteBuf nbbx = x.asNetty();
            ByteBuf nbbz = z.asNetty();
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                int byteOffsetX = 4 * offsetX;
                int byteOffsetZ = 4 * offsetZ;
                if (incrX == 1 && (x == z || incrZ == 1)) {
                    if (x == z) {
                        for (int i = 0; i < 4 * n; i += 4) {
                            int xbIdx = byteOffsetX + i;
                            nbbx.setFloat(xbIdx, op.op(nbbx.getFloat(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < 4 * n; i += 4) {
                            nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i)));
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = 0; i < 4 * n; i += 4) {
                            int xbIdx = byteOffsetX + i * incrX;
                            nbbx.setFloat(xbIdx, op.op(nbbx.getFloat(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < 4 * n; i += 4) {
                            nbbz.setFloat(byteOffsetZ + i * incrZ, op.op(nbbx.getFloat(byteOffsetX + i * incrX)));
                        }
                    }
                }
            } else {
                int byteOffsetX = 8 * offsetX;
                int byteOffsetZ = 8 * offsetZ;
                if (incrX == 1 && (x == z || incrZ == 1)) {
                    if (x == z) {
                        for (int i = 0; i < 8 * n; i += 8) {
                            int xbIdx = byteOffsetX + i;
                            nbbx.setDouble(xbIdx, op.op(nbbx.getDouble(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < 8 * n; i += 8) {
                            nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i)));
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = 0; i < 8 * n; i += 8) {
                            int xbIdx = byteOffsetX + i * incrX;
                            nbbx.setDouble(xbIdx, op.op(nbbx.getDouble(xbIdx)));
                        }
                    } else {
                        for (int i = 0; i < 8 * n; i += 8) {
                            nbbz.setDouble(byteOffsetZ + i * incrZ, op.op(nbbx.getDouble(byteOffsetX + i * incrX)));
                        }
                    }
                }
            }
        }
    }
}
