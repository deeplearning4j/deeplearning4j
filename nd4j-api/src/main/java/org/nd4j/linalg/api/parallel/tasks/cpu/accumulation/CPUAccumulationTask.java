package org.nd4j.linalg.api.parallel.tasks.cpu.accumulation;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.tasks.Task;

import java.util.ArrayList;
import java.util.List;

public class CPUAccumulationTask extends BaseCPUAccumulationTask {

    protected List<Task<Double>> subTasks;

    /**
     * Constructor for operating on subset of NDArray
     */
    public CPUAccumulationTask(Accumulation op, int threshold, int n, int offsetX, int offsetY, int incrX, int incrY,
                               boolean outerTask) {
        super(op, threshold, n, offsetX, offsetY, incrX, incrY, outerTask);
    }

    /**
     * Constructor for doing task on entire NDArray
     */
    public CPUAccumulationTask(Accumulation op, int threshold, boolean outerTask) {
        super(op, threshold, outerTask);
    }

    /**
     * Constructor for doing a 1d tensor first
     */
    public CPUAccumulationTask(Accumulation op, int threshold, int tadIdx, int tadDim, boolean outerTask) {
        super(op, threshold, tadIdx, tadDim, outerTask);
    }


    @Override
    public Double blockUntilComplete() {
        if (future == null) {
            //invokeAsync hasn't been called?
            invokeAsync();
        }
        Double accum;
        try {
            accum = future.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (subTasks != null) {
            //Callable - task was broken into subtasks, instead of executing directly
            //subTasks == null for FJ execution
            accum = op.zeroDouble();
            for (Task<Double> task : subTasks) {
                double subAccum = task.blockUntilComplete();
                accum = op.combineSubResults(accum, subAccum);
            }
        }

        if (outerTask) {
            return op.getAndSetFinalResult(accum);
        }
        return accum;
    }

    @Override
    public Double compute() {
        //Recursive decomposition (fork join)
        if (doTensorFirst) doTensorFirst(op);

        double out;
        if (n > threshold) {
            //Break into subtasks:
            int nFirst = n / 2;
            CPUAccumulationTask first = new CPUAccumulationTask(op, threshold, nFirst, offsetX, offsetY, incrX, incrY, false);
            first.fork();

            int nSecond = n - nFirst;
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            CPUAccumulationTask second = new CPUAccumulationTask(op, threshold, nSecond, offsetX2, offsetY2, incrX, incrY, false);
            second.fork();

            out = op.combineSubResults(first.join(), second.join());
        } else {
            //Execute directly
            out = execute();
        }
        if (outerTask) {
            return op.getAndSetFinalResult(out);
        } else {
            return out;
        }
    }


    @Override
    public Double call() {
        //Iterative decomposition (thread pool)
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
                int offsetYNew = offsetY + soFar * incrY;

                Task<Double> t = new CPUAccumulationTask(op, threshold, nInTask, offsetXNew, offsetYNew, incrX, incrY, false);
                t.invokeAsync();
                subTasks.add(t);

                soFar += nInTask;
            }
            return 0.0; //Not used, but to avoid null pointer on Double -> double
        } else {
            //Execute directly
            return execute();
        }
    }

    private double execute() {
        DataBuffer x = op.x().data();
        DataBuffer y = (op.y() != null ? op.y().data() : null);

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
                    return (double) accum;
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
                    return accum;
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
                    return (double) accum;
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
                    return accum;
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
                    return (double) accum;
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
                    return accum;
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
                    return (double) accum;
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
                    return accum;
                }
            }
        }
    }
}
