package org.nd4j.linalg.api.parallel.tasks.cpu.accumulation;


import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator;

import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

public class CPUAccumulations1dAction extends RecursiveAction implements Task<Void> {
    private Future future;
    private List<Task<?>> subTasks = null;
    private Accumulation op;
    private int threshold;
    private TensorCalculator tCalcx;
    private TensorCalculator tCalcy;
    private int firstTensor;
    private int lastTensor;
    private INDArray output;

    public CPUAccumulations1dAction(Accumulation op, int threshold, TensorCalculator tCalcx, TensorCalculator tCalcy,
                                    int firstTensor, int lastTensor, INDArray output) {
        this.op = op;
        this.threshold = threshold;
        this.tCalcx = tCalcx;
        this.tCalcy = tCalcy;
        this.firstTensor = firstTensor;
        this.lastTensor = lastTensor;
        this.output = output;
    }

    @Override
    protected void compute() {
        //Fork join

        int nTensors = (lastTensor - firstTensor + 1);
        int nElements = nTensors * tCalcx.getTensorLength();
        if (nTensors > 1 && nElements > threshold) {
            //Split:
            int nFirst = nTensors / 2;
            CPUAccumulations1dAction taskLeft = new CPUAccumulations1dAction(op, threshold, tCalcx, tCalcy,
                    firstTensor, firstTensor + nFirst - 1, output);
            taskLeft.fork();
            CPUAccumulations1dAction taskRight = new CPUAccumulations1dAction(op, threshold, tCalcx, tCalcy,
                    firstTensor + nFirst, lastTensor, output);
            taskRight.fork();

            taskLeft.join();
            taskRight.join();

        } else if (nTensors == 1 && nElements > threshold) {
            //Split as per IndexAccumulationAlongDimension, i.e., break into subtasks
            int offsetX = tCalcx.getOffsetForTensor(firstTensor);
            int offsetY = (tCalcy != null ? tCalcy.getOffsetForTensor(firstTensor) : 0);
            int incrX = tCalcx.getElementWiseStrideForTensor();
            int incrY = (tCalcy != null ? tCalcy.getElementWiseStrideForTensor() : 0);
            int n = tCalcx.getTensorLength();

            int nFirst = n / 2;
            RecursiveTask<Double> first = new CPUAccumulationTask(op, threshold, nFirst, offsetX,
                    offsetY, incrX, incrY, false);
            first.fork();

            int nSecond = n - nFirst;
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            RecursiveTask<Double> second = new CPUAccumulationTask(op, threshold, nSecond, offsetX2,
                    offsetY2, incrX, incrY, false);
            second.fork();

            double accum = op.combineSubResults(first.join(), second.join());
            output.putScalar(firstTensor, op.calculateFinalResult(accum, tCalcx.getTensorLength()));
        } else {
            //Calculate directly
            execute();
        }

    }

    @Override
    public Void call() {
        //Callable
        throw new UnsupportedOperationException("Not yet implemented");
    }

    private void execute() {
        DataBuffer x = op.x().data();
        DataBuffer y = (op.y() != null ? op.y().data() : null);
        int incrX = tCalcx.getElementWiseStrideForTensor();
        int n = tCalcx.getTensorLength();

        if (y != null) {
            int incrY = tCalcy.getElementWiseStrideForTensor();

            for (int tensorNum = firstTensor; tensorNum <= lastTensor; tensorNum++) {
                int offsetX = tCalcx.getOffsetForTensor(tensorNum);
                int offsetY = tCalcy.getOffsetForTensor(tensorNum);

                //Task: accum = update(accum,X,Y)
                if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
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
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
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
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
                    }
                } else {
                    ByteBuf nbbx = x.asNetty();
                    ByteBuf nbby = y.asNetty();
                    if (x.dataType() == DataBuffer.Type.FLOAT) {
                        int byteOffsetX = 4 * offsetX;
                        int byteOffsetY = 4 * offsetY;
                        float accum = op.zeroFloat();
                        int idx = 0;
                        if (incrX == 1 && incrY == 1) {
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i);
                                float fy = nbby.getFloat(byteOffsetY + i);
                                accum = op.update(accum, fx, fy);
                            }
                        } else {
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i * incrX);
                                float fy = nbby.getFloat(byteOffsetY + i * incrY);
                                accum = op.update(accum, fx, fy);
                            }
                        }
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
                    } else {
                        int byteOffsetX = 8 * offsetX;
                        int byteOffsetY = 8 * offsetY;
                        double accum = op.zeroDouble();
                        int idx = 0;
                        if (incrX == 1 && incrY == 1) {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i);
                                double dy = nbby.getDouble(byteOffsetY + i);
                                accum = op.update(accum, dx, dy);
                            }
                        } else {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i * incrX);
                                double dy = nbby.getDouble(byteOffsetY + i * incrY);
                                accum = op.update(accum, dx, dy);
                            }
                        }
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
                    }
                }
            }
        } else {
            //Task: accum = update(accum,X)
            for (int tensorNum = firstTensor; tensorNum <= lastTensor; tensorNum++) {
                int offsetX = tCalcx.getOffsetForTensor(tensorNum);

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
                        //Because this is for index accum along dimension: assign directly
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
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
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
                    }
                } else {
                    ByteBuf nbbx = x.asNetty();
                    if (x.dataType() == DataBuffer.Type.FLOAT) {
                        int byteOffsetX = 4 * offsetX;
                        float accum = op.zeroFloat();
                        int idx = 0;
                        if (incrX == 1) {
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i);
                                accum = op.update(accum, fx);
                            }
                        } else {
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i * incrX);
                                accum = op.update(accum, fx);
                            }
                        }
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
                    } else {
                        int byteOffsetX = 8 * offsetX;
                        double accum = op.zeroDouble();
                        int idx = 0;
                        if (incrX == 1) {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i);
                                accum = op.update(accum, dx);
                            }
                        } else {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i * incrX);
                                accum = op.update(accum, dx);
                            }
                        }
                        output.putScalar(tensorNum, op.calculateFinalResult(accum, n));
                    }
                }
            }
        }
    }


    @Override
    public Void invokeBlocking() {
        invokeAsync();
        return blockUntilComplete();
    }

    @Override
    public void invokeAsync() {
        this.future = TaskExecutorProvider.getTaskExecutor().executeAsync(this);
    }

    @Override
    public Void blockUntilComplete() {
        if (future == null) {
            //invokeAsync hasn't been called?
            invokeAsync();
        }
        try {
            future.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (subTasks != null) {
            for (Task<?> t : subTasks) {
                t.blockUntilComplete();
            }
        }
        return null;
    }
}
