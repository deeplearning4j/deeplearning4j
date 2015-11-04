package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import io.netty.buffer.ByteBuf;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator;

import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

/**A class used specifically for IndexAccumulation along dimensions for 1d.
 * It differs from other IndexAccumulation tasks in that it assigns the output directly
 * instead of returning, plus does multiple index accumulation ops in the one object
 * (helps to avoid object creation cost for large number of small ops)
 */
public class CPUIndexAccumulations1dAction extends RecursiveAction implements Task<Void>  {
    private Future future;
    private List<Task<?>> subTasks = null;
    private IndexAccumulation op;
    private int threshold;
    private TensorCalculator tCalcx;
    private TensorCalculator tCalcy;
    private int firstTensor;
    private int lastTensor;
    private INDArray output;

    public CPUIndexAccumulations1dAction(IndexAccumulation op, int threshold, TensorCalculator tCalcx, TensorCalculator tCalcy,
                                         int firstTensor, int lastTensor, INDArray output){
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

        int nTensors = (lastTensor-firstTensor+1);
        int nElements = nTensors*tCalcx.getTensorLength();
        if(nTensors > 1 && nElements>threshold) {
            //Split:
            int nFirst = nTensors/2;
            CPUIndexAccumulations1dAction taskLeft = new CPUIndexAccumulations1dAction(op, threshold, tCalcx, tCalcy,
                    firstTensor, firstTensor+nFirst-1, output);
            taskLeft.fork();
            CPUIndexAccumulations1dAction taskRight = new CPUIndexAccumulations1dAction(op, threshold, tCalcx, tCalcy,
                    firstTensor+nFirst, lastTensor, output);
            taskRight.fork();

            taskLeft.join();
            taskRight.join();

        } else if(nTensors == 1 && nElements>threshold ) {
            //Split as per IndexAccumulationAlongDimension, i.e., break into subtasks
            int offsetX = tCalcx.getOffsetForTensor(firstTensor);
            int offsetY = (tCalcy != null ? tCalcy.getOffsetForTensor(firstTensor) : 0);
            int incrX = tCalcx.getElementWiseStrideForTensor();
            int incrY = (tCalcy != null ? tCalcy.getElementWiseStrideForTensor() : 0);
            int n = tCalcx.getTensorLength();

            int nFirst = n/2;
            RecursiveTask<Pair<Double,Integer>> first = new CPUIndexAccumulationTask(op, threshold, nFirst, offsetX,
                    offsetY, incrX, incrY, 0, false);
            first.fork();

            int nSecond = n - nFirst;
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            RecursiveTask<Pair<Double,Integer>> second = new CPUIndexAccumulationTask(op, threshold, nSecond, offsetX2,
                    offsetY2, incrX, incrY, nFirst, false);
            second.fork();

            Pair<Double,Integer> pair = op.combineSubResults(first.join(),second.join());
            output.putScalar(firstTensor, pair.getSecond());
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

    private void execute(){
        DataBuffer x = op.x().data();
        DataBuffer y = (op.y() != null ? op.y().data() : null);
        int incrX = tCalcx.getElementWiseStrideForTensor();
        int n = tCalcx.getTensorLength();

        if (y != null) {
            int incrY = tCalcy.getElementWiseStrideForTensor();

            for( int tensorNum = firstTensor; tensorNum <= lastTensor; tensorNum++ ) {
                int offsetX = tCalcx.getOffsetForTensor(tensorNum);
                int offsetY = tCalcy.getOffsetForTensor(tensorNum);

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
                        output.putScalar(tensorNum,idxAccum);
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
                        output.putScalar(tensorNum,idxAccum);
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
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i);
                                float fy = nbby.getFloat(byteOffsetY + i);
                                idxAccum = op.update(accum, idxAccum, fx, fy, idx);
                                if (idxAccum == idx) accum = op.op(fx, fy);
                            }
                        } else {
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i * incrX);
                                float fy = nbby.getFloat(byteOffsetY + i * incrY);
                                idxAccum = op.update(accum, idxAccum, fx, fy, idx);
                                if (idxAccum == idx) accum = op.op(fx, fy);
                            }
                        }
                        output.putScalar(tensorNum,idxAccum);
                    } else {
                        int byteOffsetX = 8 * offsetX;
                        int byteOffsetY = 8 * offsetY;
                        double accum = op.zeroDouble();
                        int idxAccum = -1;
                        int idx = 0;
                        if (incrX == 1 && incrY == 1) {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i);
                                double dy = nbby.getDouble(byteOffsetY + i);
                                idxAccum = op.update(accum, idxAccum, dx, dy, idx);
                                if (idxAccum == idx) accum = op.op(dx, dy);
                            }
                        } else {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i * incrX);
                                double dy = nbby.getDouble(byteOffsetY + i * incrY);
                                idxAccum = op.update(accum, idxAccum, dx, dy, idx);
                                if (idxAccum == idx) accum = op.op(dx, dy);
                            }
                        }
                        output.putScalar(tensorNum,idxAccum);
                    }
                }
            }
        } else {
            //Task: accum = update(accum,X)
            for( int tensorNum = firstTensor; tensorNum <= lastTensor; tensorNum++ ) {
                int offsetX = tCalcx.getOffsetForTensor(tensorNum);


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
                        //Because this is for index accum along dimension: assign directly
                        output.putScalar(tensorNum,idxAccum);
                    } else {
                        double[] xd = (double[]) x.array();
                        double accum = op.zeroDouble();
                        int idxAccum = -1;
                        if (incrX == 1) {
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
                        output.putScalar(tensorNum,idxAccum);
                    }
                } else {
                    ByteBuf nbbx = x.asNetty();
                    if (x.dataType() == DataBuffer.Type.FLOAT) {
                        int byteOffsetX = 4 * offsetX;
                        float accum = op.zeroFloat();
                        int idxAccum = -1;
                        int idx = 0;
                        if (incrX == 1) {
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i);
                                idxAccum = op.update(accum, idxAccum, fx, idx);
                                if (idxAccum == idx) accum = op.op(fx);
                            }
                        } else {
                            for (int i = 0; i < 4 * n; i += 4, idx++) {
                                float fx = nbbx.getFloat(byteOffsetX + i * incrX);
                                idxAccum = op.update(accum, idxAccum, fx, idx);
                                if (idxAccum == idx) accum = op.op(fx);
                            }
                        }
                        output.putScalar(tensorNum,idxAccum);
                    } else {
                        int byteOffsetX = 8 * offsetX;
                        double accum = op.zeroDouble();
                        int idxAccum = -1;
                        int idx = 0;
                        if (incrX == 1) {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i);
                                idxAccum = op.update(accum, idxAccum, dx, idx);
                                if (idxAccum == idx) accum = op.op(dx);
                            }
                        } else {
                            for (int i = 0; i < 8 * n; i += 8, idx++) {
                                double dx = nbbx.getDouble(byteOffsetX + i * incrX);
                                idxAccum = op.update(accum, idxAccum, dx, idx);
                                if (idxAccum == idx) accum = op.op(dx);
                            }
                        }
                        output.putScalar(tensorNum,idxAccum);
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
