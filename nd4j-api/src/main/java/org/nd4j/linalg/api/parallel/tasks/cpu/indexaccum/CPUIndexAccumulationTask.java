package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import io.netty.buffer.ByteBuf;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.parallel.tasks.Task;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

public class CPUIndexAccumulationTask extends BaseCPUIndexAccumulationTask {

    protected List<Task<Pair<Double,Integer>>> subTasks;

    /**
     * Constructor for operating on subset of NDArray
     */
    public CPUIndexAccumulationTask(IndexAccumulation op, int threshold, int n, int offsetX, int offsetY, int incrX, int incrY,
                                    int elementOffset, boolean outerTask) {
        super(op, threshold, n, offsetX, offsetY, incrX, incrY, elementOffset, outerTask);
    }

    /**
     * Constructor for doing task on entire NDArray
     */
    public CPUIndexAccumulationTask(IndexAccumulation op, int threshold, boolean outerTask) {
        super(op, threshold, outerTask);
    }

    /**
     * Constructor for doing a 1d tensor first
     */
    public CPUIndexAccumulationTask(IndexAccumulation op, int threshold, int tadIdx, int tadDim, boolean outerTask) {
        super(op, threshold, tadIdx, tadDim, outerTask);
    }

    @Override
    public Pair<Double,Integer> blockUntilComplete() {
        if (future == null ) {
            //invokeAsync hasn't been called?
            invokeAsync();
        }
        Pair<Double,Integer> accum;
        try {
            accum = future.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        if(subTasks != null) {
            //Callable/ExecutorServiceTask was broken into subtasks instead of being executed directly
            accum = op.zeroPair();
            for(Task<Pair<Double,Integer>> task : subTasks){
                Pair<Double,Integer> subAccum = task.blockUntilComplete();
                accum = op.combineSubResults(accum,subAccum);
            }
        }
        if(outerTask){
            op.setFinalResult(accum.getSecond());
        }
        return accum;
    }

    @Override
    public Pair<Double,Integer> call() {
        //Callable/ExecutorService
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

                Task<Pair<Double, Integer>> t = new CPUIndexAccumulationTask(op, threshold, nInTask, offsetXNew, offsetYNew,
                        incrX, incrY, elementOffset + soFar, false);
                t.invokeAsync();
                subTasks.add(t);

                soFar += nInTask;
            }
        } else {
            //Execute directly
            return execute();
        }
        return null;
    }

    @Override
    protected Pair<Double, Integer> compute() {
        //Fork join
        if(doTensorFirst) doTensorFirst(op);

        if(n > threshold){
            //Break into subtasks:
            int nFirst = n/2;
            RecursiveTask<Pair<Double,Integer>> first = new CPUIndexAccumulationTask(op, threshold, nFirst, offsetX,
                    offsetY, incrX, incrY, elementOffset, false);
            first.fork();

            int nSecond = n - nFirst;
            int offsetX2 = offsetX + nFirst * incrX;
            int offsetY2 = offsetY + nFirst * incrY;
            RecursiveTask<Pair<Double,Integer>> second = new CPUIndexAccumulationTask(op, threshold, nSecond, offsetX2,
                    offsetY2, incrX, incrY, elementOffset+nFirst, false);
            second.fork();

            return op.combineSubResults(first.join(),second.join());
        } else {
            return execute();
        }
    }

    private Pair<Double,Integer> execute(){
        DataBuffer x = op.x().data();
        DataBuffer y = (op.y() != null ? op.y().data() : null);

        if (y != null) {
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
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
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
                    int finalIdx = idxAccum + elementOffset;
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
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
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
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
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
                }
            }
        } else {
            //Task: accum = update(accum,X)
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
                    int finalIdx = idxAccum + elementOffset;
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
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
                    int finalIdx = idxAccum + elementOffset;
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
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
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>((double) accum, finalIdx);
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
                    int finalIdx = idxAccum + elementOffset;    //idxAccum is 'local' index. Add elementOffset to get index w.r.t. original idx
                    if (outerTask) op.setFinalResult(finalIdx);
                    return new Pair<>(accum, finalIdx);
                }
            }
        }
    }
}
