package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;

import java.util.ArrayList;
import java.util.List;

public class CPUIndexAccumulationViaTensorTask extends BaseTask<Pair<Double, Integer>> {
    protected final IndexAccumulation op;
    protected final int threshold;

    protected final boolean outerTask;
    protected List<Task<Pair<Double, Integer>>> subTasks;

    public CPUIndexAccumulationViaTensorTask(IndexAccumulation op, int threshold, boolean outerTask) {
        this.op = op;
        this.threshold = threshold;
        this.outerTask = outerTask;
    }

    @Override
    public void invokeAsync() {
        INDArray x = op.x();
        INDArray y = op.y();

        //Indexing is done in row-major order, hence always have to do along d1 to get right indexes
        int tensorDim = 1;

        int nTensors = x.tensorssAlongDimension(tensorDim);
        subTasks = new ArrayList<>(nTensors);
        if (nTensors == 1) {
            //Vector: shouldn't happen
            Task<Pair<Double, Integer>> task = new CPUIndexAccumulationTask(op, threshold, false);
            task.invokeAsync();
            subTasks.add(task);

        } else {
            if (x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                DataBuffer dx = x.data();
                if (y == null) {
                    for (int i = 0; i < nTensors; i++) {
                        int offsetX = tsx.getFirstTensorOffset() + i * tsx.getTensorStartSeparation();
                        int elementOffset = i * tsx.getTensorLength();
                        Task<Pair<Double, Integer>> task = new CPUIndexAccumulationTask(op, threshold, n, offsetX, 0,
                                incrX, 0, elementOffset, false);
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                } else {
                    DataBuffer dy = y.data();
                    OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y, tensorDim);
                    int incrY = tsy.getElementWiseStride();
                    for (int i = 0; i < nTensors; i++) {
                        int offsetX = tsx.getFirstTensorOffset() + i * tsx.getTensorStartSeparation();
                        int offsetY = tsy.getFirstTensorOffset() + i * tsy.getTensorStartSeparation();
                        int elementOffset = i * tsx.getTensorLength();

                        Task<Pair<Double, Integer>> task = new CPUIndexAccumulationTask(op, threshold, n, offsetX, offsetY,
                                incrX, incrY, elementOffset, false);
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            } else {
                //3+ dimensions
                for (int i = 0; i < nTensors; i++) {
                    Task<Pair<Double,Integer>> task = new CPUIndexAccumulationTask(op,threshold,i,tensorDim,false);
                    task.invokeAsync();
                    subTasks.add(task);
                }
            }
        }
    }

    @Override
    public Pair<Double, Integer> blockUntilComplete() {
        if (subTasks == null) {
            //invokeAsync hasn't been called
            invokeAsync();
        }
        Pair<Double, Integer> accum = op.zeroPair();
        for (Task<Pair<Double, Integer>> task : subTasks) {
            Pair<Double, Integer> subAccum = task.blockUntilComplete();
            accum = op.combineSubResults(accum, subAccum);
        }
        if (outerTask) {
            op.setFinalResult(accum.getSecond());
        }
        return accum;
    }

    @Override
    public Pair<Double, Integer> call() {
        return null;    //Not applicable
    }
}
