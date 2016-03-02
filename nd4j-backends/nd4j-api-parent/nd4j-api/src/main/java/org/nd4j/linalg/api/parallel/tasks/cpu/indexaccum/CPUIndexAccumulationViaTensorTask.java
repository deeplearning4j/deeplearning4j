package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

public class CPUIndexAccumulationViaTensorTask extends BaseCPUTask<Pair<Double, Integer>> {
    protected final IndexAccumulation op;

    protected final boolean outerTask;
    protected List<Task<Pair<Double, Integer>>> subTasks;

    public CPUIndexAccumulationViaTensorTask(IndexAccumulation op, int threshold, boolean outerTask) {
        super(op,threshold);
        this.op = op;
        this.outerTask = outerTask;
    }

    @Override
    public Pair<Double, Integer> blockUntilComplete() {
        if (future == null) {
            //invokeAsync hasn't been called
            invokeAsync();
        }

        Pair<Double,Integer> accum;
        try{
            accum = future.get();
        }catch(Exception e ){
            throw new RuntimeException(e);
        }

        if(accum==null) {
            //!=null for FJ, == null for ExecutorService
            accum = op.zeroPair();
            for (Task<Pair<Double, Integer>> task : subTasks) {
                Pair<Double, Integer> subAccum = task.blockUntilComplete();
                accum = op.combineSubResults(accum, subAccum);
            }
        }
        if (outerTask) {
            op.setFinalResult(accum.getSecond());
        }
        return accum;
    }

    @Override
    public Pair<Double, Integer> call() {
        //Callable / ExecutorService
        return execute(false);
    }

    @Override
    protected Pair<Double, Integer> compute() {
        //Fork join
        return execute(true);
    }

    private Pair<Double, Integer> execute(final boolean forkJoin) {
        INDArray x = op.x();
        INDArray y = op.y();

        //Indexing is done in row-major order, hence always have to do along d1 to get right indexes
        int tensorDim = 1;

        int nTensors = x.tensorssAlongDimension(tensorDim);
        List<RecursiveTask<Pair<Double,Integer>>> fjTasks = null;
        if(forkJoin) fjTasks = new ArrayList<>(nTensors);
        else subTasks = new ArrayList<>(nTensors);
        if (nTensors == 1) {
            //Vector: shouldn't happen
            RecursiveTask<Pair<Double, Integer>> task = new CPUIndexAccumulationTask(op, threshold, false);
            return task.invoke();
        } else {
            if (x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                if (y == null) {
                    for (int i = 0; i < nTensors; i++) {
                        int offsetX = tsx.getFirstTensorOffset() + i * tsx.getTensorStartSeparation();
                        int elementOffset = i * tsx.getTensorLength();
                        CPUIndexAccumulationTask task = new CPUIndexAccumulationTask(op, threshold, n, offsetX, 0,
                                incrX, 0, elementOffset, false);
                        if(forkJoin){
                            task.fork();
                            fjTasks.add(task);
                        } else {
                            task.invokeAsync();
                            subTasks.add(task);
                        }

                    }
                } else {
                    OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y, tensorDim);
                    int incrY = tsy.getElementWiseStride();
                    for (int i = 0; i < nTensors; i++) {
                        int offsetX = tsx.getFirstTensorOffset() + i * tsx.getTensorStartSeparation();
                        int offsetY = tsy.getFirstTensorOffset() + i * tsy.getTensorStartSeparation();
                        int elementOffset = i * tsx.getTensorLength();

                        CPUIndexAccumulationTask task = new CPUIndexAccumulationTask(op, threshold, n, offsetX, offsetY,
                                incrX, incrY, elementOffset, false);
                        if(forkJoin){
                            task.fork();
                            fjTasks.add(task);
                        } else {
                            task.invokeAsync();
                            subTasks.add(task);
                        }
                    }
                }
            } else {
                //3+ dimensions
                for (int i = 0; i < nTensors; i++) {
                    CPUIndexAccumulationTask task = new CPUIndexAccumulationTask(op,threshold,i,tensorDim,false);
                    if(forkJoin){
                        task.fork();
                        fjTasks.add(task);
                    } else {
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            }
        }

        if(forkJoin) {
            Pair<Double, Integer> accum = op.zeroPair();
            for (RecursiveTask<Pair<Double, Integer>> task : fjTasks) {
                Pair<Double, Integer> subAccum = task.join();
                accum = op.combineSubResults(accum, subAccum);
            }
            return accum;
        } else return null;
    }
}
