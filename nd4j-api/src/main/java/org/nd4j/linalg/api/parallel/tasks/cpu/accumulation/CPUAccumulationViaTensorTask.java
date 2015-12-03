package org.nd4j.linalg.api.parallel.tasks.cpu.accumulation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

public class CPUAccumulationViaTensorTask extends BaseCPUTask<Double> {
    protected final Accumulation op;

    protected final boolean outerTask;
    protected List<Task<Double>> subTasks;

    public CPUAccumulationViaTensorTask(Accumulation op, int threshold, boolean outerTask) {
        super(op, threshold);
        this.op = op;
        this.outerTask = outerTask;
    }

    @Override
    public Double blockUntilComplete() {
        if (future == null) {
            //invokeAsync hasn't been called
            invokeAsync();
        }
        Double accum;
        try {
            accum = future.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        if (subTasks != null) {
            //Iterative decomposition / callable: task was broken into subtasks, instead of executing directly
            //subTasks == null for FJ execution
            accum = op.zeroDouble();
            for (Task<Double> task : subTasks) {
                double subAccum = task.blockUntilComplete();
                accum = op.combineSubResults(accum, subAccum);
            }
        }
        if (outerTask && subTasks != null ) {
            //subTasks == null in FJ, op.getAndSetFinalResult already called for FJ if(outerTask) by this point
            return op.getAndSetFinalResult(accum);
        }
        return accum;
    }

    @Override
    public Double call() {
        //Callable execution
        return execute(false);
    }

    @Override
    protected Double compute() {
        //ForkJoin execution
        double out = execute(true);
        if(outerTask){
            return op.getAndSetFinalResult(out);
        } else {
            return out;
        }
    }

    private Double execute(final boolean forkJoin) {
        INDArray x = op.x();
        INDArray y = op.y();

        //Break the accumulation op into tensors
        //Run accumulation on each tensor
        //And combine the results

        int tensorDim;
        if (y == null) tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
        else tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y);

        int nTensors = x.tensorssAlongDimension(tensorDim);
        List<RecursiveTask<Double>> fjTasks = null;
        if (forkJoin) fjTasks = new ArrayList<>(nTensors);
        else subTasks = new ArrayList<>(nTensors);
        if (nTensors == 1) {
            //Vector: shouldn't happen
            CPUAccumulationTask task = new CPUAccumulationTask(op, threshold, false);
            if (forkJoin) {
                return task.invoke();
            } else {
                task.invokeAsync();
                subTasks.add(task);
                return null;
            }
        } else {
            if (x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                if (y == null) {
                    for (int i = 0; i < nTensors; i++) {
                        int offsetX = tsx.getFirstTensorOffset() + i * tsx.getTensorStartSeparation();
                        CPUAccumulationTask task = new CPUAccumulationTask(op, threshold, n, offsetX, 0, incrX, 0, false);
                        if (forkJoin) {
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
                        CPUAccumulationTask task = new CPUAccumulationTask(op, threshold, n, offsetX, offsetY, incrX, incrY, false);
                        if (forkJoin) {
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
                    CPUAccumulationTask task = new CPUAccumulationTask(op, threshold, i, tensorDim, false);
                    if (forkJoin) {
                        task.fork();
                        fjTasks.add(task);
                    } else {
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            }
        }

        if (forkJoin) {
            double accum = op.zeroDouble();
            for (RecursiveTask<Double> task : fjTasks) {
                accum = op.combineSubResults(accum, task.join());
            }
            return accum;
        } else return null;
    }
}
