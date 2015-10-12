package org.nd4j.linalg.api.parallel.tasks.cpu.accumulation;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;


public class CPUAccumulationAlongDimensionTask extends BaseCPUTask<INDArray> {
    protected final Accumulation op;
    protected final int[] dimensions;

    protected List<Task<Double>> subTasks;

    public CPUAccumulationAlongDimensionTask(Accumulation op, int parallelThreshold, int... dimensions) {
        super(op, parallelThreshold);
        this.op = op;
        this.dimensions = dimensions;
    }

    @Override
    public INDArray blockUntilComplete() {
        if (future == null) {
            //invokeAsync() not called?
            invokeAsync();
        }

        try {
            future.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i = 0;
        for (Task<Double> task : subTasks) {
            out.putScalar(i++, task.blockUntilComplete());
        }
        op.setZ(out);
        return out;
    }

    @Override
    public INDArray call() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        subTasks = new ArrayList<>(nTensors);

        for (int i = 0; i < nTensors; i++) {
            Task<Double> task = new OpForDimTask(i);
            task.invokeAsync();
            subTasks.add(task);
        }
        return null;
    }


    private class OpForDimTask extends BaseTask<Double> {
        private int tensorNum;
        private Task<Double> subTask;
        private Future<Double> future;

        public OpForDimTask(int tensorNum){
            this.tensorNum = tensorNum;
        }

        @Override
        public void invokeAsync() {
            this.future = TaskExecutorProvider.getTaskExecutor().executeAsync(this);
        }

        @Override
        public Double blockUntilComplete() {
            try {
                future.get();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            return subTask.blockUntilComplete();
        }

        @Override
        public Double call() {
            Accumulation opOnDimension = (Accumulation) op.opForDimension(tensorNum, dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if (y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            if (canDoDirectly) {
                subTask = new CPUAccumulationTask(opOnDimension, threshold, true);
            } else {
                subTask = new CPUAccumulationViaTensorTask(opOnDimension, threshold, true);
            }

            subTask.invokeAsync();
            return null;
        }
    }
}
