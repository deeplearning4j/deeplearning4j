package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.accumulation.CPUAccumulationTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.accumulation.CPUAccumulationViaTensorTask;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;


public class CPUIndexAccumulationAlongDimensionTask extends BaseTask<INDArray> {
    protected final IndexAccumulation op;
    protected final int parallelThreshold;
    protected final int[] dimensions;

    protected List<Task<Pair<Double,Integer>>> subTasks;

    public CPUIndexAccumulationAlongDimensionTask(IndexAccumulation op, int parallelThreshold, int[] dimensions){
        this.op = op;
        this.parallelThreshold = parallelThreshold;
        this.dimensions = dimensions;
    }

    @Override
    public void invokeAsync() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        subTasks = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            IndexAccumulation opOnDimension = (IndexAccumulation)op.opForDimension(i,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            Task<Pair<Double,Integer>> task;
            if(canDoDirectly){
                task = new CPUIndexAccumulationTask(opOnDimension,parallelThreshold,true);
            } else {
                task = new CPUIndexAccumulationViaTensorTask(op,parallelThreshold,true);
            }

            task.invokeAsync();
            subTasks.add(task);
        }
    }

    @Override
    public INDArray blockUntilComplete() {
        if(subTasks==null){
            //invokeAsync() not called?
            invokeAsync();
        }

        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i=0;
        for(Task<Pair<Double,Integer>> task : subTasks ){
            Pair<Double,Integer> result = task.blockUntilComplete();
            out.putScalar(i++,result.getSecond());
        }
        op.setZ(out);
        return out;
    }

    @Override
    public INDArray call() {
        return null;    //Not applicable
    }
}
