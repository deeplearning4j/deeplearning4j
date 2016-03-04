package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator;
import org.nd4j.linalg.api.shape.tensor.TensorCalculatorFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;


public class CPUIndexAccumulationAlongDimensionTask extends BaseCPUTask<INDArray> {
    protected final IndexAccumulation op;
    protected final int[] dimensions;

    protected List<Task<Pair<Double,Integer>>> subTasks;

    public CPUIndexAccumulationAlongDimensionTask(IndexAccumulation op, int parallelThreshold, int[] dimensions){
        super(op,parallelThreshold);
        this.op = op;
        this.dimensions = dimensions;
    }

    @Override
    public INDArray blockUntilComplete() {
        if(future == null){
            //invokeAsync() not called?
            invokeAsync();
        }

        INDArray out;
        try{
            out = future.get();
        }catch(Exception e){
            throw new RuntimeException(e);
        }
        if(out != null) return out; //FJ

        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        out = Nd4j.create(retShape);
        int i=0;
        for(Task<Pair<Double,Integer>> task : subTasks) {
            Pair<Double,Integer> result = task.blockUntilComplete();
            out.putScalar(i++,result.getSecond());
        }
        op.setZ(out);
        return out;
    }

    @Override
    public INDArray call() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        subTasks = new ArrayList<>(nTensors);

        for( int i = 0; i < nTensors; i++ ){
            IndexAccumulation opOnDimension = (IndexAccumulation)op.opForDimension(i,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null)
                canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else
                canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            Task<Pair<Double,Integer>> task;
            if(canDoDirectly){
                task = new CPUIndexAccumulationTask(opOnDimension,threshold,true);
            } else {
                task = new CPUIndexAccumulationViaTensorTask(op,threshold,true);
            }

            task.invokeAsync();
            subTasks.add(task);
        }
        return null;
    }

    @Override
    protected INDArray compute() {
        //Fork join

        if(dimensions.length == 1 && !op.isPassThrough()){
            TensorCalculator tCalcx = TensorCalculatorFactory.getTensorCalculator(op.x(),dimensions[0]);
            TensorCalculator tCalcy;
            if(op.y() != null) tCalcy = TensorCalculatorFactory.getTensorCalculator(op.y(), dimensions[0]);
            else tCalcy = null;

            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
            INDArray out = Nd4j.create(retShape);

            RecursiveAction action = new CPUIndexAccumulations1dAction(op, threshold, tCalcx, tCalcy, 0,
                    tCalcx.getNumTensors()-1, out);
            action.invoke();
            op.setZ(out);
            return out;
        } else {
            int nTensors = op.x().tensorssAlongDimension(dimensions);
            List<RecursiveTask<Pair<Double, Integer>>> subTasks = new ArrayList<>(nTensors);

            for (int i = 0; i < nTensors; i++) {
                IndexAccumulation opOnDimension = (IndexAccumulation) op.opForDimension(i, dimensions);
                INDArray x2 = opOnDimension.x();
                INDArray y2 = opOnDimension.y();

                boolean canDoDirectly;
                if (y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
                else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

                RecursiveTask<Pair<Double, Integer>> task;
                if (canDoDirectly) {
                    task = new CPUIndexAccumulationTask(opOnDimension, threshold, true);
                } else {
                    task = new CPUIndexAccumulationViaTensorTask(op, threshold, true);
                }
                task.fork();
                subTasks.add(task);
            }

            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
            INDArray out = Nd4j.create(retShape);
            int i = 0;
            for (RecursiveTask<Pair<Double, Integer>> task : subTasks) {
                Pair<Double, Integer> result = task.join();
                out.putScalar(i++, result.getSecond());
            }
            op.setZ(out);
            return out;
        }
    }
}
