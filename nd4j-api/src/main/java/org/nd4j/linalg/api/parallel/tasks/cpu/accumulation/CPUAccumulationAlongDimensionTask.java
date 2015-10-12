package org.nd4j.linalg.api.parallel.tasks.cpu.accumulation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;


public class CPUAccumulationAlongDimensionTask extends BaseCPUTask<INDArray> {
    protected final Accumulation op;
    protected final int[] dimensions;

    protected List<Task<Double>> subTasks;

    public CPUAccumulationAlongDimensionTask(Accumulation op, int parallelThreshold, int... dimensions ){
        super(op,parallelThreshold);
        this.op = op;
        this.dimensions = dimensions;
    }

    @Override
    public INDArray blockUntilComplete() {
        if(future==null){
            //invokeAsync() not called?
            invokeAsync();
        }

        try{
            future.get();
        }catch(Exception e ){
            throw new RuntimeException(e);
        }

        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i=0;
        for(Task<Double> task : subTasks ){
            out.putScalar(i++,task.blockUntilComplete());
        }
        op.setZ(out);
        return out;
    }

    @Override
    public INDArray call() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        subTasks = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            Accumulation opOnDimension = (Accumulation)op.opForDimension(i,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            Task<Double> task;
            if(canDoDirectly){
                task = new CPUAccumulationTask(opOnDimension,threshold,true);
            } else {
                task = new CPUAccumulationViaTensorTask(opOnDimension,threshold,true);
            }

            task.invokeAsync();
            subTasks.add(task);
        }
        return null;
    }
}
