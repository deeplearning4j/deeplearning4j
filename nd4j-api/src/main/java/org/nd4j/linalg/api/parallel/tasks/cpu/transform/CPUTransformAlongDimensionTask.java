package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;

public class CPUTransformAlongDimensionTask extends BaseCPUAction {

    protected final TransformOp op;
    protected final int[] dimensions;

    protected List<Task<Void>> subTasks;

    public CPUTransformAlongDimensionTask(TransformOp op, int threshold, int... dimensions ){
        super(op,threshold);
        this.op = op;
        this.dimensions = dimensions;
    }

    @Override
    public Void call() {
        //Callable / ExecutorService
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        subTasks = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            TransformOp opOnDimension = (TransformOp)op.opForDimension(i,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            Task<Void> task;
            if(canDoDirectly){
                task = new CPUTransformOpAction(opOnDimension,threshold);
            } else {
                task = new CPUTransformOpViaTensorTask(opOnDimension,threshold);
            }

            task.invokeAsync();
            subTasks.add(task);
        }
        return null;
    }

    @Override
    protected void compute() {
        //Fork join
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        List<RecursiveAction> subTasks = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            TransformOp opOnDimension = (TransformOp)op.opForDimension(i,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            RecursiveAction task;
            if(canDoDirectly){
                task = new CPUTransformOpAction(opOnDimension,threshold);
            } else {
                task = new CPUTransformOpViaTensorTask(opOnDimension,threshold);
            }

            task.fork();
            subTasks.add(task);
        }
        for(RecursiveAction t : subTasks){
            t.join();
        }
    }
}
