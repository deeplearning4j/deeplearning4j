package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

@AllArgsConstructor
public class AccumulationAlongDimensionDataBufferTask extends RecursiveTask<INDArray> {
    protected final Accumulation op;
    protected final int parallelThreshold;
    protected final int[] dimensions;


    @Override
    protected INDArray compute() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        List<RecursiveTask<Double>> taskList = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            RecursiveTask<Double> task = new TensorCalculator(i);
            task.fork();
            taskList.add(task);
        }

        //Allocate return array + assign elements
        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i=0;
        for(RecursiveTask<Double> task : taskList ){
            out.putScalar(i++,task.join());
        }

        return out;
    }

    /** This TensorCalculator class is used to shift the tensor calculation from the original thread
     * to the forked thread, so all tensor calculations can be done in parallel
     *  */
    @AllArgsConstructor
    private class TensorCalculator extends RecursiveTask<Double>{
        private final int tensorNum;

        @Override
        protected Double compute() {
            Accumulation opOnDimension = (Accumulation)op.opForDimension(tensorNum,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoTransformOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoTransformOpDirectly(x2,y2);

            RecursiveTask<Double> task;
            if(canDoDirectly){
                if(y2!=null){
                    task = opOnDimension.getAccumulationOpDataBufferTask(parallelThreshold,opOnDimension.n(),x2.data(),y2.data(),
                            x2.offset(),y2.offset(),x2.elementWiseStride(),y2.elementWiseStride(),true);
                } else {
                    task = opOnDimension.getAccumulationOpDataBufferTask(parallelThreshold,opOnDimension.n(),x2.data(),null,
                            x2.offset(),0,x2.elementWiseStride(),0,true);
                }
            } else {
                task = new AccumulationViaTensorDataBufferTask(opOnDimension,parallelThreshold,x2,y2);
            }

            return task.invoke();
        }
    }
}
