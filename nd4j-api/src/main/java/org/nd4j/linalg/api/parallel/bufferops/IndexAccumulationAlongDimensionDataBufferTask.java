package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

/** DataBufferTask for executing index accumulations along one or more dimensions in parallel
 * @author Alex Black
 */
@AllArgsConstructor
public class IndexAccumulationAlongDimensionDataBufferTask extends RecursiveTask<INDArray> {

    protected final IndexAccumulation op;
    protected final int parallelThreshold;
    protected final int[] dimensions;


    @Override
    protected INDArray compute() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        List<RecursiveTask<Pair<Double,Integer>>> taskList = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            RecursiveTask<Pair<Double,Integer>> task = new TensorCalculator(i);
            task.fork();
            taskList.add(task);
        }

        //Allocate return array + assign elements
        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i=0;
        for(RecursiveTask<Pair<Double,Integer>> task : taskList ){
            Pair<Double,Integer> pair = task.join();
            out.putScalar(i++,pair.getSecond());
        }

        return out;
    }

    /** This TensorCalculator class is used to shift the tensor calculation from the original thread
     * to the forked thread, so all tensor calculations can be done in parallel
     *  */
    @AllArgsConstructor
    private class TensorCalculator extends RecursiveTask<Pair<Double,Integer>>{
        private final int tensorNum;

        @Override
        protected Pair<Double,Integer> compute() {
            IndexAccumulation opOnDimension = (IndexAccumulation)op.opForDimension(tensorNum,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            RecursiveTask<Pair<Double,Integer>> task;
            if(canDoDirectly){
                if(y2!=null){
                    task = opOnDimension.getIndexAccumulationOpDataBufferTask(parallelThreshold,opOnDimension.n(),x2.data(),y2.data(),
                            x2.offset(),y2.offset(),x2.elementWiseStride(),y2.elementWiseStride(),0,true);
                } else {
                    task = opOnDimension.getIndexAccumulationOpDataBufferTask(parallelThreshold,opOnDimension.n(),x2.data(),null,
                            x2.offset(),0,x2.elementWiseStride(),0,0,true);
                }
            } else {
                task = new IndexAccumulationViaTensorDataBufferTask(opOnDimension,parallelThreshold,x2,y2);
            }

            return task.invoke();
        }
    }
}
