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
        INDArray x = op.x();
        INDArray y = op.y();

        DataBuffer dx = x.data();
        DataBuffer dy = (y != null ? y.data() : null);

        int nTensors = x.tensorssAlongDimension(dimensions);
        List<RecursiveTask<Double>> taskList = new ArrayList<>(nTensors);

        boolean canDoDirectly = false;
        for( int i=0; i<nTensors; i++ ){
            //TODO: Push this tensor calculation into forked thread (instead of this thread)
            Accumulation opOnDimension = (Accumulation)op.opForDimension(i,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            if(i==0){
                if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoTransformOpDirectly(x2);
                else canDoDirectly = OpExecutionerUtil.canDoTransformOpDirectly(x2,y2);
            }

            RecursiveTask<Double> task;
            if(canDoDirectly){
                if(y!=null){
                    task = opOnDimension.getAccumulationOpDataBufferTask(0,opOnDimension.n(),dx,dy,x2.offset(),y2.offset(),
                            x2.elementWiseStride(),y2.elementWiseStride(),true);
                } else {
                    task = opOnDimension.getAccumulationOpDataBufferTask(0,opOnDimension.n(),dx,null,x2.offset(),0,x2.elementWiseStride(),0,true);
                }
            } else {
                task = new AccumulationViaTensorDataBufferTask(opOnDimension,parallelThreshold,x2,y2);
            }
            task.fork();
            taskList.add(task);
        }

        //Allocate return array + assign elements
        int[] retShape = ArrayUtil.removeIndex(x.shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i=0;
        for(RecursiveTask<Double> task : taskList ){
            out.putScalar(i++,task.join());
        }

        return out;
    }
}
