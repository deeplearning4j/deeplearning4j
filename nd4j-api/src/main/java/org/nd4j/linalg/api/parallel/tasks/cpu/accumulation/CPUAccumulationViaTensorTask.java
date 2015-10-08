package org.nd4j.linalg.api.parallel.tasks.cpu.accumulation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskFactory;
import org.nd4j.linalg.api.parallel.tasks.TaskFactoryProvider;
import org.nd4j.linalg.api.parallel.tasks.cpu.transform.CPUTransformOpAction;

import java.util.ArrayList;
import java.util.List;

public class CPUAccumulationViaTensorTask extends BaseTask<Double> {
    protected final Accumulation op;
    protected final int threshold;

    protected final boolean outerTask;
    protected List<Task<Double>> subTasks;

    public CPUAccumulationViaTensorTask(Accumulation op, int threshold, boolean outerTask){
        this.op = op;
        this.threshold = threshold;
        this.outerTask = outerTask;
    }

    @Override
    public void invokeAsync() {
        INDArray x = op.x();
        INDArray y = op.y();

        //Break the accumulation op into tensors
        //Run accumulation on each tensor
        //And combine the results

        int tensorDim;
        if(y==null) tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
        else tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x,y);

        int nTensors = x.tensorssAlongDimension(tensorDim);
        subTasks = new ArrayList<>(nTensors);
        if(nTensors == 1){
            //Vector: shouldn't happen
            Task<Double> task = new CPUAccumulationTask(op,threshold,false);
            task.invokeAsync();
            subTasks.add(task);
        } else {
            if(x.rank()==2){
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                DataBuffer dx = x.data();
                if(y==null){
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        Task<Double> task = new CPUAccumulationTask(op,threshold,n,offsetX,0,incrX,0,false);
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                } else {
                    DataBuffer dy = y.data();
                    OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                    int incrY = tsy.getElementWiseStride();
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                        Task<Double> task = new CPUAccumulationTask(op,threshold,n,offsetX,offsetY,incrX,incrY,false);
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            } else {
                //3+ dimensions
                for( int i=0; i<nTensors; i++ ){
                    Task<Double> task = new CPUAccumulationTask(op,threshold,i,tensorDim,false);
                    task.invokeAsync();
                    subTasks.add(task);
                }
            }
        }
    }

    @Override
    public Double blockUntilComplete() {
        if(subTasks==null){
            //invokeAsync hasn't been called
            invokeAsync();
        }
        double accum = op.zeroDouble();
        for(Task<Double> task : subTasks){
            double subAccum = task.blockUntilComplete();
            accum = op.combineSubResults(accum, subAccum);
        }
        if(outerTask){
            return op.getAndSetFinalResult(accum);
        }
        return accum;
    }

    @Override
    public Double call() {
        return null;    //Not applicable
    }
}
