package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

@AllArgsConstructor
public class AccumulationViaTensorDataBufferTask extends RecursiveTask<Double> {

    protected final Accumulation op;
    protected final int threshold;
    protected final INDArray x;
    protected final INDArray y;

    @Override
    protected Double compute() {
        //Break the accumulation op into tensors
        //Run accumulation on each tensor
        //And combine the results

        int tensorDim;
        if(y==null) tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
        else tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x,y);

        int nTensors = x.tensorssAlongDimension(tensorDim);
        if(nTensors == 1){
            INDArray tx = x.tensorAlongDimension(0,tensorDim);
            INDArray ty = (y!=null ? y.tensorAlongDimension(0,tensorDim) : null);
            int offsetX = tx.offset();
            int offsetY = (y!=null ? ty.offset() : 0);
            int incrX = tx.elementWiseStride();
            int incrY = (y!=null ? ty.elementWiseStride() : 0);
            double accum = op.getAccumulationOpDataBufferTask(threshold,tx.length(),x.data(),(y!=null?y.data():null),
                    offsetX,offsetY,incrX,incrY,true).invoke();
            return op.getAndSetFinalResult(accum);
        } else {
            List<AccumulationDataBufferTask> blockList = new ArrayList<>(nTensors);

            if(x.rank()==2){
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                DataBuffer dx = x.data();
                if(y==null){
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        AccumulationDataBufferTask task = op.getAccumulationOpDataBufferTask(threshold,n,dx,null,offsetX,0,incrX,0,false);
                        task.fork();
                        blockList.add(task);
                    }
                } else {
                    DataBuffer dy = y.data();
                    OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                    int incrY = tsy.getElementWiseStride();
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                        AccumulationDataBufferTask task = op.getAccumulationOpDataBufferTask(threshold,n,dx,dy,offsetX,offsetY,incrX,incrY,false);
                        task.fork();
                        blockList.add(task);
                    }
                }
            } else {
                //3+ dimensions
                for( int i=0; i<nTensors; i++ ){
                    AccumulationDataBufferTask task = op.getAccumulationOpDataBufferTask(i,tensorDim,threshold,x,y,false);
                    task.fork();
                    blockList.add(task);
                }
            }

            double accum = op.zeroDouble();
            for(AccumulationDataBufferTask task : blockList){
                double subAccum = task.join();
                accum = op.combineSubResults(accum,subAccum);
            }
            return op.getAndSetFinalResult(accum);
        }
    }
}
