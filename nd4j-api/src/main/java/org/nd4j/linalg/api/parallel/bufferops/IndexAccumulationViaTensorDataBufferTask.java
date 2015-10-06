package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.bufferops.impl.indexaccum.IndexAccumulationOpDataBufferTask;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

/**A DataBufferTask for executing index accumulation ops on a buffer in parallel.
 * The 'via tensor' designation in the name reflects the fact that this is done
 * by breaking the INDArray down into 1d tensors (which is necessary for example
 * when the elements of the x and y NDArrays are not contiguous in their DataBuffers)
 * @author Alex Black
 * @see IndexAccumulationDataBufferTask
 */
@AllArgsConstructor
public class IndexAccumulationViaTensorDataBufferTask extends RecursiveTask<Pair<Double,Integer>> {
    protected final IndexAccumulation op;
    protected final int threshold;
    protected final INDArray x;
    protected final INDArray y;

    @Override
    protected Pair<Double,Integer> compute() {
        //Break the index accumulation op into tensors
        //Run index accumulation on each tensor
        //And combine the results

        int tensorDim;
        if(y==null) tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
        else tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x,y);

        int nTensors = x.tensorssAlongDimension(tensorDim);
        if(nTensors == 1) {
            return new IndexAccumulationOpDataBufferTask(op, 0, tensorDim, threshold, x, y, true).invoke();
        } else {
            List<IndexAccumulationOpDataBufferTask> blockList = new ArrayList<>(nTensors);
            if(x.rank()==2){
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                DataBuffer dx = x.data();
                if(y==null){
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        int elementOffset = i*tsx.getTensorLength();
                        IndexAccumulationOpDataBufferTask task = new IndexAccumulationOpDataBufferTask(op,threshold,n,dx,null,offsetX,0,incrX,0,elementOffset,false);
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
                        int elementOffset = i*tsx.getTensorLength();
                        IndexAccumulationOpDataBufferTask task = new IndexAccumulationOpDataBufferTask(op,threshold,n,dx,dy,offsetX,offsetY,incrX,incrY,elementOffset,false);
                        task.fork();
                        blockList.add(task);
                    }
                }
            } else {
                //3+ dimensions
                for( int i=0; i<nTensors; i++ ){
                    IndexAccumulationOpDataBufferTask task = new IndexAccumulationOpDataBufferTask(op,i,tensorDim,threshold,x,y,false);
                    task.fork();
                    blockList.add(task);
                }
            }

            Pair<Double,Integer> accum = op.zeroPair();
            for(IndexAccumulationOpDataBufferTask task : blockList){
                Pair<Double,Integer> subAccum = task.join();
                accum = op.combineSubResults(accum,subAccum);
            }
            op.setFinalResult(accum.getSecond());
            return accum;
        }
    }
}
