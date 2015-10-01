package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.bufferops.impl.scalar.ScalarOpDataBufferAction;
import org.nd4j.linalg.api.parallel.bufferops.impl.transform.TransformOpDataBufferAction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;

@AllArgsConstructor
public class ScalarViaTensorDataBufferAction extends RecursiveAction {
    protected final ScalarOp op;
    protected final int threshold;
    protected final INDArray x;
    protected final INDArray z;

    @Override
    public void compute(){
        //Break the scalar op into tensors
        //Run the scalar op on each tensor

        int tensorDim;
        if(x==z){
            tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
        } else {
            tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x,z);
        }

        int nTensors = x.tensorssAlongDimension(tensorDim);
        if(nTensors==1){
            INDArray tx = x.tensorAlongDimension(0,tensorDim);
            INDArray tz;
            if(z == null) tz = null;
            else tz = (x==z ? tx : z.tensorAlongDimension(0,tensorDim));
            int offsetX = tx.offset();
            int offsetZ = (tz!=null ? tz.offset() : 0);
            int incrX = tx.elementWiseStride();
            int incrZ = (tz !=null ? tz.elementWiseStride() : 0);
            op.getScalarOpDataBufferAction(threshold,tx.length(),x.data(),(z!=null?z.data():null),
                    offsetX,offsetZ,incrX,incrZ).invoke();
            return;
        } else {
            List<ScalarDataBufferAction> blockList = new ArrayList<>(nTensors);
            if(x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                DataBuffer dx = x.data();
                if(x==z){
                    //x=Op(x)
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        ScalarDataBufferAction task = op.getScalarOpDataBufferAction(threshold, n, dx, dx, offsetX,
                                offsetX, incrX, incrX);
                        task.fork();
                        blockList.add(task);
                    }
                } else {
                    //z=Op(x)
                    DataBuffer dz = z.data();
                    OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                    int incrZ = tsz.getElementWiseStride();
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                        ScalarDataBufferAction task = op.getScalarOpDataBufferAction(threshold,n,dx,dz,offsetX,
                                offsetZ,incrX,incrZ);
                        task.fork();
                        blockList.add(task);
                    }
                }

            } else {
                //Use general purpose tensor calculation for everything else
                for (int i = 0; i < nTensors; i++) {
                    ScalarDataBufferAction task = op.getScalarOpDataBufferAction(i, tensorDim, threshold, x, z);
                    task.fork();
                    blockList.add(task);
                }
            }

            //Block until all are completed
            for(ScalarDataBufferAction task : blockList){
                task.join();
            }
        }
    }
}
