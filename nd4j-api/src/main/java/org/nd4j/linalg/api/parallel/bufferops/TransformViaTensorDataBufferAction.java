package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.bufferops.impl.transform.TransformOpDataBufferAction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;

/**A DataBufferAction for executing TransformOps on a buffer in parallel.
 * The 'via tensor' designation in the name reflects the fact that this is done
 * by breaking the INDArray down into 1d tensors (which is necessary for example
 * when the elements of the x and z NDArrays are not contiguous in their DataBuffers)
 * @author Alex Black
 * @see TransformDataBufferAction
 */
@AllArgsConstructor
public class TransformViaTensorDataBufferAction extends RecursiveAction {
    protected final TransformOp op;
    protected final int threshold;
    protected final INDArray x;
    protected final INDArray y;
    protected final INDArray z;

    @Override
    protected void compute() {
        //Break the transform op into tensors
        //Run transform on each tensor

        int tensorDim;
        if(y==null){
            if(x==z){
                //x=Op(x)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
            } else {
                //z=Op(x)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, z);
            }
        } else {
            if(x==z){
                //x=Op(x,y)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y);
            } else {
                //z=Op(x,y)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y, z);
            }
        }

        int nTensors = x.tensorssAlongDimension(tensorDim);
        if(nTensors == 1){
            INDArray tx = x.tensorAlongDimension(0,tensorDim);
            INDArray ty = (y!=null ? y.tensorAlongDimension(0,tensorDim) : null);
            INDArray tz = (z!=null ? z.tensorAlongDimension(0,tensorDim) : null);
            int offsetX = tx.offset();
            int offsetY = (y!=null ? ty.offset() : 0);
            int offsetZ = (z!=null ? tz.offset() : 0);
            int incrX = tx.elementWiseStride();
            int incrY = (y!=null ? ty.elementWiseStride() : 0);
            int incrZ = (z!=null ? tz.elementWiseStride() : 0);
            op.getTransformOpDataBufferAction(threshold,tx.length(),x.data(),(y!=null?y.data():null), (z!=null?z.data():null),
                    offsetX,offsetY,offsetZ,incrX,incrY,incrZ).invoke();
            return;
        } else {
            List<TransformDataBufferAction> blockList = new ArrayList<>(nTensors);
            if(x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                DataBuffer dx = x.data();
                if(y==null){
                    if(x==z){
                        //x=Op(x)
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            TransformDataBufferAction task = op.getTransformOpDataBufferAction(threshold,n,dx,null,dx,offsetX,
                                        0,offsetX,incrX,0,incrX);
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
                            TransformDataBufferAction task = op.getTransformOpDataBufferAction(threshold,n,dx,null,dz,offsetX,
                                    0,offsetZ,incrX,0,incrZ);
                            task.fork();
                            blockList.add(task);
                        }
                    }
                } else {
                    DataBuffer dy = y.data();
                    OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                    int incrY = tsy.elementWiseStride;
                    if(x==z){
                        //x=Op(x,y)
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            TransformDataBufferAction task = op.getTransformOpDataBufferAction(threshold,n,dx,dy,dx,offsetX,
                                    offsetY,offsetX,incrX,incrY,incrX);
                            task.fork();
                            blockList.add(task);
                        }
                    } else {
                        //z=Op(x,y)
                        DataBuffer dz = z.data();
                        OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                        int incrZ = tsz.getElementWiseStride();
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                            TransformDataBufferAction task = op.getTransformOpDataBufferAction(threshold,n,dx,dy,dz,offsetX,
                                    offsetY,offsetZ,incrX,incrY,incrZ);
                            task.fork();
                            blockList.add(task);
                        }
                    }
                }
            } else {
                //Use general purpose tensor calculation for everything else
                for (int i = 0; i < nTensors; i++) {
                    TransformDataBufferAction task = op.getTransformOpDataBufferAction(i,tensorDim,threshold,x,y,z);
                    task.fork();
                    blockList.add(task);
                }
            }

            //Block until all are completed
            for(TransformDataBufferAction task : blockList){
                task.join();
            }
        }
    }
}
