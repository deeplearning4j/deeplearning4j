package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskFactory;
import org.nd4j.linalg.api.parallel.tasks.TaskFactoryProvider;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class CPUTransformOpViaTensorTask extends BaseTask<Void> {
    protected final TransformOp op;
    protected final int threshold;

    protected List<Task<Void>> subTasks;

    public CPUTransformOpViaTensorTask(TransformOp op, int threshold){
        this.op = op;
        this.threshold = threshold;
    }

    @Override
    public void invokeAsync() {
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

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
        subTasks = new ArrayList<>(nTensors);
        if(nTensors == 1){
            //Generally shouldn't be called if nTensors = 1, as this is a vector
            Task<Void> task = new CPUTransformOpAction(op,threshold);
            task.invokeAsync();
            subTasks.add(task);
        } else {
            if(x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                if(y==null){
                    if(x==z){
                        //x=Op(x)
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            Task<Void> task = new CPUTransformOpAction(op,threshold,n,offsetX,0,offsetX,incrX,0,incrX);
                            task.invokeAsync();
                            subTasks.add(task);
                        }
                    } else {
                        //z=Op(x)
                        OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                        int incrZ = tsz.getElementWiseStride();
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                            Task<Void> task = new CPUTransformOpAction(op,threshold,n,offsetX,0,offsetZ,incrX,0,incrZ);
                            task.invokeAsync();
                            subTasks.add(task);
                        }
                    }
                } else {
                    OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                    int incrY = tsy.elementWiseStride;
                    if(x==z){
                        //x=Op(x,y)
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            Task<Void> task = new CPUTransformOpAction(op,threshold,n,offsetX,offsetY,offsetX,incrX,incrY,incrX);
                            task.invokeAsync();
                            subTasks.add(task);
                        }
                    } else {
                        //z=Op(x,y)
                        OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                        int incrZ = tsz.getElementWiseStride();
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                            Task<Void> task = new CPUTransformOpAction(op,threshold,n,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
                            task.invokeAsync();
                            subTasks.add(task);
                        }
                    }
                }
            } else {
                //Use general purpose tensor calculation for everything else
                for (int i = 0; i < nTensors; i++) {
                    Task<Void> task = new CPUTransformOpAction(op,threshold,i,tensorDim);
                    task.invokeAsync();
                    subTasks.add(task);
                }
            }
        }
    }

    @Override
    public Void blockUntilComplete() {
        if(subTasks==null){
            //invokeAsync hasn't been called
            invokeAsync();
        }
        for(Task task : subTasks){
            task.blockUntilComplete();
        }
        return null;
    }

    @Override
    public Void call() {
        return null;    //Not applicable
    }
}
