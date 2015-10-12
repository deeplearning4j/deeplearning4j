package org.nd4j.linalg.api.parallel.tasks.cpu.scalar;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskFactory;
import org.nd4j.linalg.api.parallel.tasks.TaskFactoryProvider;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;

import java.util.ArrayList;
import java.util.List;

public class CPUScalarOpViaTensorAction extends BaseCPUAction {
    protected final ScalarOp op;

    public CPUScalarOpViaTensorAction(ScalarOp op, int threshold){
        super(op,threshold);
        this.op = op;
    }

    @Override
    public Void call() {
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
            Task<Void> task = new CPUScalarOpAction(op,threshold);
            task.invokeAsync();
            subTasks.add(task);
        } else {
            if(x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                if(x==z){
                    //x=Op(x)
                    for( int i=0; i<nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                        Task<Void> task = new CPUScalarOpAction(op,threshold,n,offsetX,offsetX,incrX,incrX);
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
                        Task<Void> task = new CPUScalarOpAction(op,threshold,n,offsetX,offsetZ,incrX,incrZ);
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            } else {
                //Use general purpose tensor calculation for everything else
                for (int i = 0; i < nTensors; i++) {
                    Task<Void> task = new CPUScalarOpAction(op,threshold,i,tensorDim);
                    task.invokeAsync();
                    subTasks.add(task);
                }
            }
        }
        return null;
    }
}
