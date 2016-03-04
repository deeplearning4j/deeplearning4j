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
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

public class CPUScalarOpViaTensorAction extends BaseCPUAction {
    protected final ScalarOp op;

    public CPUScalarOpViaTensorAction(ScalarOp op, int threshold){
        super(op,threshold);
        this.op = op;
    }

    @Override
    public Void call() {
        //Callable/ExecutorService
        execute(false);
        return null;
    }


    @Override
    protected void compute() {
        //Fork join
        execute(true);
    }

    private void execute(final boolean forkJoin) {
        //Fork join
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        int tensorDim;
        if(y == null){
            if(x == z){
                //x=Op(x)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
            } else {
                //z=Op(x)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, z);
            }
        } else {
            if(x == z){
                //x=Op(x,y)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y);
            } else {
                //z=Op(x,y)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y, z);
            }
        }

        int nTensors = x.tensorssAlongDimension(tensorDim);
        List<RecursiveAction> fjTasks = null;
        if(forkJoin) fjTasks = new ArrayList<>(nTensors);
        else subTasks = new ArrayList<>(nTensors);
        if(nTensors == 1) {
            //Generally shouldn't be called if nTensors = 1, as this is a vector
            CPUScalarOpAction task = new CPUScalarOpAction(op,threshold);
            if(forkJoin){
                task.invoke();
                return;
            } else {
                task.invokeAsync();
                subTasks.add(task);
            }
        } else {
            if(x.rank() == 2) {
                //Use fast tensor calculation for 2d
                OpExecutionerUtil.Tensor1DStats tsx = OpExecutionerUtil.get1DTensorStats(x, tensorDim);
                int n = tsx.getTensorLength();
                int incrX = tsx.getElementWiseStride();
                if(x == z){
                    //x=Op(x)
                    for( int i = 0; i < nTensors; i++) {
                        int offsetX = tsx.getFirstTensorOffset() + i * tsx.getTensorStartSeparation();
                        CPUScalarOpAction task = new CPUScalarOpAction(op,threshold,n,offsetX,offsetX,incrX,incrX);
                        if(forkJoin){
                            task.fork();
                            fjTasks.add(task);
                        } else {
                            task.invokeAsync();
                            subTasks.add(task);
                        }
                    }
                } else {
                    //z=Op(x)
                    OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                    int incrZ = tsz.getElementWiseStride();
                    for( int i = 0; i < nTensors; i++){
                        int offsetX = tsx.getFirstTensorOffset() + i * tsx.getTensorStartSeparation();
                        int offsetZ = tsz.getFirstTensorOffset() + i * tsz.getTensorStartSeparation();
                        CPUScalarOpAction task = new CPUScalarOpAction(op,threshold,n,offsetX,offsetZ,incrX,incrZ);
                        if(forkJoin){
                            task.fork();
                            fjTasks.add(task);
                        } else {
                            task.invokeAsync();
                            subTasks.add(task);
                        }
                    }
                }
            } else {
                //Use general purpose tensor calculation for everything else
                for (int i = 0; i < nTensors; i++) {
                    CPUScalarOpAction task = new CPUScalarOpAction(op,threshold,i,tensorDim);
                    if(forkJoin){
                        task.fork();
                        fjTasks.add(task);
                    } else {
                        task.invokeAsync();
                        subTasks.add(task);
                    }
                }
            }
        }
        if(forkJoin) {
            for (RecursiveAction t : fjTasks) {
                t.join();
            }
        }
    }
}
