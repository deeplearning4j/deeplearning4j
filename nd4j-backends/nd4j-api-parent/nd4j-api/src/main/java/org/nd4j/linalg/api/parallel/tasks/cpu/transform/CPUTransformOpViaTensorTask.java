package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;

public class CPUTransformOpViaTensorTask extends BaseCPUAction {
    protected final TransformOp op;

    public CPUTransformOpViaTensorTask(TransformOp op, int threshold){
        super(threshold,0,0,0,0,0,0,0);
        //Zeros: Don't care about these values, as they aren't used anyway (creating subtasks here)
        //get required offsets etc from either Tensor1DStats OR via doing tensor in CPUTransformOpAction (with doTensorFirst == true)
        //Don't use other consructor -> does reshape calcs etc that we don't need here
        this.op = op;
    }

    @Override
    public Void call() {
        //Callable / ExecutorService
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
        List<RecursiveAction> fjTasks = null;
        if(forkJoin) fjTasks = new ArrayList<>(nTensors);
        else subTasks = new ArrayList<>(nTensors);

        if(nTensors == 1){
            //Generally shouldn't be called if nTensors = 1, as this is a vector
            CPUTransformOpAction task = new CPUTransformOpAction(op,threshold);
            if(forkJoin){
                task.invoke();
            } else {
                task.invokeAsync();
                subTasks.add(task);
            }
            return;
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
                            CPUTransformOpAction task = new CPUTransformOpAction(op,threshold,n,offsetX,0,offsetX,incrX,0,incrX);
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
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                            CPUTransformOpAction task = new CPUTransformOpAction(op,threshold,n,offsetX,0,offsetZ,incrX,0,incrZ);
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
                    OpExecutionerUtil.Tensor1DStats tsy = OpExecutionerUtil.get1DTensorStats(y,tensorDim);
                    int incrY = tsy.elementWiseStride;
                    if(x==z){
                        //x=Op(x,y)
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            CPUTransformOpAction task = new CPUTransformOpAction(op,threshold,n,offsetX,offsetY,offsetX,incrX,incrY,incrX);
                            if(forkJoin){
                                task.fork();
                                fjTasks.add(task);
                            } else {
                                task.invokeAsync();
                                subTasks.add(task);
                            }
                        }
                    } else {
                        //z=Op(x,y)
                        OpExecutionerUtil.Tensor1DStats tsz = OpExecutionerUtil.get1DTensorStats(z, tensorDim);
                        int incrZ = tsz.getElementWiseStride();
                        for( int i=0; i<nTensors; i++){
                            int offsetX = tsx.getFirstTensorOffset() + i*tsx.getTensorStartSeparation();
                            int offsetY = tsy.getFirstTensorOffset() + i*tsy.getTensorStartSeparation();
                            int offsetZ = tsz.getFirstTensorOffset() + i*tsz.getTensorStartSeparation();
                            CPUTransformOpAction task = new CPUTransformOpAction(op,threshold,n,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
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
            } else {
                //Use general purpose tensor calculation for everything else
                for (int i = 0; i < nTensors; i++) {
                    CPUTransformOpAction task = new CPUTransformOpAction(op,threshold,i,tensorDim);
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
