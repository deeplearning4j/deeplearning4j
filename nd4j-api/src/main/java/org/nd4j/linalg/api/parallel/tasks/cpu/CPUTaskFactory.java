package org.nd4j.linalg.api.parallel.tasks.cpu;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskFactory;
import org.nd4j.linalg.api.parallel.tasks.cpu.accumulation.CPUAccumulationAlongDimensionTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.accumulation.CPUAccumulationTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.accumulation.CPUAccumulationViaTensorTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum.CPUIndexAccumulationAlongDimensionTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum.CPUIndexAccumulationTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum.CPUIndexAccumulationViaTensorTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.scalar.CPUScalarOpAction;
import org.nd4j.linalg.api.parallel.tasks.cpu.scalar.CPUScalarOpViaTensorAction;
import org.nd4j.linalg.api.parallel.tasks.cpu.transform.CPUTransformAlongDimensionTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.transform.CPUTransformOpAction;
import org.nd4j.linalg.api.parallel.tasks.cpu.transform.CPUTransformOpViaTensorTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.vector.CPUVectorOp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class CPUTaskFactory implements TaskFactory {
    public static final String PARALLEL_THRESHOLD = "org.nd4j.parallel.cpu.threshold";
    private static Logger log = LoggerFactory.getLogger(CPUTaskFactory.class);
    protected int parallelThreshold = 8192;

    public CPUTaskFactory(){
        //Check if user has specified a parallel threshold via VM argument:
        String thresholdString = System.getProperty(PARALLEL_THRESHOLD,null);
        if(thresholdString != null){
            int threshold = -1;
            try{
                threshold = Integer.parseInt(thresholdString);
            }catch(NumberFormatException e ){
                log.warn("Error parsing CPUTaskFactory parallel threshold: \"" + thresholdString + "\"");
                log.warn("CPUTaskFactory parallel threshold set to default: " + parallelThreshold);
            }
            if(threshold != -1){
                if(threshold <= 0){
                    log.warn("Invalid CPUTaskFactory parallel threshold; using default: " + parallelThreshold);
                } else {
                    parallelThreshold = threshold;
                }
            }
        }
    }


    @Override
    public Task<Void> getTransformAction(TransformOp op) {
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        //If can do directly...
        boolean canDoDirectly;
        if(y == null){
            if(x==z) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,z);
        } else {
            if(x==z) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,y);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,y,z);
        }


        if(canDoDirectly){
            return new CPUTransformOpAction(op, parallelThreshold);
        } else {
            //Need to break into tensors
            return new CPUTransformOpViaTensorTask(op, parallelThreshold);
        }
    }

    @Override
    public Task<Void> getTransformAction(TransformOp op, int... dimension ){
        return new CPUTransformAlongDimensionTask(op,parallelThreshold,dimension);
    }

    @Override
    public Task<Void> getScalarAction(ScalarOp op) {
        INDArray x = op.x();
        INDArray z = op.z();

        //If can do directly...
        boolean canDoDirectly;
        if(x==z) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x);
        else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,z);

        if(canDoDirectly){
            return new CPUScalarOpAction(op, parallelThreshold);
        } else {
            //Need to break into tensors
            return new CPUScalarOpViaTensorAction(op, parallelThreshold);
        }
    }

    @Override
    public Task<Double> getAccumulationTask(Accumulation op) {
        INDArray x = op.x();
        INDArray y = op.y();

        boolean canDoDirectly;
        if (y == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x);
        else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x, y);

        if (canDoDirectly) {
            return new CPUAccumulationTask(op, parallelThreshold,true);
        } else {
            //Need to break the accumulation into tensors first
            return new CPUAccumulationViaTensorTask(op,parallelThreshold,true);
        }
    }

    @Override
    public Task<INDArray> getAccumulationTask(Accumulation op, int... dimension) {
        return new CPUAccumulationAlongDimensionTask(op,parallelThreshold,dimension);
    }

    @Override
    public Task<Pair<Double,Integer>> getIndexAccumulationTask(IndexAccumulation op) {
        INDArray x = op.x();
        INDArray y = op.y();

        boolean canDoDirectly;
        if(y==null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x);
        else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,y);

        if(canDoDirectly){
            return new CPUIndexAccumulationTask(op,parallelThreshold,true);
        } else {
            return new CPUIndexAccumulationViaTensorTask(op,parallelThreshold,true);
        }
    }

    @Override
    public Task<INDArray> getIndexAccumulationTask(IndexAccumulation op, int... dimension) {
        return new CPUIndexAccumulationAlongDimensionTask(op,parallelThreshold,dimension);
    }

    @Override
    public Task<Void> getVectorOpAction(VectorOp op) {
        return new CPUVectorOp(op,parallelThreshold);
    }
}
