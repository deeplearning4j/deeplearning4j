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
import org.nd4j.linalg.api.parallel.tasks.cpu.misc.CPUCol2ImTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.misc.CPUIm2ColTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.scalar.CPUScalarOpAction;
import org.nd4j.linalg.api.parallel.tasks.cpu.scalar.CPUScalarOpViaTensorAction;
import org.nd4j.linalg.api.parallel.tasks.cpu.transform.CPUTransformAlongDimensionTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.transform.CPUTransformOpAction;
import org.nd4j.linalg.api.parallel.tasks.cpu.transform.CPUTransformOpViaTensorTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.vector.CpuBroadcastOp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/** TaskFactory for CPU backends */
public class CPUTaskFactory implements TaskFactory {
    public static final String PARALLEL_THRESHOLD = "org.nd4j.parallel.cpu.threshold";
    private static Logger log = LoggerFactory.getLogger(CPUTaskFactory.class);
    protected int parallelThreshold = 1024;

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

    /** Set the threshold for breaking up CPU tasks for parallel execution.
     * If a given task/op has length greater than the threshold, it will be broken
     * down into a number of smaller tasks for execution.
     * @param threshold New threshold to use for parallel execution
     */
    public void setParallelThreshold(int threshold){
        this.parallelThreshold = threshold;
    }

    /** Get the current threshold for parallel execution of tasks */
    public int getParallelThreshold(){
        return parallelThreshold;
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
            else{
                canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,z);
                if(!Arrays.equals(x.shape(), z.shape())){
                    throw new IllegalArgumentException("Shapes do not match: x.shape=" + Arrays.toString(x.shape()) +
                            ", z.shape=" + Arrays.toString(z.shape()));
                }
            }
        } else {
            if( x== z) {
                canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,y);
                if(!Arrays.equals(x.shape(), y.shape())){
                    throw new IllegalArgumentException("Shapes do not match: x.shape=" + Arrays.toString(x.shape()) +
                            ", y.shape="+Arrays.toString(y.shape()));
                }
            }
            else{
                canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,y,z);
                if(!Arrays.equals(x.shape(), y.shape()) || !Arrays.equals(x.shape(), z.shape())){
                    throw new IllegalArgumentException("Shapes do not match: x.shape=" + Arrays.toString(x.shape()) +
                            ", y.shape="+Arrays.toString(y.shape()) + ", z.shape=" + Arrays.toString(z.shape()));
                }
            }
        }

        if(canDoDirectly) {
            return new CPUTransformOpAction(op, parallelThreshold);
        } else {
            //Need to break into tensors
            return new CPUTransformOpViaTensorTask(op, parallelThreshold);
        }
    }

    @Override
    public Task<Void> getTransformAction(TransformOp op, int... dimension ){
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();
        if(y == null){
            if (x != z && !Arrays.equals(x.shape(), z.shape())) {
                throw new IllegalArgumentException("Shapes do not match: x.shape=" + Arrays.toString(x.shape()) +
                        ", z.shape=" + Arrays.toString(z.shape()));
            }
        }
        else {
            if(x == z ) {
                if(!Arrays.equals(x.shape(), y.shape())){
                    throw new IllegalArgumentException("Shapes do not match: x.shape="+Arrays.toString(x.shape()) +
                            ", y.shape=" + Arrays.toString(y.shape()));
                }
            }
            else {
                if(!Arrays.equals(x.shape(), y.shape()) || !Arrays.equals(x.shape(), z.shape())){
                    throw new IllegalArgumentException("Shapes do not match: x.shape="+Arrays.toString(x.shape()) +
                            ", y.shape=" + Arrays.toString(y.shape()) + ", z.shape="+Arrays.toString(z.shape()));
                }
            }
        }

        return new CPUTransformAlongDimensionTask(op,parallelThreshold,dimension);
    }

    @Override
    public Task<Void> getScalarAction(ScalarOp op) {
        INDArray x = op.x();
        INDArray z = op.z();

        //If can do directly...
        boolean canDoDirectly;
        if(x==z) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x);
        else{
            canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x,z);
            if(!Arrays.equals(x.shape(), z.shape())) {
                throw new IllegalArgumentException("Shapes do not match: x.shape= " + Arrays.toString(x.shape()) +
                        ", z.shape="+Arrays.toString(z.shape()));
            }
        }

        if(canDoDirectly){
            return new CPUScalarOpAction(op, parallelThreshold);
        } else {
            //Need to break into tensors
            return new CPUScalarOpViaTensorAction(op, parallelThreshold);
        }
    }

    @Override
    public Task<Double> getAccumulationTask(Accumulation op, boolean outerTask) {
        INDArray x = op.x();
        INDArray y = op.y();

        boolean canDoDirectly;
        if (y == null)
            canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x);
        else {
            canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x, y);
            if(!Arrays.equals(x.shape(), y.shape())) {
                throw new IllegalArgumentException("Shapes do not match: x.shape= " + Arrays.toString(x.shape()) +
                        ", y.shape= " + Arrays.toString(y.shape()));
            }
        }

        if (canDoDirectly) {
            return new CPUAccumulationTask(op, parallelThreshold,outerTask);
        } else {
            //Need to break the accumulation into tensors first
            return new CPUAccumulationViaTensorTask(op,parallelThreshold,outerTask);
        }
    }
    @Override
    public Task<Double> getAccumulationTask(Accumulation op) {
        return getAccumulationTask(op,true);
    }

    @Override
    public Task<INDArray> getAccumulationTask(Accumulation op, int... dimension) {
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();
        if(y == null){
            if (x != z && !Arrays.equals(x.shape(), z.shape())) {
                throw new IllegalArgumentException("Shapes do not match: x.shape=" + Arrays.toString(x.shape()) +
                        ", z.shape=" + Arrays.toString(z.shape()));
            }
        } else {
            if(x == z) {
                if(!Arrays.equals(x.shape(), y.shape())){
                    throw new IllegalArgumentException("Shapes do not match: x.shape="+Arrays.toString(x.shape()) +
                            ", y.shape="+Arrays.toString(y.shape()));
                }
            }
            else{
                if(!Arrays.equals(x.shape(), y.shape()) || !Arrays.equals(x.shape(), z.shape())){
                    throw new IllegalArgumentException("Shapes do not match: x.shape="+Arrays.toString(x.shape()) +
                            ", y.shape="+Arrays.toString(y.shape()) + ", z.shape="+Arrays.toString(z.shape()));
                }
            }
        }

        return new CPUAccumulationAlongDimensionTask(op,parallelThreshold,dimension);
    }

    @Override
    public Task<Pair<Double,Integer>> getIndexAccumulationTask(IndexAccumulation op) {
        INDArray x = op.x();
        INDArray y = op.y();

        if(y != null && !Arrays.equals(x.shape(),y.shape())){
            throw new IllegalArgumentException("Shapes do not match: x.shape="+Arrays.toString(x.shape()) +
                    ", y.shape="+Arrays.toString(y.shape()));
        }

        //Due to the indexing being done on row-major order: can only do directly on C order
        boolean canDoDirectly;
        if(x.isVector()){
            canDoDirectly = true;
        } else if(x.ordering() == 'c' ) {
            if (y == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x, y);
        } else {
            canDoDirectly = false;
        }

        if(canDoDirectly){
            return new CPUIndexAccumulationTask(op,parallelThreshold,true);
        } else {
            return new CPUIndexAccumulationViaTensorTask(op,parallelThreshold,true);
        }
    }

    @Override
    public Task<INDArray> getIndexAccumulationTask(IndexAccumulation op, int... dimension) {
        INDArray x = op.x();
        INDArray y = op.y();

        if(y != null && !Arrays.equals(x.shape(),y.shape())){
            throw new IllegalArgumentException("Shapes do not match: x.shape="+Arrays.toString(x.shape()) +
                    ", y.shape="+Arrays.toString(y.shape()));
        }

        return new CPUIndexAccumulationAlongDimensionTask(op,parallelThreshold,dimension);
    }

    @Override
    public Task<Void> getBroadcastOpAction(BroadcastOp op) {
        INDArray x = op.x();
        INDArray y = op.y();
        if(x.size(op.getDimension()[0]) != y.length()){
            throw new IllegalArgumentException("Shapes do not match: x.shape="+Arrays.toString(x.shape()) +
                    ", y.shape="+Arrays.toString(y.shape()) + ", y should be vector with length=x.size(" + op.getDimension() + ")");
        }
        return new CpuBroadcastOp(op,parallelThreshold);
    }

    @Override
    public Task<INDArray> getIm2ColTask(INDArray img, int kernelHeight, int kernelWidth, int strideY, int strideX, int padHeight, int padWidth, boolean coverAll) {
        return new CPUIm2ColTask(img, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth, coverAll, parallelThreshold);
    }

    @Override
    public Task<INDArray> getCol2ImTask(INDArray col, int strideY, int strideX, int padHeight, int padWidth, int imgHeight, int imgWidth) {
        return new CPUCol2ImTask(col, strideY, strideX, padHeight, padWidth, imgHeight, imgWidth, parallelThreshold);
    }
}
