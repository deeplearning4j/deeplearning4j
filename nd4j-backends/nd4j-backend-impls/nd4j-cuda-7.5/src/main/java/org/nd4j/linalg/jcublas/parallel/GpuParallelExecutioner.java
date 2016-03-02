package org.nd4j.linalg.jcublas.parallel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.parallel.DefaultParallelExecutioner;
import org.nd4j.linalg.api.parallel.ParallelExecutioner;
import org.nd4j.linalg.api.parallel.TaskCreator;
import org.nd4j.linalg.jcublas.ops.executioner.JCudaExecutioner;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.Future;

/**
 * Created by agibsonccc on 10/3/15.
 */
public class GpuParallelExecutioner extends DefaultParallelExecutioner {

    @Override
    public INDArray execBasedOnArraysAlongDimension(INDArray arr, Accumulation task, OpExecutioner executioner, int... dimension) {
        JCudaExecutioner jCudaExecutioner = (JCudaExecutioner) executioner;

        return null;
    }

    @Override
    public void execBasedOnArraysAlongDimension(INDArray arr, Op task, OpExecutioner executioner, int... dimension) {
        JCudaExecutioner jCudaExecutioner = (JCudaExecutioner) executioner;

    }

    @Override
    public void execBasedOnSlices(INDArray arr, Op task, OpExecutioner executioner) {
        JCudaExecutioner jCudaExecutioner = (JCudaExecutioner) executioner;

    }

    @Override
    public void execBasedOnArraysAlongDimension(INDArray arr, TaskCreator.INDArrayTask task, int... dimension) {

    }

    @Override
    public void execBasedOnArraysAlongDimension(INDArray[] arr, TaskCreator.INDArrayTask task, int... dimension) {

    }

    @Override
    public void execBasedOnSlices(INDArray arr, TaskCreator.INDArrayTask task) {

    }

    @Override
    public Future exec(Runnable runnable) {
        return null;
    }

    @Override
    public <T> void exec(ForkJoinTask<T> task) {

    }
}
