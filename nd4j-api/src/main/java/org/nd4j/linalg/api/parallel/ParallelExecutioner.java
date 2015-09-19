package org.nd4j.linalg.api.parallel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.Future;

/**
 * Executes operations in parallel
 *
 * @author Adam Gibson
 */
public interface ParallelExecutioner {

    /**
     *
     * @param arr
     * @param task
     * @param dimension
     */
    INDArray execBasedOnArraysAlongDimension(INDArray arr, Accumulation task, OpExecutioner executioner, int... dimension);


    /**
     *
     * @param arr
     * @param task
     * @param dimension
     */
    void execBasedOnArraysAlongDimension(INDArray arr, Op task, OpExecutioner executioner, int... dimension);

    /**
     *
     * @param arr
     * @param task
     */
    void execBasedOnSlices(INDArray arr,Op task,OpExecutioner executioner);

    /**
     *
     * @param arr
     * @param task
     * @param dimension
     */
    void execBasedOnArraysAlongDimension(INDArray arr, TaskCreator.INDArrayTask task, int... dimension);


    /**
     *
     * @param arr
     * @param task
     * @param dimension
     */
    void execBasedOnArraysAlongDimension(INDArray[] arr, TaskCreator.INDArrayTask task, int... dimension);

    /**
     *
     * @param arr
     * @param task
     */
    void execBasedOnSlices(INDArray arr,TaskCreator.INDArrayTask task);

    /**
     *
     * @param runnable
     * @return
     */
    Future exec(Runnable runnable);

    /**
     *
     * @param task
     * @param <T>
     */
    <T> void  exec(ForkJoinTask<T> task);

}
