package org.nd4j.linalg.api.ops.aggregates;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Aggregates are ops that work with custom operands,
 * that are not limited to traditional X, Y and Z constraints.
 *
 * @author raver119@gmail.com
 */
public interface Aggregate {

    /**
     *
     * @return
     */
    String name();

    /**
     *
     * @return
     */
    int opNum();


    /**
     *
     * @param result
     */
    void setFinalResult(Number result);

    /**
     *
     * @return
     */
    Number getFinalResult();

    /**
     *
     * @return
     */
    List<INDArray> getArguments();

    /**
     *
     * @return
     */
    List<DataBuffer> getShapes();

    /**
     *
     * @return
     */
    List<Integer> getIndexingArguments();

    /**
     *
     * @return
     */
    List<Number> getRealArguments();

    /**
     *
     * @return
     */
    List<int[]> getIntArrayArguments();

    /*
       Methods related to batch memory manipulations
     */

    /**
     * This method returns maximum number of shapes being passed per Aggregate
     *
     * @return
     */
    int maxArguments();

    /**
     * This method returns maximum number of shapes being passed per Aggregate
     *
     * @return
     */
    int maxShapes();

    /**
     * This method returns maximum number of IntArrays being passed per Aggregate
     *
     * @return
     */
    int maxIntArrays();

    /**
     * This method returns maximum length for IntArrays, if any
     *
     * @return
     */
    int maxIntArraySize();

    /**
     * This method returns maximum number of IndexArguments per Aggregate
     *
     * @return
     */
    int maxIndexArguments();

    /**
     * This method returns maximum number of real (float/double) per Aggregate
     *
     * @return
     */
    int maxRealArguments();

    /**
     * This method returns amount of memory required for batch creation for this specific Aggregate
     * @return
     */
    long getRequiredBatchMemorySize();

    /**
     * This method returns amount of shared memory required for this specific Aggregate.
     * PLEASE NOTE: this method is especially important for
     * CUDA backend. On CPU backend it might be ignored, depending on Aggregate.
     *
     * @return
     */
    int getSharedMemorySize();

    /**
     * This method returns desired number of threads per Aggregate instance
     * PLEASE NOTE: this method is especially important for CUDA backend.
     * On CPU backend it might be ignored, depending on Aggregate.
     *
     * @return
     */
    int getThreadsPerInstance();
}
