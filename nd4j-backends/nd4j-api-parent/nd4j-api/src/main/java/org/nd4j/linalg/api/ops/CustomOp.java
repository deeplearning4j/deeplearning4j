package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * This interface describe "custom operations.
 * Originally these operations are designed for SameDiff, and execution within graph,
 * but we also want to provide option to use them with regular ND4J methods via NativeOpExecutioner
 *
 * @author raver119@gmail.com
 */
public interface CustomOp {
    /**
     * This method returns op opName as string
     * @return
     */
    String opName();

    /**
     * This method returns LongHash of the opName()
     * @return
     */
    long opHash();

    /**
     * This method returns true if op is supposed to be executed inplace
     * @return
     */
    boolean isInplaceCall();


    /**
     * Input arguments for
     * this op
     * @return
     */
    List<INDArray> getInputArguments();

    /**
     * Output arguments for this
     * @return
     */
    List<INDArray> getOutputArguments();

    /**
     * Integer input arguments for this
     * @return
     */
    List<Integer> getIArguments();

    /**
     * Floating point arguments
     * for this op.
     * The "T" stands for
     * a generic floating point type
     * in c++.
     * @return
     */
    List<Double> getTArguments();


    /**
     * Calculate the output shape for this op
     * @return
     */
    List<int[]> calculateOutputShape();
}
