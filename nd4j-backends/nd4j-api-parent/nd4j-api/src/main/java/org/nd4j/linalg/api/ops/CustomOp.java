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
     * This method returns op name as string
     * @return
     */
    String opName();

    /**
     * This method returns LongHash of the opName()
     * @return
     */
    long opHash();


    List<INDArray> getInputArguments();

    List<INDArray> getOutputArguments();

    List<Integer> getIArguments();

    List<Double> getTArguments();
}
