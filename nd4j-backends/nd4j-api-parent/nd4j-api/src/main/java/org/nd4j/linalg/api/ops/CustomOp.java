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




    INDArray[] outputArguments();

    INDArray[] inputArguments();

    int[] iArgs();

    double[] tArgs();

    void addIArgument(int... arg);


    void removeIArgument(Integer arg);

    Integer getIArgument(int index);

    int numIArguments();

    void addTArgument(double... arg);

    void removeTArgument(Double arg);

    Double getTArgument(int index);

    int numTArguments();


    void addInputArgument(INDArray... arg);

    void removeInputArgument(INDArray arg);

    INDArray getInputArgument(int index);

    int numInputArguments();


    void addOutputArgument(INDArray... arg);

    void removeOutputArgument(INDArray arg);

    INDArray getOutputArgument(int index);

    int numOutputArguments();



    /**
     * Calculate the output shape for this op
     * @return
     */
    List<int[]> calculateOutputShape();
}
