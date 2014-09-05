package org.nd4j.linalg.api.ndarray;

/**
 * Slice wise operation
 *
 * @author Adam Gibson
 */
public interface SliceOp {
    /**
     * Operates on an ndarray slice
     * @param nd the result to operate on
     */
   void operate(DimensionSlice nd);

    /**
     * Operates on an ndarray slice
     * @param nd the result to operate on
     */
    void operate(INDArray nd);


}
