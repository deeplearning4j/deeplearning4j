package org.deeplearning4j.linalg.jblas;

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




}
