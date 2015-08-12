package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class NDArrayIndexAll extends IntervalIndex {

    /**
     * @param inclusive whether to include the last number
     */
    public NDArrayIndexAll(boolean inclusive) {
        super(inclusive, 1);
    }


    @Override
    public void init(INDArray arr, int begin, int dimension) {
        this.begin = 0;
        this.end = arr.size(dimension);
    }


}
