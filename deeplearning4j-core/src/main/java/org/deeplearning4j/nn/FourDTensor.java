package org.deeplearning4j.nn;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

/**
 * Created by agibsonccc on 5/3/14.
 */
public class FourDTensor extends Tensor {

    protected int numTensor;

    /**
     * Creates this tensor with the specified number of rows, columns and slices
     * Note that this will throw an illegal argument exception if any of the given
     * params are less than 1
     *
     * @param rows
     * @param columns
     * @param slices
     * @param numTensor the number of tensors
     */
    public FourDTensor(int rows, int columns, int slices,int numTensor) {
        super(rows, columns, slices * numTensor);
        this.numTensor = numTensor;
    }

    public FourDTensor(DoubleMatrix t) {
        super(t);
    }

    /**
     * Initializes this tensor's data
     * and the number of slices and per matrix rows
     *
     * @param t the tensor to initialize with
     */
    public FourDTensor(Tensor t) {
        super(t);
    }

    /**
     * Retrieves the tensor at the specified index
     * @param tensor the tensor to retrieve
     * @return the tensor at the specified index
     */
    public Tensor getTensor(int tensor) {
        DoubleMatrix ret = get(RangeUtils.interval(tensor * slices(),tensor * slices * rows),RangeUtils.interval(0,columns()));
        return new Tensor(ret,slices(),rows());
    }
    
    public DoubleMatrix getSliceOfTensor(int tensor, int slice) {
        DoubleMatrix ret = get(RangeUtils.interval(tensor * slices(),tensor * slices * rows),RangeUtils.interval(0,columns()));
        return new Tensor(ret,slices(),rows()).getSlice(slice);
    }


}
