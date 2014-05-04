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
        int tensorIndex = tensor *  slices();
        int end = tensorIndex + slices();
        DoubleMatrix ret = get(RangeUtils.interval(tensorIndex ,end),RangeUtils.interval(0,columns()));
        return new Tensor(ret,slices(),rows());
    }


    /**
     * Assigns an element at the specific tensor,slice,row,column
     * @param tensor the tensor to assign to
     * @param slice the slice to assign to
     * @param row the row to assign to
     * @param column the column to assign to
     * @param element the element to assign
     */
    public void put(int tensor,int slice,int row,int column,double element) {
        int tensorIndex = tensor *  slices();
        put(tensorIndex  + slice  + row, column,element);
    }

    /**
     * Gets an individual element
     * @param tensor the tensor to retrieve from
     * @param slice the slice of the tensor to retrieve from
     * @param row the row of the element
     * @param column the column of the element
     * @return
     */
    public double get(int tensor,int slice,int row,int column) {
        return getSliceOfTensor(tensor,slice).get(row,column);
    }


    /**
     * Returns a slice of a tensor
     * @param tensor the tensor to get the slice of
     * @param slice the slice of the tensor to get
     * @return the slice of the specified tensor
     */
    public DoubleMatrix getSliceOfTensor(int tensor, int slice) {
        int tensorIndex = tensor *  slices();
        int end = tensorIndex + slices();
        DoubleMatrix ret = get(RangeUtils.interval(tensorIndex,end),RangeUtils.interval(0,columns()));
        return new Tensor(ret,slices(),rows()).getSlice(slice);
    }

    /**
     * Returns the number of tensors in this tensor
     * @return the number of tensors in this tensor
     */
    public int numTensors() {
        return numTensor;
    }



}
