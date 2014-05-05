package org.deeplearning4j.nn;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.Range;
import org.jblas.ranges.RangeUtils;

/**
 * Four dimensional tensor
 * @author Adam Gibson
 */
public class FourDTensor extends Tensor {
    //number of tensors for the fourth dimension
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

    /**
     * Initializes this tensor as a t.rows x 1 x 1 x 1 4d tensor
     * @param t the matrix to use for data
     */
    public FourDTensor(DoubleMatrix t) {
        super(t);
    }

    /**
     * Initializes this four d tensor with the given data and
     * the specified dimensions
     * @param t the baseline data for this tensor
     * @param rows the number of rows per slice
     * @param columns the number of columns for the tensor
     * @param slices the number of slices per tensor
     * @param tensor the number of tensors for this tensor
     */
    public FourDTensor(DoubleMatrix t,int rows,int columns,int slices,int tensor) {
        super(t,rows,columns,slices * tensor);
        this.perMatrixRows = rows;
        this.columns = columns;
        this.slices = slices;
        this.numTensor = tensor;

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
     * Sets the slice of the given tensor
     * @param tensor the tensor to insert in to
     * @param slice the slice to set
     * @param put the matrix to put
     */
    public void put(int tensor,int slice,DoubleMatrix put) {
        int tensorIndex = tensor *  slices();
        //row of the tensor
        int row = tensorIndex * slice * rows();
        Range rows = RangeUtils.interval(row,row + put.rows);
        Range columns = RangeUtils.interval(0,put.columns);
        put(rows,columns,put);
    }

    /**
     * Returns the dimensions of this fourd tensor as a row matrix, in the following order:
     * rows,columns,slices,tensors
     * @return a 1 x 4 matrix with the dimensions of this tensor
     */
    @Override
    public DoubleMatrix shape() {
        DoubleMatrix ret = new DoubleMatrix(1,4);
        ret.put(0,rows());
        ret.put(1,columns());
        ret.put(2,slices());
        ret.put(3,numTensors());
        return ret;
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
