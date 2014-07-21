package org.deeplearning4j.nn.linalg;

import org.apache.commons.math3.distribution.RealDistribution;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.Range;
import org.jblas.ranges.RangeUtils;
import static org.deeplearning4j.util.MatrixUtil.createBasedOn;
/**
 * Four dimensional tensor
 * @author Adam Gibson
 */
public class FourDTensor extends Tensor {
    //number of tensors for the fourth dimension
    protected int numTensor;

    public FourDTensor() {}

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
     * @param copy whether to copy the input matrix
     */
    public FourDTensor(DoubleMatrix t,boolean copy) {
        this.slices = 1;
        this.perMatrixRows = 1;

        if(copy) {
            this.data = new double[t.length];
            System.arraycopy(t.data,0,this.data,0,this.data.length);
        }
        else
            this.data = t.data;
    }

    /**
     * Initializes this tensor as a t.rows x 1 x 1 x 1 4d tensor
     * @param t the matrix to use for data
     */
    public FourDTensor(DoubleMatrix t) {
        this(t,true);
    }


    /**
     * Initializes this four d tensor with the given data and
     * the specified dimensions
     * @param t the baseline data for this tensor
     * @param rows the number of rows per slice
     * @param columns the number of columns for the tensor
     * @param slices the number of slices per tensor
     * @param tensor the number of tensors for this tensor
     * @param copy whether to copy the input data or reference it directly (DANGEROUS)
     */
    public FourDTensor(DoubleMatrix t,int rows,int columns,int slices,int tensor,boolean copy) {
        super(t,rows,columns,slices * tensor,copy);

        this.perMatrixRows = rows;
        this.columns = columns;
        this.slices = slices;
        this.numTensor = tensor;

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
     * Sets the tensor at the specified index
     * @param tensor the tensor to set
     * @param set the new tensor
     * @return the tensor at the specified index
     */
    public Tensor setTensor(int tensor,Tensor set) {
        int tensorIndex = tensor *  slices();
        int end = tensorIndex + slices();
        put(RangeUtils.interval(tensorIndex ,end),RangeUtils.interval(0,columns()),set);
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
        int tensorIndex = tensor *  slices();
        return super.get(tensorIndex  + slice  + row, column);
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
        //row of the tensor
        int row = tensorIndex * slice;
        Range rows = RangeUtils.interval(row,row + perMatrixRows);
        Range columns = RangeUtils.interval(0,columns());
        DoubleMatrix ret = get(rows,columns);
        return ret;
    }

    /**
     * Returns the number of tensors in this tensor
     * @return the number of tensors in this tensor
     */
    public int numTensors() {
        return numTensor;
    }



    /**
     * Returns a zero tensor
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @param numTensor the number of tensors
     * @return the tensor with the specified slices and the zeros matrices
     */
    public static FourDTensor zeros(int rows, int cols,int slices,int numTensor) {
        return new FourDTensor(rows,cols,slices,numTensor);

    }

    /**
     * Creates a random fourd tensor
     * @param rows the the number of rows
     * @param cols the number of columns
     * @param slices the number of slices per tensor
     * @param numTensor the number of tensors
     * @param sample the distribution to sample from
     * @return a randomly initialized tensor based on the passed in probability distribution
     */
    public static FourDTensor rand(int rows,int cols,int slices,int numTensor,RealDistribution sample) {
        FourDTensor tensor = new FourDTensor(rows,cols,slices,numTensor);
        for(int i = 0; i < tensor.rows; i++) {
            tensor.putRow(i,new DoubleMatrix(sample.sample(cols)));
        }
        return tensor;
    }




    /**
     * Add a matrix (in place).
     *
     * @param other
     */
    @Override
    public FourDTensor addi(DoubleMatrix other) {
        return createBasedOn(super.addi(other), this);
    }

    /**
     * Add a matrix (in place).
     *
     * @param other
     */
    @Override
    public FourDTensor add(DoubleMatrix other) {
        return createBasedOn(super.add(other), this);
    }
    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor add(double v) {
        return createBasedOn(super.add(v), this);
    }
    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor addi(double v) {
        return createBasedOn(super.addi(v), this);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public FourDTensor subi(DoubleMatrix other) {
        return createBasedOn(super.subi(other), this);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public FourDTensor sub(DoubleMatrix other) {
        return createBasedOn(super.sub(other), this);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor subi(double v) {
        return createBasedOn(super.subi(v), this);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor sub(double v) {
        return createBasedOn(super.sub(v), this);
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FourDTensor muli(DoubleMatrix other) {
        return createBasedOn(super.muli(other), this);
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FourDTensor mul(DoubleMatrix other) {
        return createBasedOn(super.mul(other), this);
    }
    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor mul(double v) {
        return createBasedOn(super.mul(v), this);
    }

    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor muli(double v) {
        return createBasedOn(super.muli(v), this);
    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FourDTensor divi(DoubleMatrix other) {
        return createBasedOn(super.divi(other), this);
    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public  FourDTensor div(DoubleMatrix other) {
        return createBasedOn(super.div(other), this);
    }
    /**
     * Elementwise divide by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor div(double v) {
        return createBasedOn(super.div(v), this);
    }

    /**
     * Elementwise divide by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FourDTensor divi(double v) {
        return createBasedOn(super.divi(v), this);
    }

    public int getNumTensor() {
        return numTensor;
    }

    public void setNumTensor(int numTensor) {
        this.numTensor = numTensor;
    }


    public static FourDTensor ones(int rows,int columns,int slices,int tensors) {
        FourDTensor ret = new FourDTensor(rows,columns,slices,tensors);
        ret.assign(1);
        return ret;
    }

}
