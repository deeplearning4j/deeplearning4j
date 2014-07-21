package org.deeplearning4j.nn.linalg;

import org.apache.commons.math3.distribution.RealDistribution;
import org.jblas.FloatMatrix;
import org.jblas.ranges.Range;
import org.jblas.ranges.RangeUtils;

import static org.deeplearning4j.util.MatrixUtil.createBasedOn;

/**
 * Four dimensional tensor
 * @author Adam Gibson
 */
public class FloatFourDTensor extends FloatTensor {
    //number of tensors for the fourth dimension
    protected int numTensor;

    public FloatFourDTensor() {}

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
    public FloatFourDTensor(int rows, int columns, int slices, int numTensor) {
        super(rows, columns, slices * numTensor);
        this.numTensor = numTensor;
    }

    /**
     * Initializes this tensor as a t.rows x 1 x 1 x 1 4d tensor
     * @param t the matrix to use for data
     * @param copy whether to copy the input matrix
     */
    public FloatFourDTensor(FloatMatrix t, boolean copy) {
        this.slices = 1;
        this.perMatrixRows = 1;

        if(copy) {
            this.data = new float[t.length];
            System.arraycopy(t.data,0,this.data,0,this.data.length);
        }
        else
            this.data = t.data;
    }

    /**
     * Initializes this tensor as a t.rows x 1 x 1 x 1 4d tensor
     * @param t the matrix to use for data
     */
    public FloatFourDTensor(FloatMatrix t) {
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
    public FloatFourDTensor(FloatMatrix t, int rows, int columns, int slices, int tensor, boolean copy) {
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
    public FloatFourDTensor(FloatMatrix t, int rows, int columns, int slices, int tensor) {
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
    public FloatFourDTensor(FloatTensor t) {
        super(t);
    }

    /**
     * Retrieves the tensor at the specified index
     * @param tensor the tensor to retrieve
     * @return the tensor at the specified index
     */
    public FloatTensor getTensor(int tensor) {
        int tensorIndex = tensor *  slices();
        int end = tensorIndex + slices();
        FloatMatrix ret = get(RangeUtils.interval(tensorIndex ,end),RangeUtils.interval(0,columns()));
        return new FloatTensor(ret,slices(),rows());
    }

    /**
     * Sets the tensor at the specified index
     * @param tensor the tensor to set
     * @param set the new tensor
     * @return the tensor at the specified index
     */
    public FloatTensor setTensor(int tensor,FloatTensor set) {
        int tensorIndex = tensor *  slices();
        int end = tensorIndex + slices();
        Range rows = RangeUtils.interval(tensorIndex ,end);
        Range columns = RangeUtils.interval(0,columns());
        put(rows,columns,set);

        FloatMatrix ret = get(RangeUtils.interval(tensorIndex ,end),RangeUtils.interval(0,columns()));
        return new FloatTensor(ret,slices(),rows());
    }


    /**
     * Assigns an element at the specific tensor,slice,row,column
     * @param tensor the tensor to assign to
     * @param slice the slice to assign to
     * @param row the row to assign to
     * @param column the column to assign to
     * @param element the element to assign
     */
    public void put(int tensor,int slice,int row,int column,float element) {
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
    public float get(int tensor,int slice,int row,int column) {
        return getSliceOfTensor(tensor,slice).get(row,column);
    }

    /**
     * Sets the slice of the given tensor
     * @param tensor the tensor to insert in to
     * @param slice the slice to set
     * @param put the matrix to put
     */
    public void put(int tensor,int slice,FloatMatrix put) {
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
    public FloatMatrix shape() {
        FloatMatrix ret = new FloatMatrix(1,4);
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
    public FloatMatrix getSliceOfTensor(int tensor, int slice) {
        int tensorIndex = tensor *  slices();
        int end = tensorIndex + slices();
        FloatMatrix ret = get(RangeUtils.interval(tensorIndex,end),RangeUtils.interval(0,columns()));
        return new FloatTensor(ret,slices(),rows()).getSlice(slice);
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
    public static FloatFourDTensor zeros(int rows, int cols,int slices,int numTensor) {
        return new FloatFourDTensor(rows,cols,slices,numTensor);

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
    public static FloatFourDTensor rand(int rows,int cols,int slices,int numTensor,RealDistribution sample) {
        FloatFourDTensor tensor = new FloatFourDTensor(rows,cols,slices,numTensor);
        for(int i = 0; i < tensor.rows; i++) {
            double[] sample2 = sample.sample(cols);
            float[] put = new float[sample2.length];
            for(int j = 0; j < sample2.length; j++)
                put[j] = (float) sample2[j];
            tensor.putRow(i,new FloatMatrix(put));
        }
        return tensor;
    }




    /**
     * Add a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor addi(FloatMatrix other) {
        return createBasedOn(super.addi(other), this);
    }

    /**
     * Add a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor add(FloatMatrix other) {
        return createBasedOn(super.add(other), this);
    }
    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor add(float v) {
        return createBasedOn(super.add(v), this);
    }
    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor addi(float v) {
        return createBasedOn(super.addi(v), this);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor subi(FloatMatrix other) {
        return createBasedOn(super.subi(other), this);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor sub(FloatMatrix other) {
        return createBasedOn(super.sub(other), this);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor subi(float v) {
        return createBasedOn(super.subi(v), this);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor sub(float v) {
        return createBasedOn(super.sub(v), this);
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor muli(FloatMatrix other) {
        return createBasedOn(super.muli(other), this);
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor mul(FloatMatrix other) {
        return createBasedOn(super.mul(other), this);
    }
    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor mul(float v) {
        return createBasedOn(super.mul(v), this);
    }

    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor muli(float v) {
        return createBasedOn(super.muli(v), this);
    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor divi(FloatMatrix other) {
        return createBasedOn(super.divi(other), this);
    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatFourDTensor div(FloatMatrix other) {
        return createBasedOn(super.div(other), this);
    }
    /**
     * Elementwise divide by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor div(float v) {
        return createBasedOn(super.div(v), this);
    }

    /**
     * Elementwise divide by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatFourDTensor divi(float v) {
        return createBasedOn(super.divi(v), this);
    }

    public int getNumTensor() {
        return numTensor;
    }

    public void setNumTensor(int numTensor) {
        this.numTensor = numTensor;
    }


    public static FloatFourDTensor ones(int rows,int columns,int slices,int tensors) {
        FloatFourDTensor ret = new FloatFourDTensor(rows,columns,slices,tensors);
        ret.assign(1);
        return ret;
    }

}
