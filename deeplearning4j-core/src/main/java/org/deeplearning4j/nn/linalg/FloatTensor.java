package org.deeplearning4j.nn.linalg;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.FloatMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.Range;
import org.jblas.ranges.RangeUtils;

import java.io.Serializable;
import java.util.Arrays;

import static org.deeplearning4j.util.MatrixUtil.createBasedOn;

/**
 * Tensor represents a set of matrices of all the same dimensions.
 * Based on the Recursive Neural Tensor Network by Socher et. al
 * @author Adam Gibson
 */
public class FloatTensor extends FloatMatrix implements Serializable {

    protected int slices,perMatrixRows;

    public FloatTensor(){}

    /**
     * Creates this tensor with the specified number of rows, columns and slices
     * Note that this will throw an illegal argument exception if any of the given
     * params are less than 1
     * @param baseLineMatrix the matrix to get the data from
     * @param rows the number of rows per slice
     * @param columns the number of columns in the matrix
     * @param slices the number of slices in the tensor
     * @param copy whether to copy the data or directly reference
     */
    public FloatTensor(FloatMatrix baseLineMatrix, int rows, int columns, int slices, boolean copy) {
        this.rows = rows * slices;
        this.columns = columns;
        if(copy) {
            this.data = new float[baseLineMatrix.length];
            System.arraycopy(baseLineMatrix.data,0,this.data,0,baseLineMatrix.length);
        }
        else
            this.data = baseLineMatrix.data;

        if(baseLineMatrix.length != rows * columns * slices)
            throw new IllegalArgumentException("Illegal matrix, amount of data does not match the specified dimensions");
        if(slices < 1)
            throw new IllegalArgumentException("Tensor has no slices");
        if(rows < 1)
            throw new IllegalArgumentException("Illegal number of rows");
        if(columns < 1)
            throw new IllegalArgumentException("Illegal number of columns");
        this.slices = slices;
        this.perMatrixRows = rows;
    }

    /**
     * Creates this tensor with the specified number of rows, columns and slices
     * Note that this will throw an illegal argument exception if any of the given
     * params are less than 1
     * @param baseLineMatrix the matrix to get the data from
     * @param rows the number of rows per matrix
     * @param columns the number of columns for the tensor
     * @param slices the number of slices in the tensor
     */
    public FloatTensor(FloatMatrix baseLineMatrix, int rows, int columns, int slices) {
       this(baseLineMatrix,rows,columns,slices,true);
    }



    /**
     * Creates this tensor with the specified number of rows, columns and slices
     * Note that this will throw an illegal argument exception if any of the given
     * params are less than 1
     * @param rows
     * @param columns
     * @param slices
     */
    public FloatTensor(int rows, int columns, int slices) {
        super(rows * slices,columns);
        if(slices < 1)
            throw new IllegalArgumentException("Tensor has no slices");
        if(rows < 1)
            throw new IllegalArgumentException("Illegal number of rows");
        if(columns < 1)
            throw new IllegalArgumentException("Illegal number of columns");
        this.slices = slices;
        this.perMatrixRows = rows;
    }

    /**
     * Initializes this tensor with 1 slice
     * and the number of slice rows equal
     * the number of rows in the passed in matrix
     * @param t the matrix to initialize this tensor with
     */
    public FloatTensor(FloatMatrix t, boolean copy) {
        this.rows  = t.rows;
        this.perMatrixRows = 1;
        this.columns = t.columns;
        if(copy) {
            this.data = new float[t.length];
            System.arraycopy(t.data,0,this.data,0,this.data.length);
            this.rows  = t.rows;
            this.perMatrixRows = 1;
            this.columns = t.columns;
        }
        else
            this.data = t.data;
    }
    /**
     * Initializes this tensor with 1 slice
     * and the number of slice rows equal
     * the number of rows in the passed in matrix
     * @param t the matrix to initialize this tensor with
     */
    public FloatTensor(FloatMatrix t) {
        this(t,false);
    }

    /**
     * Initializes the given matrix with the
     * number of slices
     * @param t the matrix to use as a base for
     *  this tensor
     * @param slices the number of slices of the matrix
     * @param rows the number of rows per slice
     * @param copy whether to copy the data or use a direct reference (DANGEROUS)
     */
    public FloatTensor(FloatMatrix t, int slices, int rows, boolean copy) {
        this.slices = slices;
        this.perMatrixRows = rows;
        if(copy) {
            this.data = new float[t.length];
            System.arraycopy(t.data,0,this.data,0,this.data.length);
            this.rows = t.rows;
            this.perMatrixRows = rows;
            this.columns = t.columns;
        }
        else {
            this.rows = rows;
            this.columns = t.columns;
            this.perMatrixRows = rows;
            this.data = t.data;
        }
    }

    /**
     * Initializes the given matrix with the
     * number of slices
     * @param t the matrix to use as a base for
     *  this tensor
     * @param slices the number of slices of the matrix
     * @param rows the number of rows per slice
     */
    public FloatTensor(FloatMatrix t, int slices, int rows) {
        this(t,slices,rows,false);
    }

    /**
     * Initializes this tensor's data
     * and the number of slices and per matrix rows
     * @param t the tensor to initialize with
     */
    public FloatTensor(FloatTensor t) {
        super(t.toArray2());
        this.slices = t.slices();
        this.perMatrixRows = t.perMatrixRows;
    }

    /**
     * Returns a 1 x 3 matrix with the dimensions of this tensor
     * in the following order:
     * rows,columns,slices
     * @return the dimensions of this tensor
     */
    public FloatMatrix shape() {
        FloatMatrix ret = new FloatMatrix(1,3);
        ret.put(0,rows());
        ret.put(1,columns());
        ret.put(2,slices());
        return ret;
    }



    private Range getColIndicesForSlice() {
        return RangeUtils.all();
    }
    /* Gets a block of the matrix such that the slice represents a subset of the rows in the matrix */
    private Range getRowIndicesForSlice(int slice) {
        int start = slice * rows();
        int end =  (slice * rows()) + rows();
        return RangeUtils.interval(start,end);
    }

    public FloatTensor sliceColumnSums() {
        FloatTensor ret = new FloatTensor(1,columns(),slices);
        for(int i = 0; i < slices(); i++) {
            ret.setSlice(i,getSlice(i).columnSums());
        }
        return ret;
    }



    /**
     * Sums the elements along the third dimension such that
     * each element's i,j is the sum of the element at
     * all the i,j's in the slice
     * @return the slice wise element sums
     */
    public FloatMatrix sliceElementSums() {
        FloatMatrix ret = new FloatMatrix(rows(),columns());
        for(int i = 0; i < ret.rows; i++) {
            for(int j = 0; j < ret.columns; j++) {
                float sum = 0;
                for(int slice = 0; slice < slices(); slice++) {
                    sum += getSlice(slice).get(i,j);
                }
                ret.put(i,j,sum);

            }


        }

        return ret;
    }


    public FloatTensor sliceRowSums() {
        FloatTensor ret = new FloatTensor(1,rows(),slices);
        for(int i = 0; i < slices(); i++) {
            FloatMatrix rowSums = getSlice(i).rowSums().transpose();
            ret.setSlice(i,rowSums);
        }
        return ret;
    }


    /**
     * Returns the given slice
     * @param index the index of the slice
     * @return the given slice
     */
    public FloatMatrix getSlice(int index) {
        if(index >= slices())
            throw new IllegalArgumentException("Unable to get slice " + index + " out of bounds");

        try {
            FloatMatrix slice =  get(RangeUtils.interval(index,index + rows()),RangeUtils.interval(0,columns()));
            return slice;

        } catch(Exception e) {
            throw new IllegalArgumentException("Unable to get a slice ",e);
        }
    }

    /**
     * Sets the slice, note that the given slice must be
     * the same dimensions
     * @param index the slice to set
     * @param slice the new slice
     */
    public void setSlice(int index,FloatMatrix slice) {
        if(slice.rows != rows() || slice.columns != columns()) {
            if(slice.rows < rows() || slice.columns < columns())
                slice = MatrixUtil.padWithZeros(slice,rows(),columns());
            else
                slice = MatrixUtil.truncate(slice,rows(),columns());
            if(slice.rows != rows() || slice.columns != columns())
                throw new IllegalStateException("WTF IS THIS");
        }

        put(RangeUtils.interval(index,index + rows()),RangeUtils.interval(0,slice.columns),slice);
    }



    /**
     * Clones this tensor
     * @return a copy of this tensor
     */
    @Override
    public FloatTensor dup() {
        return new FloatTensor(this);
    }


    /**
     * Assigns the corresponding slice to the passed in elements for each slice[i] in tensor
     * @param tensor the tensor to set
     * @param rowIndices the row indices for each slice
     * @param columnIndices the column indices for each slice
     */
    public void set(FloatTensor tensor,int[] rowIndices,int[] columnIndices) {
        for(int i = 0; i < slices(); i++) {
            setSlice(i,tensor);
        }

    }

    /**
     * Sets the passed in matrix to each of the row/column indices in each slice
     * @param toSet the matrix to set
     * @param rowIndices the row indices to set
     * @param columnIndices the column indices to set
     */
    public void set(FloatMatrix toSet,int[] rowIndices,int[] columnIndices) {
        for(int i = 0; i < slices(); i++) {
            setSlice(i, toSet);
        }

    }

    public FloatTensor columnSums() {
        return new FloatTensor(columnsSums());
    }



    /**
     * Returns this tensor as a matrix such that
     * all values in all slices are concacneated in to one matrix.
     * This matrix will be a t.rows() * t.slices() x t.columns() matrix
     * @return
     */
    public FloatMatrix toMatrix() {
        return this;
    }

    /**
     * The column sums of each matrix in this tensor
     * @return a tensor populated by the column sums of each matrix in the tensor
     */
    public FloatTensor columnsSums() {
        //each slice will have a column sum
        FloatTensor t = new FloatTensor(1,columns,slices);

        for(int i = 0; i < slices(); i++) {
            FloatMatrix sums = getSlice(i).columnSums();
            t.putRow(i,sums);

        }
        return t;
    }



    public void set(int slice,int i,int j,float val) {
        put(rows * slice + i,j,val);
    }

    public float get(int i,int j,int slice) {
        return get(i + slice,j);
    }

    /**
     * Gets the specified row from each slice
     * @param row the row to get from each slice
     * @return a slices() x column matrix of each slices row $row
     */
    public FloatMatrix getRows(int row) {
        int[] indices = new int[slices()];
        for(int i = 0; i < slices(); i++)
            indices[i] = row * i;
        FloatMatrix ret = get(indices);
        return ret;
    }


    /**
     * Gets the specified row from each slice
     * @param column the row to get from each slice
     * @return a slices() x column matrix of each slices row $row
     */
    public FloatMatrix getColumns(int column) {
        FloatMatrix rows = new FloatMatrix(slices(),columns());
        for(int i = 0; i < slices(); i++) {
            rows.putRow(i,getSlice(i).getColumn(column));
        }
        return rows;
    }



    /**
     * Gets the specified row from the specified slice
     * @param row the row to get from each slice
     * @param slice the slice to get
     * @return a slices() x column matrix of each slices row $row
     */
    public FloatMatrix getRow(int row,int slice) {
        FloatMatrix rows = get(getRowIndicesForSlice(slice),getColIndicesForSlice());
        return rows;
    }


    public FloatTensor ff() {
        FloatTensor ret = new FloatTensor(this);
        for(int i = 0; i < slices(); i++) {
            FloatMatrix slice = MatrixUtil.reverse(getSlice(i));
            setSlice(i,slice);
        }
        return ret;
    }

    /**
     * Gets the specified column from each slice
     * @param column the row to get from each slice
     * @param slice the slice to get
     * @return a slices() x column matrix of each slices row $row
     */
    public FloatMatrix getColumn(int column,int slice) {
        return getSlice(slice).getColumn(column);
    }


    /**
     * Transposes the tensor in a number of ways relative to the passed in vals:
     * Very similar to matlabs permute over 3d matrices
     * @param nums the nums (only 1,2,3)  to transform with
     * @return a tensor that is permuted with the elements of this tensor
     */
    public FloatTensor permute(int[] nums) {
        if (Arrays.equals(nums, new int[]{1, 2, 3})) {
            return dup();
        } else if (Arrays.equals(nums, new int[]{2, 1, 3}))
            return transpose();
            //number of rows becomes the number of slices
            //number of slices becomes number of rows, columns stay the same
        else if (Arrays.equals(nums, new int[]{3, 2, 1})) {
            FloatTensor ret = new FloatTensor(slices(), columns(), rows());
            for(int i = 0; i < ret.slices(); i++) {
                ret.setSlice(i,getRows(i));
            }
            return ret;
        }

        else if(Arrays.equals(nums,new int[]{2,3,1})) {
            FloatTensor t = new FloatTensor(slices(),columns(),rows());
            int currI = 0,currJ = 0;

            for(int i = 0; i < t.slices(); i++) {
                FloatMatrix slice = new FloatMatrix(t.rows(),t.columns());
                for(int row = 0; row < slice.rows; row++) {
                    for(int l = 0; l < slices(); l++) {
                        float val = get(currI,currJ,l);
                        slice.put(row,l,val);

                    }
                    currJ++;


                }
                t.setSlice(i,slice);
                if(currJ == columns()) {
                    currJ = 0;
                    currI++;
                }
            }
            return t;
        }

        else if(Arrays.equals(nums,new int[]{1,3,2})) {
            FloatTensor ret = new FloatTensor(rows(),slices(),columns());
            int column = 0;
            for(int i = 0; i < slices(); i++) {
                FloatMatrix slice = new FloatMatrix(ret.rows(),ret.columns());
                for(int j = 0; j < slice.columns; j++) {
                    FloatMatrix c = getColumn(column,j);
                    slice.putColumn(j,c);
                }
                ret.setSlice(i,slice);
            }
            return ret;
        }

        else if(Arrays.equals(nums,new int[]{3,1,2})) {
            FloatTensor t = new FloatTensor(slices(),rows(),columns());
            int column = 0;

            for(int i = 0; i < t.slices(); i++) {
                FloatMatrix slice = new FloatMatrix(t.rows(),t.columns());
                for(int row = 0; row < slice.rows; row++) {
                    for(int l = 0; l < slices(); l++) {
                        FloatMatrix val = getColumn(column,l);
                        val = val.reshape(1,val.length);
                        slice.putRow(row,val);

                    }


                }

                t.setSlice(i,slice);
                column++;

                if(column == t.columns())
                    column = 0;

            }
            return t;
        }


        throw new IllegalArgumentException("Illegal argument: Passed in array must be a unique array containing only" +
                "the numbers 1,2 or 3");
    }




    /**
     * This tensor with all the matrices transposed
     * @return a copy of this tensor with the matrices transposed
     */
    public FloatTensor transpose() {
        return new FloatTensor(super.transpose());
    }

    /**
     * A tensor populated by the row sums of each matrix in this tensor
     * @return a new tensor populated by the row sums of each matrix
     * in this tensor
     */
    public FloatTensor rowSums() {
        FloatTensor t = new FloatTensor(slices(),columns,slices);
        for(int i = 0 ; i < slices; i++) {
            t.setSlice(i,getSlice(i).rowSums());
        }

        return t;
    }

    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    public FloatMatrix bilinearProducts(FloatMatrix in) {
        if (in.columns != 1) {
            throw new AssertionError("Expected a column vector");
        }
        if (in.rows != columns()) {
            throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
        }
        if (rows != columns) {
            throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
        }

        FloatMatrix inT = in.transpose();
        FloatMatrix out = new FloatMatrix(slices, 1);
        for (int slice = 0; slice < slices; ++slice) {
            float result = inT.mul(getSlice(slice)).mul(in).get(0);
            out.put(slice, result);
        }

        return out;
    }

    /**
     * Slices up an individual matrix into a tensor
     * @param matrix the matrix to slice
     * @param numSlices the number of slices for the tensor
     * @return the new tensor with the specified number of slices
     * based on the origin matrix
     */
    public static FloatTensor create(FloatMatrix matrix,int numSlices) {
        FloatTensor ret = new FloatTensor(matrix);
        ret.slices = numSlices;
        return ret;
    }

    /**
     * Returns a zero tensor
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static FloatTensor zeros(int rows, int cols,int slices) {
        return new FloatTensor(rows,cols,slices);

    }

    /**
     * Returns a 1 tensor
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static FloatTensor ones(int rows, int cols,int slices) {
        FloatTensor ret =  new FloatTensor(rows,cols,slices);
        ret.assign(1);
        return ret;
    }



    /**
     * Returns a random tensor sampling from the normal distribution
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static FloatTensor rand(int rows, int cols,int slices,float min,float max) {
        FloatTensor t = new FloatTensor(rows ,cols,slices);
        for(int i = 0; i < slices; i++) {
            FloatMatrix d = new FloatMatrix(rows,cols);
            for(int j = 0; j < d.length; j++) {
                float val =  MathUtils.randomFloatBetween(min,max);
                if(val == 0)
                    val += Math.random();
                d.put(j,val);

            }
            t.setSlice(i,d);
        }


        return t;

    }


    /**
     * Returns a random tensor sampling from the normal distribution
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static FloatTensor rand(int rows, int cols,int slices,float min,float max,RandomGenerator rng) {
        FloatTensor t = new FloatTensor(rows ,cols,slices);
        for(int i = 0; i < slices; i++) {
            FloatMatrix d = new FloatMatrix(rows,cols);
            for(int j = 0; j < d.length; j++) {
                float val =  MathUtils.randomFloatBetween(min,max);
                if(val == 0)
                    val += Math.random();
                d.put(j,val);

            }
            t.setSlice(i,d);
        }


        return t;

    }


    /**
     * Returns a random tensor sampling from the normal distribution
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static FloatTensor rand(int rows, int cols,int slices) {
        return rand(rows,cols,slices, Distributions.uniform(new MersenneTwister(123), 0, 1));

    }
    /**
     * Returns a random tensor sampling from the given distribution
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @param dist the distribution to sample from
     * @return the tensor with the specified slices and the random matrices
     */
    public static FloatTensor rand(int rows, int cols,int slices,RealDistribution dist) {
        FloatTensor t = new FloatTensor(rows,cols,slices);
        for(int i = 0; i < slices; i++) {
            FloatMatrix d = new FloatMatrix(rows,cols);
            for(int j = 0; j < d.length; j++)
                d.put(j,(float) dist.sample());
            t.setSlice(i,d);
        }

        return t;

    }




    /**
     * Element wise subtraction of each slice of the passed in tensor
     * @param tensor the tensor to subtract by
     * @return the element wise subtraction of the passed in tensor
     * with this tensor
     */
    public FloatTensor sub(FloatTensor tensor)  {
        FloatTensor ret = new FloatTensor(super.sub(tensor));
        ret.slices = slices;
        ret.perMatrixRows = perMatrixRows;
        return ret;
    }

    /**
     * Element wise addition of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public FloatTensor add(FloatTensor tensor) {
        FloatTensor ret = new FloatTensor(super.add(tensor));
        ret.slices = slices;
        ret.perMatrixRows = perMatrixRows;
        return ret;
    }

    /**
     * Element wise multiplication of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public FloatTensor mul(FloatTensor tensor) {
        FloatTensor ret = new FloatTensor(mul(tensor));
        ret.slices = slices;
        ret.perMatrixRows = perMatrixRows;
        return ret;
    }


    /**
     * This tensor subtracted by val
     * @param val the tensor to subtract from
     * @return this tensor with values - val
     */
    public FloatTensor sub(float val) {
        FloatTensor ret = new FloatTensor(super.sub(val));
        ret.perMatrixRows = perMatrixRows;
        ret.slices = slices;
        return ret;
    }

    /**
     *Adds a value to each element in the tensor
     * @param val the value to add
     * @return a tensor with the elements of this tensor added by val
     */
    public FloatTensor add(float val) {
        FloatTensor ret = new FloatTensor(super.add(val));
        ret.perMatrixRows = perMatrixRows;
        ret.slices = slices;
        return ret;
    }

    /**
     * This tensor's elements multiplied by val
     * @param val the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public FloatTensor div(float val) {
        FloatTensor ret = new FloatTensor(super.div(val));
        ret.perMatrixRows = perMatrixRows;
        ret.slices = slices;
        return ret;
    }


    /**
     * This tensor's elements multiplied by val
     * @param val the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public FloatTensor mul(float val) {
        FloatTensor ret = new FloatTensor(super.mul(val));
        ret.slices = slices;
        ret.perMatrixRows = perMatrixRows;
        return ret;
    }

    /**
     * This tensor's elements divided by val
     * @param val the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public FloatTensor div(FloatTensor val) {
        FloatTensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i, get(i) / val.get(i));
        return copy;
    }


    public FloatTensor scale(float value) {
        FloatTensor ret = new FloatTensor(SimpleBlas.scal(value,this));
        ret.slices = slices;
        ret.perMatrixRows = perMatrixRows;
        return ret;
    }


    /**
     * Assigns all values of this tensor to the passed in value
     * @param val the value to assign
     */
    public void assign(float val) {
        for(int i = 0;i < length; i++)
            put(i,val);
    }


    /**
     * A copy of this tensor with tanh applied
     * @return a copy of this tensor with tanh applied
     */
    public FloatTensor tanh() {
        FloatTensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i, (float) FastMath.tanh(get(i)));

        return copy;
    }
    /**
     * A copy of this tensor with sigmoid applied
     * @return a copy of this tensor with sigmoid applied
     */
    public FloatTensor sigmoid() {
        FloatTensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i, (float) MathUtils.sigmoid(get(i)));
        return copy;
    }

    /**
     * A copy of this tensor with exp applied
     * @return a copy of this tensor with exp applied
     */
    public FloatTensor exp() {
        FloatTensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i,(float) FastMath.exp(get(i)));
        return copy;
    }

    /**
     * The number of columns of each slice
     * @return the number of columns of each slice
     */
    public int columns() {
        return columns;
    }

    /**
     * Returns the number of rows in
     * each slice
     * @return the number of rows
     */
    public int rows() {
        return perMatrixRows;
    }

    /**
     * The number of slices in this tensor
     * @return the number of slices in this tensor
     */
    public int slices() {
        return this.slices;
    }

    public void setSlices(int slices) {
        this.slices = slices;
    }

    public void setPerMatrixRows(int perMatrixRows) {
        this.perMatrixRows = perMatrixRows;
    }



    /**
     * Add a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor addi(FloatMatrix other) {
        return createBasedOn(super.addi(other),this);
    }

    /**
     * Add a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor add(FloatMatrix other) {
        return createBasedOn(super.add(other),this);
    }

    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatTensor addi(float v) {
        return createBasedOn(super.addi(v),this);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor subi(FloatMatrix other) {
        return createBasedOn(super.subi(other),this);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor sub(FloatMatrix other) {
        return createBasedOn(super.sub(other),this);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatTensor subi(float v) {
        return createBasedOn(super.subi(v),this);
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor muli(FloatMatrix other) {
        return createBasedOn(super.muli(other),this);
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor mul(FloatMatrix other) {
        return createBasedOn(super.mul(other),this);
    }

    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatTensor muli(float v) {
        return createBasedOn(super.muli(v),this);
    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor divi(FloatMatrix other) {
        return createBasedOn(super.divi(other),this);
    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public FloatTensor div(FloatMatrix other) {
        return createBasedOn(super.div(other),this);
    }

    /**
     * Elementwise divide by a scalar (in place).
     *
     * @param v
     */
    @Override
    public FloatTensor divi(float v) {
        return createBasedOn(super.divi(v),this);
    }

}
