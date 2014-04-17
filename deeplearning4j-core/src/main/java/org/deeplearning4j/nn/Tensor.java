package org.deeplearning4j.nn;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.Geometry;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Tensor represents a set of matrices of all the same dimensions.
 * Based on the Recursive Neural Tensor Network by Socher et. al
 * @author Adam Gibson
 */
public class Tensor extends DoubleMatrix implements Serializable {

    private int slices,rows;


    public Tensor(int rows, int columns, int slices) {
        super(rows * slices,columns);
        this.slices = slices;
        this.rows = rows;
    }

    public Tensor(DoubleMatrix t) {
        super(t.toArray2());
    }
    public Tensor(Tensor t) {
        super(t.toArray2());
    }


    private int[] getColIndicesForSlice() {
        int[] ret = new int[columns];
        for(int i = 0; i < ret.length; i++)
            ret[i] = i;
        return ret;
    }
    /* Gets a block of the matrix such that the slice represents a subset of the rows in the matrix */
    private int[] getRowIndicesForSlice(int slice) {
        int[] ret = new int[Math.abs(slice * rows - (slice * rows) + rows)];
        int start = slice * rows;
        for(int i = 0; i < ret.length; i++) {
            ret[i] = start;
            start++;
        }

        return ret;
    }


    public DoubleMatrix getSlice(int index) {
        return get(getRowIndicesForSlice(index),getColIndicesForSlice());
    }

    public void setSlice(int index,DoubleMatrix slice) {
        put(getRowIndicesForSlice(index),getColIndicesForSlice(),slice);
    }

    /**
     * Clones this tensor
     * @return a copy of this tensor
     */
    @Override
    public Tensor dup() {
        return new Tensor(this);
    }


    /**
     * Assigns the corresponding slice to the passed in elements for each slice[i] in tensor
     * @param tensor the tensor to set
     * @param rowIndices the row indices for each slice
     * @param columnIndices the column indices for each slice
     */
    public void set(Tensor tensor,int[] rowIndices,int[] columnIndices) {
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
    public void set(DoubleMatrix toSet,int[] rowIndices,int[] columnIndices) {
        for(int i = 0; i < slices(); i++) {
            setSlice(i, toSet);
        }

    }

    public Tensor columnSums() {
       return new Tensor(columnsSums());
    }

    public Tensor getIndicesSlices(int[] rowIndices,int[] columnIndices) {
        DoubleMatrix first = getSlice(0).get(rowIndices, columnIndices);
        Tensor ret = new Tensor(first.rows,first.columns,slices());
        ret.setSlice(0,first);
        for(int i = 1; i < slices(); i++) {
            ret.setSlice(i, getSlice(i).get(rowIndices, columnIndices));;
        }

        return ret;
    }

    /**
     * Returns this tensor as a matrix such that
     * all values in all slices are concacneated in to one matrix.
     * This matrix will be a t.rows() * t.slices() x t.columns() matrix
     * @return
     */
    public DoubleMatrix toMatrix() {
       return dup();
    }

    /**
     * The column sums of each matrix in this tensor
     * @return a tensor populated by the column sums of each matrix in the tensor
     */
    public Tensor columnsSums() {
        //each slice will have a column sum
        Tensor t = new Tensor(1,columns,slices);

        for(int i = 0; i < slices(); i++) {
            DoubleMatrix sums = getSlice(i).columnSums();
            t.putRow(i,sums);

        }
        return t;
    }



    public void set(int slice,int i,int j,double val) {
        put(rows * slice + i,j,val);
    }

    public double get(int i,int j,int slice) {
        return get(i + slice,j);
    }

    /**
     * Gets the specified row from each slice
     * @param row the row to get from each slice
     * @return a slices() x column matrix of each slices row $row
     */
    public DoubleMatrix getRows(int row) {
        int[] indices = new int[slices()];
        for(int i = 0; i < slices(); i++)
            indices[i] = row * i;
        DoubleMatrix ret = get(indices,getColIndicesForSlice());
        return ret;
    }


    /**
     * Gets the specified row from each slice
     * @param column the row to get from each slice
     * @return a slices() x column matrix of each slices row $row
     */
    public DoubleMatrix getColumns(int column) {
        DoubleMatrix rows = new DoubleMatrix(slices(),columns());
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
    public DoubleMatrix getRow(int row,int slice) {
        DoubleMatrix rows = get(this.getRowIndicesForSlice(slice),this.getColIndicesForSlice());
        return rows;
    }

    /**
     * Gets the specified column from each slice
     * @param column the row to get from each slice
     * @param slice the slice to get
     * @return a slices() x column matrix of each slices row $row
     */
    public DoubleMatrix getColumn(int column,int slice) {
             return getSlice(slice).getColumn(column);
    }


    /**
     * Transposes the tensor in a number of ways relative to the passed in vals:
     * Very similar to matlabs permute over 3d matrices
     * @param nums the nums (only 1,2,3)  to transform with
     * @return a tensor that is permuted with the elements of this tensor
     */
    public Tensor permute(int[] nums) {
        if (Arrays.equals(nums, new int[]{1, 2, 3})) {
            return dup();
        } else if (Arrays.equals(nums, new int[]{2, 1, 3}))
            return transpose();
            //number of rows becomes the number of slices
            //number of slices becomes number of rows, columns stay the same
        else if (Arrays.equals(nums, new int[]{3, 2, 1})) {
            Tensor ret = new Tensor(slices(), columns(), rows());
            for(int i = 0; i < ret.slices(); i++) {
                ret.setSlice(i,getRows(i));
            }
            return ret;
        }

        else if(Arrays.equals(nums,new int[]{2,3,1})) {
            Tensor t = new Tensor(slices(),columns(),rows());
            int currI = 0,currJ = 0;

            for(int i = 0; i < t.slices(); i++) {
                DoubleMatrix slice = new DoubleMatrix(t.rows(),t.columns());
                for(int row = 0; row < slice.rows; row++) {
                    for(int l = 0; l < slices(); l++) {
                        double val = get(currI,currJ,l);
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
            Tensor ret = new Tensor(rows(),slices(),columns());
            int column = 0;
            for(int i = 0; i < slices(); i++) {
                DoubleMatrix slice = new DoubleMatrix(ret.rows(),ret.columns());
                for(int j = 0; j < slice.columns; j++) {
                    DoubleMatrix c = getColumn(column,j);
                    slice.putColumn(j,c);
                }
                ret.setSlice(i,slice);
            }
            return ret;
        }

        else if(Arrays.equals(nums,new int[]{3,1,2})) {
            Tensor t = new Tensor(slices(),rows(),columns());
            int column = 0;

            for(int i = 0; i < t.slices(); i++) {
                DoubleMatrix slice = new DoubleMatrix(t.rows(),t.columns());
                for(int row = 0; row < slice.rows; row++) {
                    for(int l = 0; l < slices(); l++) {
                        DoubleMatrix val = getColumn(column,l);
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
    public Tensor transpose() {
       return new Tensor(super.transpose());
    }

    /**
     * A tensor populated by the row sums of each matrix in this tensor
     * @return a new tensor populated by the row sums of each matrix
     * in this tensor
     */
    public Tensor rowSums() {
        Tensor t = new Tensor(slices(),columns,slices);
        for(int i = 0 ; i < slices; i++) {
            t.setSlice(i,getSlice(i).rowSums());
        }

        return t;
    }

    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    public DoubleMatrix bilinearProducts(DoubleMatrix in) {
        if (in.columns != 1) {
            throw new AssertionError("Expected a column vector");
        }
        if (in.rows != columns()) {
            throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
        }
        if (rows != columns) {
            throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
        }

        DoubleMatrix inT = in.transpose();
        DoubleMatrix out = new DoubleMatrix(slices, 1);
        for (int slice = 0; slice < slices; ++slice) {
            double result = inT.mul(getSlice(slice)).mul(in).get(0);
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
    public static Tensor create(DoubleMatrix matrix,int numSlices) {
           return new Tensor(matrix);

    }

    /**
     * Returns a zero tensor
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static Tensor zeros(int rows, int cols,int slices) {
         return new Tensor(DoubleMatrix.zeros(rows * slices,cols));

    }




    /**
     * Returns a random tensor sampling from the normal distribution
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static Tensor rand(int rows, int cols,int slices,double min,double max) {
        Tensor t = new Tensor(rows ,cols,slices);
        for(int i = 0; i < slices; i++) {
            DoubleMatrix d = new DoubleMatrix(rows,cols);
            for(int j = 0; j < d.length; j++) {
                double val =  MathUtils.randomDoubleBetween(min,max);
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
    public static Tensor rand(int rows, int cols,int slices) {
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
    public static Tensor rand(int rows, int cols,int slices,RealDistribution dist) {
        Tensor t = new Tensor(rows,cols,slices);
        for(int i = 0; i < slices; i++) {
            DoubleMatrix d = new DoubleMatrix(rows,cols);
            for(int j = 0; j < d.length; j++)
                d.put(j,dist.sample());
            t.setSlice(i,d);
        }

        return t;

    }

    public Tensor repmat(int rows,int cols) {
        return new Tensor(repmat(rows,cols));
    }


    /**
     * Element wise subtraction of each slice of the passed in tensor
     * @param tensor the tensor to subtract by
     * @return the element wise subtraction of the passed in tensor
     * with this tensor
     */
    public Tensor sub(Tensor tensor) {
        return new Tensor(sub(tensor));
    }

    /**
     * Element wise addition of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor add(Tensor tensor) {
        return new Tensor(add(tensor));

    }

    /**
     * Element wise multiplication of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor mul(Tensor tensor) {
        return new Tensor(mul(tensor));

    }


    /**
     * This tensor subtracted by val
     * @param val the tensor to subtract from
     * @return this tensor with values - val
     */
    public Tensor sub(double val) {
        return new Tensor(sub(val));
    }

    /**
     *Adds a value to each element in the tensor
     * @param val the value to add
     * @return a tensor with the elements of this tensor added by val
     */
    public Tensor add(double val) {
         return new Tensor(add(val));
    }

    /**
     * This tensor's elements multiplied by val
     * @param val the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor mul(double val) {
        return new Tensor(mul(val));
    }


    public Tensor scale(double value) {
       return new Tensor(SimpleBlas.scal(value,this));
    }


    /**
     * Assigns all values of this tensor to the passed in value
     * @param val the value to assign
     */
    public void assign(double val) {
        put(new int[]{0,rows},new int[]{0,columns},val);
    }


    /**
     * A copy of this tensor with tanh applied
     * @return a copy of this tensor with tanh applied
     */
    public Tensor tanh() {
        return new Tensor(MatrixFunctions.tanh(this));
    }
    /**
     * A copy of this tensor with sigmoid applied
     * @return a copy of this tensor with sigmoid applied
     */
    public Tensor sigmoid() {
        return new Tensor(MatrixUtil.sigmoid(this));
    }

    /**
     * A copy of this tensor with exp applied
     * @return a copy of this tensor with exp applied
     */
    public Tensor exp() {
        return new Tensor(MatrixFunctions.exp(this));
    }

    /**
     * Rows * cols * numSlices - the total number of elements in this tensor
     * @return rows * cols * slices
     */
    public int numElements() {
        return length;
    }


    public int columns() {
        return columns;
    }

    public int rows() {
        return rows;
    }

    public int slices() {
        return this.slices;
    }

}
