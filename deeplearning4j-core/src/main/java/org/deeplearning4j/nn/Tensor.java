package org.deeplearning4j.nn;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.Geometry;
import org.jblas.MatrixFunctions;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Tensor represents a set of matrices of all the same dimensions.
 * Based on the Recursive Neural Tensor Network by Socher et. al
 * @author Adam Gibson
 */
public class Tensor implements Serializable {

    private DoubleMatrix[] slices;
    private int rows,cols;


    public Tensor(int rows, int columns, int slices) {
        this.rows = rows;
        this.cols = columns;
        this.slices = new DoubleMatrix[slices];

        for(int i = 0; i < slices; i++) {
            this.slices[i] = DoubleMatrix.zeros(rows,columns);
        }
    }


    public Tensor(Tensor t) {
        this.slices = new DoubleMatrix[t.slices()];
        this.rows = t.rows();
        this.cols = t.columns();
        System.arraycopy(t.slices,0,this.slices,0,t.slices());
    }

    public Tensor(DoubleMatrix[] slices) {
        ensureSameSize(slices);
        this.slices = new DoubleMatrix[slices.length];
        for(int i = 0; i < slices.length; i++) {
            this.slices[i] = slices[i].dup();
        }

    }

    private void ensureSameSize(DoubleMatrix[] slices) {
        if(slices == null || slices.length < 1)
            throw new IllegalArgumentException("Illegal argument, please pass in slices array >= length 1");
        this.rows = slices[0].rows;
        this.cols = slices[0].columns;
        for(int i = 1; i < slices.length; i++) {
            if(slices[i].rows != rows || slices[i].columns != cols)
                throw new IllegalArgumentException("All slices must be of the same number of rows and columns");
        }
    }


    public DoubleMatrix getSlice(int index) {
        return slices[index];
    }

    public void setSlice(int index,DoubleMatrix slice) {
        if(slice.rows != rows || slice.columns != cols)
            throw new IllegalArgumentException("Illegal matrix passed in, must be of same dimenions as specified slices");

        slices[index] = slice;
    }

    public double sum() {
        double sum = 0.0;
        for(int i = 0;i < slices(); i++)
            sum += slices[i].sum();
        return sum;
    }

    /**
     * Clones this tensor
     * @return a copy of this tensor
     */
    public Tensor dup() {
        return new Tensor(this);
    }


    public Tensor get(int[] rowIndices,int[] columnIndices) {
        DoubleMatrix first = slices[0].get(rowIndices,columnIndices);
        Tensor ret = new Tensor(first.rows,first.columns,slices());
        ret.slices[0]  = first;
        for(int i = 1; i < slices(); i++) {
            ret.slices[1] = slices[i].get(rowIndices,columnIndices);
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
        DoubleMatrix ret = new DoubleMatrix(rows() * slices(),columns());
        int currSlice = 0;
        int row = 0;
        for(int i = 0; i < ret.rows; i++) {
            DoubleMatrix slice = getSlice(currSlice);
            DoubleMatrix rowM = slice.getRow(row);
            ret.putRow(i,rowM);
            row++;
            if(row >= rows()) {
                row = 0;
                currSlice++;
            }
        }
        return ret;
    }

    /**
     * The column sums of each matrix in this tensor
     * @return a tensor populated by the column sums of each matrix in the tensor
     */
    public Tensor columnsSums() {
        DoubleMatrix first = slices[0].columnSums();
        Tensor t = new Tensor(first.rows,first.columns,slices.length);
        t.slices[0] = first;
        for(int i =1 ; i < slices.length; i++) {
            t.slices[i] = slices[i].columnSums();
        }
        return t;
    }


    @Override
    public String toString() {
        return "Tensor{" +
                "slices=" + Arrays.toString(slices) +
                ", rows=" + rows +
                ", cols=" + cols +
                ", slices=" + slices.length +
                '}';
    }


    public void set(int slice,int i,int j,double val) {
        slices[slice].put(i,j,val);
    }

    public double get(int i,int j,int slice) {
        return slices[slice].get(i,j);
    }

    /**
     * Gets the specified row from each slice
     * @param row the row to get from each slice
     * @return a slices() x column matrix of each slices row $row
     */
    public DoubleMatrix getRows(int row) {
        DoubleMatrix rows = new DoubleMatrix(slices(),columns());
        for(int i = 0; i < slices(); i++) {
            rows.putRow(i,slices[i].getRow(row));
        }
        return rows;
    }


    /**
     * Gets the specified row from each slice
     * @param column the row to get from each slice
     * @return a slices() x column matrix of each slices row $row
     */
    public DoubleMatrix getColumns(int column) {
        DoubleMatrix rows = new DoubleMatrix(slices(),columns());
        for(int i = 0; i < slices(); i++) {
            rows.putRow(i,slices[i].getColumn(column));
        }
        return rows;
    }



    /**
     * Gets the specified row from each slice
     * @param row the row to get from each slice
     * @param slice the slice to get
     * @return a slices() x column matrix of each slices row $row
     */
    public DoubleMatrix getRow(int row,int slice) {
        DoubleMatrix rows = slices[slice].getRow(row);
        return rows;
    }

    /**
     * Gets the specified row from each slice
     * @param column the row to get from each slice
     * @param slice the slice to get
     * @return a slices() x column matrix of each slices row $row
     */
    public DoubleMatrix getColumn(int column,int slice) {
        DoubleMatrix c = slices[slice].getColumn(column);
        return c;
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
                ret.slices[i] = getRows(i);
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

                t.slices[i] = slice;
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

                ret.slices[i] = slice;
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

                t.slices[i] = slice;
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
        Tensor ret = new Tensor(columns(),rows,slices());
        for(int i = 0;i  < slices(); i++) {
            ret.slices[i] = slices[i].transpose();
        }
        return ret;
    }

    /**
     * A tensor populated by the row sums of each matrix in this tensor
     * @return a new tensor populated by the row sums of each matrix
     * in this tensor
     */
    public Tensor rowSums() {
        DoubleMatrix first = slices[0].rowSums();
        Tensor t = new Tensor(first.rows,first.columns,slices.length);
        t.slices[0] = first;
        for(int i =1 ; i < slices.length; i++) {
            t.slices[i] = slices[i].rowSums();
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
        if (rows != cols) {
            throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
        }

        DoubleMatrix inT = in.transpose();
        DoubleMatrix out = new DoubleMatrix(slices.length, 1);
        for (int slice = 0; slice < slices.length; ++slice) {
            double result = inT.mul(slices[slice]).mul(in).get(0);
            out.put(slice, result);
        }

        return out;
    }


    /**
     * Returns a zero tensor
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static Tensor zeros(int rows, int cols,int slices) {
        Tensor t = new Tensor(rows,cols,slices);
        for(int i = 0; i < slices; i++)
            t.slices[i] = DoubleMatrix.zeros(rows,cols);;


        return t;

    }
    /**
     * Returns a random tensor sampling from the normal distribution
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static Tensor rand(int rows, int cols,int slices,double min,double max) {
        Tensor t = new Tensor(rows,cols,slices);
        for(int i = 0; i < slices; i++) {
            DoubleMatrix d = new DoubleMatrix(rows,cols);
            for(int j = 0; j < d.length; j++) {
                double val =  MathUtils.randomDoubleBetween(min,max);
                if(val == 0)
                    val += Math.random();
                d.put(j,val);

            }
            t.slices[i] = d;
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
            t.slices[i] = d;
        }

        return t;

    }

    /**
     * Element wise subtraction of each slice of the passed in tensor
     * @param tensor the tensor to subtract by
     * @return the element wise subtraction of the passed in tensor
     * with this tensor
     */
    public Tensor sub(Tensor tensor) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].sub(tensor.slices[i]);
        return t;
    }

    /**
     * Element wise addition of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor add(Tensor tensor) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].add(tensor.slices[i]);
        return t;
    }

    /**
     * Element wise multiplication of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor mul(Tensor tensor) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].mul(tensor.slices[i]);
        return t;
    }


    /**
     * This tensor subtracted by val
     * @param val the tensor to subtract from
     * @return this tensor with values - val
     */
    public Tensor sub(double val) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].sub(val);
        return t;
    }

    /**
     *Adds a value to each element in the tensor
     * @param val the value to add
     * @return a tensor with the elements of this tensor added by val
     */
    public Tensor add(double val) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].add(val);
        return t;
    }

    /**
     * This tensor's elements multiplied by val
     * @param val the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor mul(double val) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].mul(val);
        return t;
    }


    public Tensor scale(double value) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].mul(value);
        return t;
    }


    /**
     * Assigns all values of this tensor to the passed in value
     * @param val the value to assign
     */
    public void assign(double val) {
        for(int i = 0; i < slices.length; i++)
            slices[i] = DoubleMatrix.zeros(rows,columns()).add(val);
    }


    /**
     * A copy of this tensor with tanh applied
     * @return a copy of this tensor with tanh applied
     */
    public Tensor tanh() {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = MatrixFunctions.tanh(slices[i]);
        return t;
    }
    /**
     * A copy of this tensor with sigmoid applied
     * @return a copy of this tensor with sigmoid applied
     */
    public Tensor sigmoid() {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = MatrixUtil.sigmoid(slices[i]);
        return t;
    }

    /**
     * A copy of this tensor with exp applied
     * @return a copy of this tensor with exp applied
     */
    public Tensor exp() {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = MatrixFunctions.exp(slices[i]);
        return t;
    }

    /**
     * Rows * cols * numSlices - the total number of elements in this tensor
     * @return rows * cols * slices
     */
    public int numElements() {
        return rows * cols * slices.length;
    }


    public int columns() {
        return cols;
    }

    public int rows() {
        return rows;
    }

    public int slices() {
        return slices.length;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor)) return false;

        Tensor tensor = (Tensor) o;

        if (cols != tensor.cols) return false;
        if (rows != tensor.rows) return false;
        if (!Arrays.equals(slices, tensor.slices)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(slices);
        result = 31 * result + rows;
        result = 31 * result + cols;
        return result;
    }
}
