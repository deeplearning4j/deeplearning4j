package org.deeplearning4j.nn;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.Geometry;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.jblas.ranges.Range;
import org.jblas.ranges.RangeUtils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;

/**
 * Tensor represents a set of matrices of all the same dimensions.
 * Based on the Recursive Neural Tensor Network by Socher et. al
 * @author Adam Gibson
 */
public class Tensor extends DoubleMatrix implements Serializable {

    private int slices,perMatrixRows;

    /**
     * Creates this tensor with the specified number of rows, columns and slices
     * Note that this will throw an illegal argument exception if any of the given
     * params are less than 1
     * @param rows
     * @param columns
     * @param slices
     */
    public Tensor(int rows, int columns, int slices) {
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

    public Tensor(DoubleMatrix t) {
        super(t.toArray2());
        this.slices = 1;
    }

    /**
     * Initializes this tensor's data
     * and the number of slices and per matrix rows
     * @param t the tensor to initialize with
     */
    public Tensor(Tensor t) {
        super(t.toArray2());
        this.slices = t.slices();
        this.perMatrixRows = t.perMatrixRows;
    }

    public DoubleMatrix shape() {
        DoubleMatrix ret = new DoubleMatrix(1,3);
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

    public Tensor sliceColumnSums() {
        Tensor ret = new Tensor(1,columns(),slices);
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
    public DoubleMatrix sliceElementSums() {
        DoubleMatrix ret = new DoubleMatrix(rows(),columns());
        for(int i = 0; i < ret.rows; i++) {
            for(int j = 0; j < ret.columns; j++) {
                double sum = 0;
               for(int slice = 0; slice < slices(); slice++) {
                   sum += getSlice(slice).get(i,j);
               }
                ret.put(i,j,sum);

            }


        }

        return ret;
    }


    public Tensor sliceRowSums() {
        Tensor ret = new Tensor(1,rows(),slices);
        for(int i = 0; i < slices(); i++) {
            DoubleMatrix rowSums = getSlice(i).rowSums().transpose();
            ret.setSlice(i,rowSums);
        }
        return ret;
    }


    /**
     * Returns the given slice
     * @param index the index of the slice
     * @return the given slice
     */
    public DoubleMatrix getSlice(int index) {
      if(index >= slices())
          throw new IllegalArgumentException("Unable to get slice " + index + " out of bounds");

       try {
           DoubleMatrix slice =  get(RangeUtils.interval(index,index + rows()),RangeUtils.interval(0,columns()));
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
    public void setSlice(int index,DoubleMatrix slice) {
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



    /**
     * Returns this tensor as a matrix such that
     * all values in all slices are concacneated in to one matrix.
     * This matrix will be a t.rows() * t.slices() x t.columns() matrix
     * @return
     */
    public DoubleMatrix toMatrix() {
        return this;
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
        DoubleMatrix ret = get(indices);
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
        DoubleMatrix rows = get(getRowIndicesForSlice(slice),getColIndicesForSlice());
        return rows;
    }


    public Tensor ff() {
        Tensor ret = new Tensor(this);
        for(int i = 0; i < slices(); i++) {
            DoubleMatrix slice = MatrixUtil.reverse(getSlice(i));
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
        Tensor ret = new Tensor(matrix);
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
    public static Tensor zeros(int rows, int cols,int slices) {
        return new Tensor(rows,cols,slices);

    }

    /**
     * Returns a 1 tensor
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @return the tensor with the specified slices and the random matrices
     */
    public static Tensor ones(int rows, int cols,int slices) {
        Tensor ret =  new Tensor(rows,cols,slices);
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
    public static Tensor rand(int rows, int cols,int slices,double min,double max,RandomGenerator rng) {
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




    /**
     * Element wise subtraction of each slice of the passed in tensor
     * @param tensor the tensor to subtract by
     * @return the element wise subtraction of the passed in tensor
     * with this tensor
     */
    public Tensor sub(Tensor tensor)  {
        Tensor ret = new Tensor(super.sub(tensor));
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
    public Tensor add(Tensor tensor) {
        Tensor ret = new Tensor(super.add(tensor));
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
    public Tensor mul(Tensor tensor) {
        Tensor ret = new Tensor(mul(tensor));
        ret.slices = slices;
        ret.perMatrixRows = perMatrixRows;
        return ret;
    }


    /**
     * This tensor subtracted by val
     * @param val the tensor to subtract from
     * @return this tensor with values - val
     */
    public Tensor sub(double val) {
        Tensor ret = new Tensor(super.sub(val));
        ret.perMatrixRows = perMatrixRows;
        ret.slices = slices;
        return ret;
    }

    /**
     *Adds a value to each element in the tensor
     * @param val the value to add
     * @return a tensor with the elements of this tensor added by val
     */
    public Tensor add(double val) {
        Tensor ret = new Tensor(super.add(val));
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
    public Tensor div(double val) {
        Tensor ret = new Tensor(super.div(val));
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
    public Tensor mul(double val) {
        Tensor ret = new Tensor(super.mul(val));
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
    public Tensor div(Tensor val) {
        Tensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i, get(i) / val.get(i));
        return copy;
    }


    public Tensor scale(double value) {
        Tensor ret = new Tensor(SimpleBlas.scal(value,this));
        ret.slices = slices;
        ret.perMatrixRows = perMatrixRows;
        return ret;
    }


    /**
     * Assigns all values of this tensor to the passed in value
     * @param val the value to assign
     */
    public void assign(double val) {
       for(int i = 0;i < length; i++)
           put(i,val);
    }


    /**
     * A copy of this tensor with tanh applied
     * @return a copy of this tensor with tanh applied
     */
    public Tensor tanh() {
        Tensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i, FastMath.tanh(get(i)));

        return copy;
    }
    /**
     * A copy of this tensor with sigmoid applied
     * @return a copy of this tensor with sigmoid applied
     */
    public Tensor sigmoid() {
        Tensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i, MathUtils.sigmoid(get(i)));
        return copy;
    }

    /**
     * A copy of this tensor with exp applied
     * @return a copy of this tensor with exp applied
     */
    public Tensor exp() {
        Tensor copy = dup();
        for(int i = 0; i < length; i++)
            copy.put(i,FastMath.exp(get(i)));
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

}
