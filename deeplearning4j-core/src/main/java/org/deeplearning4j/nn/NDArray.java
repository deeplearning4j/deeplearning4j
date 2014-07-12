package org.deeplearning4j.nn;

import org.deeplearning4j.util.ArrayUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * NDArray: (think numpy)
 * @author Adam Gibson
 */
public class NDArray extends DoubleMatrix {

    private int[] shape;
    private int[] stride;
    private int offset = 0;



    public NDArray(List<DoubleMatrix> slices,int[] shape) {
        List<double[]> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++)
            list.add(slices.get(i).data);

        this.data = ArrayUtil.combine(list);
        this.shape = shape;
        this.length = ArrayUtil.prod(shape);
    }



    public NDArray(double[] data,int[] shape,int[] stride,int offset) {
        if(data != null  && data.length > 0)
            this.data = data;
        this.shape = shape;
        this.offset = offset;
        this.stride = stride;
        this.length = ArrayUtil.prod(shape);
    }

    public NDArray(double[] data,int[] shape) {
        this(data,shape,0);
    }

    public NDArray(double[] data,int[] shape,int offset) {
        this(data,shape,ArrayUtil.calcStrides(shape),offset);
    }



    public NDArray(int[] shape,int[] stride,int offset) {
        this(new double[]{},shape,stride,offset);
    }


    public NDArray(int[] shape,int offset) {
        this(shape,ArrayUtil.calcStrides(shape),offset);
    }


    public NDArray(int[] shape) {
        this(shape,0);
    }





    @Override
    public NDArray dup() {
        double[] dupData = new double[data.length];
        System.arraycopy(data,0,dupData,0,dupData.length);
        NDArray ret = new NDArray(dupData,shape,stride,offset);
        return ret;
    }

    @Override
    public NDArray put(int row,int column,double value) {
        if (shape.length == 2)
            data[offset + row * stride[0]  + column * stride[1]] = value;

        else
              throw new UnsupportedOperationException("Invalid set for a non 2d array");
        return this;
    }


    @Override
    public DoubleMatrix put(int[] indexes, double value) {
        int ix=offset;
        if (indexes.length!= shape.length)
            throw new IllegalArgumentException("Unable to set values: number of indices must be equal to the shape");
        for (int i=0; i< shape.length; i++) {
            ix+=indexes[i] * stride[i];
        }

        data[ix] = value;
        return this;
    }


    @Override
    public DoubleMatrix put(int i, double v) {
        data[i + offset] = v;
        return this;
    }

    @Override
    public double get(int i) {
        return super.get(offset + i);
    }


    /**
     * Returns the specified slice of this matrix
     * @param slice the slice to return
     * @return the specified slice of this matrix
     */
    public NDArray slice(int slice) {

        if (shape.length == 0) {
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");
        } else if (shape.length == 1) {
            return new NDArray(data,new int[]{1},new int[]{1},offset + slice * stride[0]);
        } else if (shape.length == 2) {
            int st = stride[1];
            if (st == 1) {
                return new NDArray(data, new int[]{ shape[1] },new int[]{1},offset + slice * stride[0]);
            } else {

                return new NDArray(data,new int[]{ shape[1] } ,new int[]{ stride[1] },offset + slice * stride[0]);
            }
        } else {
            return new NDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1,shape.length),offset + slice * stride[0]);
        }
    }


    /**
     * Returns the slice of this from the specified dimension
     * @param dimension the dimension to return from
     * @param index the index of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and index
     */
    public NDArray slice(int dimension, int index) {
        if (dimension == 0)
            return slice(index);
        if (shape.length == 2) {
            if (dimension!=1)
               throw new IllegalArgumentException("Unable to retrieve dimension " + dimension + " from a 2d array");
            return new NDArray(data, new int[]{shape[0]}, new int[]{stride[0]}, offset + index * stride[1]);
        }
        return new NDArray(data,
                ArrayUtil.removeIndex(shape,index),
                ArrayUtil.removeIndex(stride,index),
                offset + index * stride[dimension]);
    }

    /**
     * Fetch a particular number on a multi dimensional scale.
     * @param indexes the indexes to get a number from
     * @return the number at the specified indices
     */
    public double getMulti(int... indexes) {
        int ix = 0;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return data[ix];
    }



    /**
     * Add a scalar to a matrix (in-place).
     * @param result
     */
    @Override
    public NDArray addi(DoubleMatrix result) {
         super.addi(result);
        return this;
    }

    /**
     * Subtract two matrices (in-place).
     *
     * @param result
     */
    @Override
    public NDArray subi(DoubleMatrix result) {
         super.subi(result);
        return this;
    }



    /**
     * Elementwise multiplication (in-place).
     *
     * @param result
     */
    @Override
    public NDArray muli(DoubleMatrix result) {
         super.muli(result);
        return this;
    }


    /**
     * Matrix-matrix multiplication (in-place).
     * @param result
     */
    @Override
    public NDArray mmuli( DoubleMatrix result) {
         super.mmuli(result);
        return this;
    }



    /**
     * Elementwise division (in-place).
     *
     * @param result
     */
    @Override
    public NDArray divi(DoubleMatrix result) {
         super.divi(result);
        return this;
    }

    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray addi(double v) {
         super.addi(v);
        return this;
    }

    /**
     * Compute elementwise logical and against a scalar.
     *
     * @param value
     */
    @Override
    public NDArray andi(double value) {
         super.andi(value);
        return this;
    }

    /**
     * Elementwise divide by a scalar (in place).
     *
     * @param v
     */
    @Override
    public DoubleMatrix divi(double v) {
        return super.divi(v);
    }

    /**
     * Matrix-multiply by a scalar.
     *
     * @param v
     */
    @Override
    public DoubleMatrix mmul(double v) {
        return super.mmul(v);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public DoubleMatrix subi(double v) {
        return super.subi(v);
    }

    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public NDArray transpose() {
        NDArray n = new NDArray(data,ArrayUtil.reverseCopy(shape),ArrayUtil.reverseCopy(stride),offset);
        return n;

    }

    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     * @param shape
     * @return
     */
    public NDArray reshape(int[] shape) {
        long ec = 1;
        for (int i = 0; i < shape.length; i++) {
            int si = shape[i];
            if (( ec * si ) != (((int) ec ) * si ))
                throw new IllegalArgumentException("Too many elements");
            ec *= shape[i];
        }
        int n= (int) ec;

        if (ec != n)
            throw new IllegalArgumentException("Too many elements");

        NDArray ndArray = new NDArray(data,shape,stride,offset);
        return ndArray;

    }


    public void checkDimensions(NDArray other) {
        assert Arrays.equals(shape,other.shape) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape);
        assert Arrays.equals(stride,other.stride) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride);
        assert offset == other.offset : "Offset of this array is " + offset + " but other was " + other.offset;

    }



    public NDArray mmul(NDArray arr) {
        List<DoubleMatrix> ret = new ArrayList<>();
        for(int i = 0; i < shape.length; i++) {
            ret.add(slice(i).mmul(arr.slice(i)));
        }

        return new NDArray(ret,ArrayUtil.consArray(shape[0],new int[]{ret.get(0).rows,ret.get(0).columns}));
    }


    public DoubleMatrix sliceDot(DoubleMatrix a) {
        int dims= shape.length;
        switch (dims) {

            case 1: {
                return DoubleMatrix.scalar(SimpleBlas.dot(this,a));
            }
            case 2: {
                return DoubleMatrix.scalar(SimpleBlas.dot(this, a));
            }
        }


        int sc = shape[0];
        DoubleMatrix d = new DoubleMatrix(1,sc);

        for (int i = 0; i < sc; i++)
            d.put(i, slice(i).dot(a));

        return d;
    }


    /**
     * Add a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray add(DoubleMatrix other) {
        NDArray ret = (NDArray) super.addi(other,new NDArray(new double[data.length],shape,stride,offset));
        return ret;
    }

    /**
     * Add a scalar.
     *
     * @param v
     */
    @Override
    public NDArray add(double v) {
        NDArray ret = (NDArray) super.addi(v,new NDArray(new double[data.length],shape,stride,offset));
        return ret;
    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray div(DoubleMatrix other) {
        NDArray ret = (NDArray) super.divi(other,new NDArray(new double[data.length],shape,stride,offset));
        return ret;
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray sub(DoubleMatrix other) {
        NDArray ret = (NDArray) super.subi(other,new NDArray(new double[data.length],shape,stride,offset));
        return ret;
    }

    /**
     * Subtract a scalar.
     *
     * @param v
     */
    @Override
    public NDArray sub(double v) {
        NDArray ret = (NDArray) super.subi(v,new NDArray(new double[data.length],shape,stride,offset));
        return ret;
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray mul(DoubleMatrix other) {
        NDArray ret = (NDArray) super.muli(other,new NDArray(new double[data.length],shape,stride,offset));
        return ret;
    }

    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray muli(double v) {
        NDArray ret = (NDArray) super.muli(v,new NDArray(new double[data.length],shape,stride,offset));
        return ret;
    }

    /**
     * Compare two matrices. Returns true if and only if other is also a
     * DoubleMatrix which has the same size and the maximal absolute
     * difference in matrix elements is smaller than 1e-6.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        if(!o.getClass().isAssignableFrom(NDArray.class))
            return false;
        NDArray n = (NDArray) o;
        if(!Arrays.equals(shape(),n.shape()))
            return false;
        if(!Arrays.equals(stride(),n.stride()))
               return false;

        DoubleMatrix diff = MatrixFunctions.absi(sub(n));

        return diff.max() / (length) < 1e-6;

    }

    /**
     * Returns the shape(dimensions) of this array
     * @return the shape of this matrix
     */
    public int[] shape() {
        return shape;
    }

    /**
     * Returns the stride(indices along the linear index for which each slice is accessed) of this array
     * @return the stride of this array
     */
    public int[] stride() {
        return stride;
    }

    /**
     * Returns the size of this array
     * along a particular dimension
     * @param dimension the dimension to return from
     * @return the shape of the specified dimension
     */
    public int size(int dimension) {
        return shape[dimension];
    }


    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    public NDArray permute(int[] rearrange) {
        checkArrangeArray(rearrange);
        int[] newDims = doPermuteSwap(shape,rearrange);
        int[] newStrides = doPermuteSwap(stride,rearrange);
        NDArray ret = new NDArray(data,newDims,newStrides,offset);

        return ret;
    }

    private int[] doPermuteSwap(int[] shape,int[] rearrange) {
        int[] ret = new int[shape.length];
        for(int i = 0; i < shape.length; i++) {
            ret[i] = shape[rearrange[i]];
        }
        return ret;
    }


    private void checkArrangeArray(int[] arr) {
        assert arr.length == shape.length : "Invalid rearrangement: number of arrangement != shape";
        for(int i = 0; i < arr.length; i++) {
            if (arr[i] >= arr.length)
                throw new IllegalArgumentException("The specified dimensions can't be swapped. Given element " + i + " was >= number of dimensions");
            if (arr[i] < 0)
                throw new IllegalArgumentException("Invalid dimension: " + i + " : negative value");


        }

        for(int i = 0; i < arr.length; i++) {
            for(int j = 0; j < arr.length; j++) {
                if(i != j && arr[i] == arr[j])
                    throw new IllegalArgumentException("Permute array must have unique elements");
            }
        }

    }




}
