package org.deeplearning4j.nn;

import static org.deeplearning4j.util.ArrayUtil.calcStrides;
import static org.deeplearning4j.util.ArrayUtil.reverseCopy;

import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.NDArrayUtil;
import org.jblas.*;
import org.jblas.ranges.Range;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

        if(this.shape.length == 2) {
            rows = shape[0];
            columns = shape[1];
        }
    }



    public NDArray(double[] data,int[] shape,int[] stride) {
        this(data,shape,stride,0);
    }


    public NDArray(double[] data,int[] shape,int[] stride,int offset) {
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");

       this.shape = shape;
        this.offset = offset;
        this.stride = stride;
        this.length = ArrayUtil.prod(shape);

        if(data != null  && data.length > 0)
            this.data = data;


        if(this.shape.length == 2) {
            rows = shape[0];
            columns = shape[1];
        }
    }

    public NDArray(double[] data,int[] shape) {
        this(data,shape,0);
    }

    public NDArray(double[] data,int[] shape,int offset) {
        this(data,shape,calcStrides(shape),offset);
    }



    public NDArray(int[] shape,int[] stride,int offset) {
        this(new double[]{},shape,stride,offset);
    }


    public NDArray(int[] shape,int offset) {
        this(shape,calcStrides(shape),offset);
    }


    public NDArray(int[] shape) {
        this(shape,0);
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public NDArray(int newRows, int newColumns) {
        super(newRows, newColumns);
        if(newRows > 1 && newColumns >1)
            this.shape = new int[]{newRows,newColumns};
            //vector shapes only contain one element
        else if(newRows > 1)
            this.shape = new int[]{newRows};
        else
            this.shape = new int[]{newColumns};
        offset = 0;
        stride = ArrayUtil.calcStrides(this.shape);
        this.length = newRows * newColumns;
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


    /**
     * Copy a row back into the matrix.
     *
     * @param r
     * @param v
     */
    @Override
    public void putRow(int r, DoubleMatrix v) {
        NDArray n = NDArray.wrap(this,v);
        if(n.isVector() && n.length != columns())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");
        for(int i = 0; i < v.length; i++) {
            put(r  + i * stride[0],v.get(i));
        }

    }

    /**
     * Copy a column back into the matrix.
     *
     * @param c
     * @param v
     */
    @Override
    public void putColumn(int c, DoubleMatrix v) {
        NDArray n = NDArray.wrap(this,v);
        if(n.isVector() && n.length != rows())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");
        for(int i = 0; i < v.length; i++) {
            put(c * rows(),v.get(i));
        }

    }

    /**
     * Test whether a matrix is scalar.
     */
    @Override
    public boolean isScalar() {
        return shape.length == 0 || shape.length == 1 && shape[0] == 1;
    }

    /**
     *
     * @param indexes
     * @param value
     * @return
     */
    @Override
    public DoubleMatrix put(int[] indexes, double value) {
        int ix = offset;
        if (indexes.length != shape.length)
            throw new IllegalArgumentException("Unable to set values: number of indices must be equal to the shape");

        for (int i = 0; i< shape.length; i++)
            ix += indexes[i] * stride[i];


        data[ix] = value;
        return this;
    }


    /**
     * Assigns the given matrix (put) to the specified slice
     * @param slice the slice to assign
     * @param put the slice to set
     * @return this for chainability
     */
    public NDArray putSlice(int slice,NDArray put) {
        int[] sliceShape = put.shape();
        int[] requiredShape = ArrayUtil.removeIndex(shape(),0);
        assert Arrays.equals(sliceShape,requiredShape) : "Invalid shape size. Unable to assign slice";
        assert slice <= slices() : "Invalid slice specified " + slice;
        NDArray view = slice(slice);
        if(put.isVector())
            for(int i = 0; i < put.length; i++)
                view.put(i,put.get(i));
        if(put.shape().length == 2) {
            for(int i = 0; i < put.rows(); i++) {
                for(int j = 0; j < put.columns(); j++) {
                    view.put(i,j,put.get(i,j));
                }
            }
        }
        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for(int i = 0; i < put.slices(); i++) {
                view.slice(i).putSlice(i,view.slice(i));
            }
        }

        return this;

    }


    public NDArray get(Object...o) {


        int[] shape = shapeFor(shape(),o,true);
        int[] indexingShape = shapeFor(shape(),o,false);
        int[] stride = calcStrides(shape);
        int[] query = queryForObject(shape(),o);

        if(query.length == 1)
            return NDArray.scalar(this,query[0]);



        //promising
        int index = offset + indexingShape[0] * stride[0];
        //int[] baseLineIndices = new int[]
        return new NDArray(data,
                shape,
                stride,
                index);
    }


    public int[] offsetsForSlices(int[] shape) {
        int[] ret = new int[shape.length];
        for(int i = 0; i < shape.length; i++) {
            ret[i] = offset + shape[i] - this.shape[i];
        }
        return ret;
    }


    public int[] offsetsForSlices() {
        return offsetsForSlices(shape);
    }


    public NDArray subArray(int[] shape) {
        return subArray(offsetsForSlices(),shape);
    }


    /**
     * Number of slices: aka shape[0]
     * @return the number of slices
     * for this nd array
     */
    public int slices() {
        return shape[0];
    }



    public NDArray subArray(int[] offsets, int[] shape,int[] stride) {
        int n = shape.length;
        if (offsets.length != n)
            throw new IllegalArgumentException("Invalid offset " + Arrays.toString(offsets));
        if (shape.length != n)
            throw new IllegalArgumentException("Invalid shape " + Arrays.toString(shape));

        if (Arrays.equals(shape, this.shape)) {
            if (ArrayUtil.isZero(offsets)) {
                return this;
            } else {
                throw new IllegalArgumentException("Invalid subArray offsets");
            }
        }

        return new NDArray(
                data
                , Arrays.copyOf(shape,shape.length)
                , stride
                ,offset + ArrayUtil.dotProduct(offsets, stride)
        );
    }




    public NDArray subArray(int[] offsets, int[] shape) {
        return subArray(offsets,shape,stride);
    }



    public static int[] queryForObject(int[] shape,Object[] o) {
        //allows us to put it in to shape format
        Object[] copy =  o;
        int[] ret = new int[copy.length];
        for(int i = 0; i < copy.length; i++) {
            //give us the whole thing
            if(copy[i] == ':')
                ret[i] = shape[i];
                //only allow indices
            else if(copy[i] instanceof Number)
                ret[i] = (Integer) copy[i];
            else if(copy[i] instanceof Range) {
                Range r = (Range) copy[i];
                int len = MatrixUtil.toIndices(r).length;
                ret[i] = len;
            }
            else
                throw new IllegalArgumentException("Unknown kind of index of type: " + o[i].getClass());

        }


        //drop all shapes of 0
        int[] realRet = ret;

        for(int i = 0; i < ret.length; i++) {
            if(ret[i] <= 0)
                realRet = ArrayUtil.removeIndex(ret,i);
        }


        return realRet;
    }




    public static Integer[] queryForObject(Integer[] shape,Object[] o,boolean dropZeros) {
        //allows us to put it in to shape format
        Object[] copy = o;
        Integer[] ret = new Integer[o.length];
        for(int i = 0; i < o.length; i++) {
            //give us the whole thing
            if(copy[i] == ':')
                ret[i] = shape[i];
                //only allow indices
            else if(copy[i] instanceof Number)
                ret[i] = (Integer) copy[i];
            else if(copy[i] instanceof Range) {
                Range r = (Range) copy[i];
                int len = MatrixUtil.toIndices(r).length;
                ret[i] = len;
            }
            else
                throw new IllegalArgumentException("Unknown kind of index of type: " + o[i].getClass());

        }

        if(!dropZeros)
            return ret;

        //drop all shapes of 0
        Integer[] realRet = ret;

        for(int i = 0; i < ret.length; i++) {
            if(ret[i] <= 0)
                realRet = ArrayUtil.removeIndex(ret,i);
        }



        return realRet;
    }



    public static int[] shapeFor(int[] shape,Object[] o,boolean dropZeros) {
        //allows us to put it in to shape format
        Object[] copy = reverseCopy(o);
        int[] ret = new int[copy.length];
        for(int i = 0; i < copy.length; i++) {
            //give us the whole thing
            if(copy[i] == ':')
                ret[i] = shape[i];
                //only allow indices
            else if(copy[i] instanceof Number)
                ret[i] = (Integer) copy[i];
            else if(copy[i] instanceof Range) {
                Range r = (Range) copy[i];
                int len = MatrixUtil.toIndices(r).length;
                ret[i] = len;
            }
            else
                throw new IllegalArgumentException("Unknown kind of index of type: " + o[i].getClass());

        }


        if(!dropZeros)
            return ret;


        //drop all shapes of 0
        int[] realRet = ret;

        for(int i = 0; i < ret.length; i++) {
            if(ret[i] <= 0)
                realRet = ArrayUtil.removeIndex(ret,i);
        }


        return realRet;
    }




    public static Integer[] shapeForObject(Integer[] shape,Object[] o) {
        //allows us to put it in to shape format
        Object[] copy = reverseCopy(o);
        Integer[] ret = new Integer[o.length];
        for(int i = 0; i < o.length; i++) {
            //give us the whole thing
            if(copy[i] == ':')
                ret[i] = shape[i];
                //only allow indices
            else if(copy[i] instanceof Number)
                ret[i] = (Integer) copy[i];
            else if(copy[i] instanceof Range) {
                Range r = (Range) copy[i];
                int len = MatrixUtil.toIndices(r).length;
                ret[i] = len;
            }
            else
                throw new IllegalArgumentException("Unknown kind of index of type: " + o[i].getClass());

        }

        //drop all shapes of 0
        Integer[] realRet = ret;

        for(int i = 0; i < ret.length; i++) {
            if(ret[i] <= 0)
                realRet = ArrayUtil.removeIndex(ret,i);
        }



        return realRet;
    }


    @Override
    public DoubleMatrix put(int i, double v) {
        data[i + offset] = v;
        return this;
    }

    @Override
    public double get(int i) {
        if(shape().length > 1)
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int realStride = stride == null || stride.length < 1 ? 1 : stride[0];
        return data[offset + i * realStride];
    }


    /**
     * Returns the specified slice of this matrix.
     * In matlab, this would be equivalent to (given a 2 x 2 x 2):
     * A(:,:,x) where x is the slice you want to return.
     *
     * The slice is always relative to the final dimension of the matrix.
     *
     * @param slice the slice to return
     * @return the specified slice of this matrix
     */
    public NDArray slice(int slice) {

        if (shape.length == 0) {
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");
        }
        //slice of a vector is a scalar
        else if (shape.length == 1) {
            return new NDArray(data,new int[]{},new int[]{},offset + slice * stride[0]);
        }
        //slice of a matrix is a vector
        else if (shape.length == 2) {
            int st = stride[1];
            if (st == 1) {
                return new NDArray(data, new int[]{ shape[1] },new int[]{1},offset + slice * stride[0]);
            } else {

                return new NDArray(data,new int[]{ 1,shape[1] } ,new int[]{ stride[1] },offset + slice * stride[0]);
            }
        } else {
            return new NDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset + slice * stride[0]);
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
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return data[ix];
    }


    /** Retrieve matrix element */
    public double get(int rowIndex, int columnIndex) {
        return getMulti(new int[]{rowIndex, columnIndex});
    }

    @Override
    public NDArray get(int[] indices) {
        NDArray result = new NDArray(data,new int[]{1,indices.length},stride,offset);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, get(indices[i] + offset));
        }

        return result;
    }


    /**
     * Mainly an internal method (public for testing)
     * for given an offset and stride, and index,
     * calculating the beginning index of a query given indices
     * @param offset the desired offset
     * @param stride the desired stride
     * @param indexes the desired indexes to test on
     * @return the index for a query given stride and offset
     */
    public static int getIndex(int offset,int[] stride,int...indexes) {
        if(stride.length > indexes.length)
            throw new IllegalArgumentException("Invalid number of items in stride array: should be <= number of indexes");

        int ix = offset;


        for (int i = 0; i < indexes.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return ix;
    }

    /**
     * Returns the begin index of a query
     * given the stride, array offset
     * @param indexes the desired indexes to test on
     * @return the index of the begin of this query
     */
    public int getIndex(int... indexes) {
        return getIndex(offset,stride,indexes);
    }


    private void ensureSameShape(NDArray arr1,NDArray arr2) {
        assert true == Arrays.equals(arr1.shape(),arr2.shape());

    }

    /**
     * Return index of minimal element per column.
     */
    @Override
    public int[] columnArgmaxs() {
        return super.columnArgmaxs();
    }

    /**
     * Return index of minimal element per column.
     */
    @Override
    public int[] columnArgmins() {
        if(shape().length == 2)
            return super.columnArgmins();
        else {
            throw new IllegalStateException("Unable to get column mins for dimensions more than 2");
        }
    }




    /**
     * Return column-wise maximums.
     */
    @Override
    public NDArray columnMaxs() {
        if(shape().length == 2)
            return NDArray.wrap(super.columnMaxs());

        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.COLUMN_MAX,this);

    }

    /**
     * Return a vector containing the means of all columns.
     */
    @Override
    public NDArray columnMeans() {
        if(shape().length == 2) {
            return NDArray.wrap(super.columnMeans());

        }

        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.COLUMN_MEAN,this);

    }

    /**
     * Return column-wise minimums.
     */
    @Override
    public NDArray columnMins() {
        if(shape().length == 2) {
            return NDArray.wrap(super.columnMins());

        }
        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.COLUMN_MIN,this);

    }

    /**
     * Return a vector containing the sums of the columns (having number of columns many entries)
     */
    @Override
    public NDArray columnSums() {
        if(shape().length == 2) {
            return NDArray.wrap(super.columnSums());

        }
        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.COLUMN_SUM,this);

    }

    /**
     * Computes the cumulative sum, that is, the sum of all elements
     * of the matrix up to a given index in linear addressing.
     */
    @Override
    public DoubleMatrix cumulativeSum() {
        return super.cumulativeSum();
    }

    /**
     * Computes the cumulative sum, that is, the sum of all elements
     * of the matrix up to a given index in linear addressing (in-place).
     */
    @Override
    public DoubleMatrix cumulativeSumi() {
        return super.cumulativeSumi();
    }


    /**
     * Return index of minimal element per row.
     */
    @Override
    public int[] rowArgmaxs() {
        return super.rowArgmaxs();
    }

    /**
     * Return index of minimal element per row.
     */
    @Override
    public int[] rowArgmins() {
        return super.rowArgmins();
    }

    /**
     * Return row-wise maximums for each slice.
     */
    @Override
    public NDArray rowMaxs() {
        if(shape().length == 2) {
            return NDArray.wrap(super.rowMaxs());

        }
        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.ROW_MAX,this);

    }

    /**
     * Return a vector containing the means of the rows for each slice.
     */
    @Override
    public NDArray rowMeans() {
        if(shape().length == 2) {
            return NDArray.wrap(super.rowMeans());

        }
        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.ROW_MEAN,this);

    }

    /**
     * Return row-wise minimums for each slice.
     */
    @Override
    public NDArray rowMins() {
        if(shape().length == 2) {
            return NDArray.wrap(super.rowMins());

        }
        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.ROW_MIN,this);

    }

    /**
     * Return a matrix with the row sums for each slice
     */
    @Override
    public NDArray rowSums() {
        if(shape().length == 2) {
            return NDArray.wrap(super.rowSums());

        }

        else
            return NDArrayUtil.doSliceWise(NDArrayUtil.MatrixOp.ROW_SUM,this);

    }

    /** Add a scalar to a matrix (in-place). */
    public NDArray addi(double v, NDArray result) {

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) + v);
        }
        return result;
    }

    /** Add two matrices (in-place). */
    public NDArray addi(NDArray other, NDArray result) {
        ensureSameShape(other,result);

        if (other.isScalar()) {
            return addi(other.get(0), result);
        }

        if (isScalar()) {
            return other.addi(get(0), result);
        }

        assertSameLength(other);

        if (result == this) {
            SimpleBlas.axpy(1.0, other, result);
        } else if (result == other) {
            SimpleBlas.axpy(1.0, this, result);
        } else {
            /*SimpleBlas.copy(this, result);
            SimpleBlas.axpy(1.0, other, result);*/
            JavaBlas.rzgxpy(length, result.data, data, other.data);
        }

        return result;
    }


    /**
     * Add a scalar to a matrix (in-place).
     * @param result
     */
    @Override
    public NDArray addi(DoubleMatrix result) {
        return add(result);
    }

    /**
     * Subtract two matrices (in-place).
     *
     * @param result
     */
    @Override
    public NDArray subi(DoubleMatrix result) {
        return sub(result);
    }



    /**
     * Elementwise multiply by a scalar.
     *
     * @param v
     */
    @Override
    public NDArray mul(double v) {
        int dims= shape().length;
        if (dims == 0) {
            put(0, data[0] * v);
        } else {
            int n = slices();
            for (int i = 0; i < n; i++)
                slice(i).muli(v);

        }
        return this;
    }


    /**
     * Elementwise multiplication (in-place).
     *
     * @param result
     */
    @Override
    public NDArray muli(DoubleMatrix result) {
        return mul(result);
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
        return div(result);
    }

    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray addi(double v) {
        return add(v);
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
    public NDArray divi(double v) {
        int dims= shape().length;
        if (dims == 0) {
            put(0, data[0] / v);
        } else {
            int n = slices();
            for (int i = 0; i < n; i++)
                slice(i).divi(v);

        }
        return this;
    }

    /**
     * Matrix-multiply by a scalar.
     *
     * @param v
     */
    @Override
    public NDArray mmul(double v) {
        return mmul(NDArray.scalar(v));
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray subi(double v) {
        return sub(v);
    }

    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public NDArray transpose() {
        NDArray n = new NDArray(data,reverseCopy(shape),ArrayUtil.reverseCopy(stride),offset);
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



    @Override
    public NDArray mmul(DoubleMatrix a) {
        NDArray arr = NDArray.wrap(this,a);
        List<DoubleMatrix> ret = new ArrayList<>();

        if(shape().length == 2) {
            rows = shape[0];
            columns = shape[1];
            return NDArray.wrap(super.mmul(a));
        }

        for(int i = 0; i < shape.length; i++) {
            ret.add(slice(i).mmul(arr.slice(i)));
        }

        return new NDArray(ret,ArrayUtil.consArray(shape[0],new int[]{ret.get(0).rows,ret.get(0).columns}));
    }


    public DoubleMatrix sliceDot(DoubleMatrix a) {
        int dims = shape.length;
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
     * @param o
     */
    @Override
    public NDArray add(DoubleMatrix o) {
        NDArray other = (NDArray) o;
        int dims = shape.length;
        if (dims == 0 || shape.length == 1 && shape[0] == 1) {
            return add(data[0]);
        }


        int adims = other.shape().length;
        int n = slices();
        int na = other.slices();
        if (dims == adims) {
            if (n != na)
                throw new IllegalArgumentException("Must have same number of slices");
            for (int i = 0; i<n; i++) {
                slice(i).add(other.slice(i));
            }
        }
        else if (adims < dims) {
            for (int i = 0; i < n; i++) {
                slice(i).add(other);
            }
        } else {
            throw new IllegalArgumentException("Invalid shapes for addition");
        }
        return this;
    }

    /**
     * Add a scalar.
     *
     * @param v
     */
    @Override
    public NDArray add(double v) {
        if (isVector() || isScalar()) {
            put(0, v + data[0]);
        }
        else {

            int n = slices();
            for (int i = 0; i < n; i++)
                slice(i).add(v);

        }
        return this;

    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray div(DoubleMatrix other) {
        NDArray a = NDArray.wrap(this,other);
        int dims = shape().length;
        if (dims == 0) {
            put(0, data[0] / a.data[0]);
            return this;
        }

        int adims = a.shape().length;
        if (adims == 0) {
            muli(1.0 / a.data[0]);
            return this;
        }

        int n = slices();
        int na = a.slices();

        if (dims == adims) {
            if (n != na)
                throw new IllegalArgumentException("Invalid shapes for element wise operator");
            for (int i = 0; i < n; i++) {
                slice(i).div(a.slice(i));
            }
        } else if (adims < dims) {
            for (int i = 0; i < n; i++) {
                slice(i).div(a);
            }
        } else {
            throw new IllegalArgumentException("Invalid shapes for element wise operator");
        }

        return this;
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray sub(DoubleMatrix other) {
        NDArray a = NDArray.wrap(this,other);
        int dims = shape().length;
        if (dims==0) {
            sub(data[0]);
            return this;
        }

        int n = slices();
        int na= a.slices();
        int adims = a.shape().length;
        if (dims==adims) {
            if (n!=na)
                throw new IllegalArgumentException("Invalid shapes ");
            for (int i=0; i<n; i++) {
                slice(i).sub(a.slice(i));
            }
        }

        else if (adims < dims) {
            for (int i=0; i < n; i++) {
                slice(i).sub(a);
            }
        } else {
            throw new IllegalArgumentException("Invalid shape ");
        }
        return this;
    }

    /**
     * Subtract a scalar.
     *
     * @param v
     */
    @Override
    public NDArray sub(double v) {
        int dims= shape().length;
        if (dims == 0 || shape().length == 1 && shape[0] == 1) {
            put(0, v - data[0]);
        } else {
            int n = slices();
            for (int i = 0; i < n; i++)
                slice(i).sub(v);

        }
        return this;
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray mul(DoubleMatrix other) {
        NDArray a = NDArray.wrap(this,other);
        int dims= a.shape().length;
        if (dims == 0) {
            put(0,data[0] * a.data[0]);
            return this;
        }

        int adims = a.shape().length;
        if (adims == 0) {
            muli(a.data[0]);
            return this;
        }

        int n = slices();
        int na = a .slices();
        if (dims == adims) {
            if (n != na)
                throw new IllegalArgumentException("Invalid shape");
            for (int i = 0; i < n; i++) {
                slice(i).muli(a.slice(i));
            }
        } else if (adims < dims) {
            for (int i  =0; i < n; i++) {
                slice(i).muli(a);
            }
        } else {
            throw new IllegalArgumentException("Invalid shape");
        }

        return this;
    }

    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray muli(double v) {
        int dims= shape().length;
        if (dims == 0 || shape().length == 1 && shape[0] == 1) {
            put(0, v * data[0]);
        } else {
            int n = slices();
            for (int i = 0; i < n; i++)
                slice(i).muli(v);

        }
        return this;
    }

    /**
     * Create a new matrix with <i>newRows</i> rows, <i>newColumns</i> columns
     * using <i>newData></i> as the data. The length of the data is not checked!
     *
     * @param newRows
     * @param newColumns
     * @param newData
     */
    public NDArray(int newRows, int newColumns, double... newData) {
        super(newRows, newColumns, newData);
    }

    /**
     * Computes the sum of all elements of the matrix.
     */
    @Override
    public double sum() {
        if(isVector() || shape().length == 2)
            return super.sum();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.SUM,this);
    }

    /**
     * Computes the product of all elements of the matrix
     */
    @Override
    public double prod() {
        if(isVector() || shape().length == 2)
            return super.prod();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.PROD,this);

    }

    /**
     * Returns the maximal element of the matrix.
     */
    @Override
    public double max() {
        if(isVector() || shape().length == 2)
            return super.max();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MAX,this);

    }

    /**
     * Returns the minimal element of the matrix.
     */
    @Override
    public double min() {
        if(isVector() || shape().length == 2)
            return super.min();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MIN,this);

    }

    /**
     * Computes the mean value of all elements in the matrix,
     * that is, <code>x.sum() / x.length</code>.
     */
    @Override
    public double mean() {
        if(isVector() || shape().length == 2)
            return super.mean();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MEAN,this);

    }

    /**
     * Returns the linear index of the maximal element of the matrix. If
     * there are more than one elements with this value, the first one
     * is returned.
     */
    @Override
    public int argmax() {
        if(isVector() || shape().length == 2)
            return super.argmax();
        return (int) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.ARG_MAX,this);

    }

    /**
     * Returns the linear index of the minimal element. If there are
     * more than one elements with this value, the first one is returned.
     */
    @Override
    public int argmin() {
        if(isVector() || shape().length == 2)
            return super.argmin();
        return (int) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.ARG_MIN,this);

    }

    /**
     * Converts the matrix to a one-dimensional array of doubles.
     */
    @Override
    public double[] toArray() {
        length = ArrayUtil.prod(shape);
        double[] ret = new double[length];
        for(int i = 0; i < ret.length; i++)
            ret[i] = data[ offset + i];
        return ret;
    }


    public int columns() {
        if(shape.length == 2)
            return shape[1];
        throw new IllegalStateException("Unable to get number of of rows for a non 2d matrix");
    }

    public int rows() {
        if(shape.length == 2)
            return shape[0];
        throw new IllegalStateException("Unable to get number of rows for a non 2d matrix");
    }





    /**
     * Reshape the matrix. Number of elements must not change.
     *
     * @param newRows
     * @param newColumns
     */
    @Override
    public NDArray reshape(int newRows, int newColumns) {
        return reshape(new int[]{newRows,newColumns});
    }

    /**
     * Get the specified column
     *
     * @param c
     */
    @Override
    public NDArray getColumn(int c) {
        if(shape.length == 2)
            return new NDArray(
                    data,
                    new int[]{shape[0]},
                    new int[]{1},
                    offset + c * rows()
            );
        else
            throw new IllegalArgumentException("Unable to get row of non 2d matrix");
    }


    /**
     * Get a copy of a row.
     *
     * @param r
     */
    @Override
    public NDArray getRow(int r) {
        if(shape.length == 2)
            return new NDArray(
                    data,
                    new int[]{shape[1]},
                    new int[]{shape[0]},
                    offset + r * columns()
            );
        else
            throw new IllegalArgumentException("Unable to get row of non 2d matrix");
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
        NDArray n = null;
        if(o.getClass().isAssignableFrom(DoubleMatrix.class)) {
            //chance for comparison of the matrices if the shape of this matrix is 2
            if(shape().length > 2)
                return false;
            else {
                DoubleMatrix d = (DoubleMatrix) o;
                n = NDArray.wrap(this,d);

            }
        }
        else if(!o.getClass().isAssignableFrom(NDArray.class))
            return false;

        if(n == null)
            n = (NDArray) o;


        if(!Arrays.equals(shape(),n.shape()))
            return false;
        if(!Arrays.equals(stride(),n.stride()))
            return false;
        //epsilon equals
        if(isScalar())
            return Math.abs(get(0) - n.get(0)) < 1e-6;
        else if(isVector()) {
            for(int i = 0; i < length; i++)
                if(Math.abs(get(i) - n.get(i)) > 1e-6)
                    return false;
        }

        for (int i=0; i< slices(); i++) {
            if (!(slice(i).equals(n.slice(i))))
                return false;
        }

        return true;


    }







    public List<NDArray> rowList() {
        if(shape.length == 1)
            return new ArrayList<>(Collections.singletonList(NDArray.scalar(this, 1)));
        else if(shape.length == 2) {

        }

        return new ArrayList<>();
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


    public int offset() {
        return offset;
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


    /**
     * Checks whether the matrix is a vector.
     */
    @Override
    public boolean isVector() {
        return shape.length == 1;
    }

    /** Generate string representation of the matrix. */
    @Override
    public String toString() {
        if (shape.length == 1) {
            return Double.toString(get(0));
        }

        StringBuilder sb = new StringBuilder();
        int length= shape[0];
        sb.append('[');
        if (length > 0) {
            sb.append(slice(0).toString());
            for (int i = 1; i < length; i++) {
                sb.append(',');
                sb.append(slice(i).toString());
            }
        }
        sb.append(']');
        return sb.toString();
    }

    /**
     * Generate string representation of the matrix, with specified
     * format for the entries. For example, <code>x.toString("%.1f")</code>
     * generates a string representations having only one position after the
     * decimal point.
     */
    @Override
    public String toString(String fmt) {
        return toString(fmt, "[", "]", ", ", "; ");
    }

    /**
     * Generate string representation of the matrix, with specified
     * format for the entries, and delimiters.
     *
     * @param fmt entry format (passed to String.format())
     * @param open opening parenthesis
     * @param close closing parenthesis
     * @param colSep separator between columns
     * @param rowSep separator between rows
     */
    @Override
    public String toString(String fmt, String open, String close, String colSep, String rowSep) {
        StringWriter s = new StringWriter();
        PrintWriter p = new PrintWriter(s);

        p.print(open);

        if(shape.length == 1)
            return Arrays.toString(data);

        if(shape.length == 2) {
            rows = shape[0];
            columns = shape[1];
            return super.toString(fmt,open,close,colSep,rowSep);
        }
        else {
            if (shape.length == 0) {
                return Double.toString(data[0]);
            }

            StringBuilder sb = new StringBuilder();
            int length = shape[0];
            sb.append(open);
            if (length > 0) {
                sb.append(slice(0).toString());
                for (int i = 1; i < length; i++) {
                    sb.append(',');
                    sb.append(slice(i).toString());
                }
            }
            sb.append(close);
            return sb.toString();
        }

    }


    public static NDArray scalar(NDArray from,int index) {
        return new NDArray(from.data,new int[]{1},new int[]{1},index);
    }


    public static NDArray scalar(double num) {
        return new NDArray(new double[]{num},new int[]{1},new int[]{1},0);
    }

    public static NDArray wrap(NDArray ndArray,DoubleMatrix toWrap) {
        if(toWrap instanceof NDArray)
            return (NDArray) toWrap;
        NDArray ret = new NDArray(toWrap.data,ndArray.shape(),ndArray.stride(),0);
        return ret;
    }


    public static NDArray wrap(DoubleMatrix toWrap) {
        if(toWrap instanceof NDArray)
            return (NDArray) toWrap;
        int[] shape = new int[]{toWrap.rows,toWrap.columns};
        NDArray ret = new NDArray(toWrap.data,shape,ArrayUtil.calcStrides(shape));
        return ret;
    }




}
