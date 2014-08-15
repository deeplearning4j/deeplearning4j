package org.deeplearning4j.linalg;

import jcuda.jcublas.JCublas;
import jcuda.Pointer;
import jcuda.Sizeof;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.DimensionSlice;
import org.deeplearning4j.linalg.api.ndarray.SizeException;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.TwoArrayOps;
import org.deeplearning4j.linalg.ops.elementwise.AddOp;
import org.deeplearning4j.linalg.ops.elementwise.DivideOp;
import org.deeplearning4j.linalg.ops.elementwise.MultiplyOp;
import org.deeplearning4j.linalg.ops.elementwise.SubtractOp;
import org.deeplearning4j.linalg.ops.reduceops.Ops;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.IterationResult;
import org.deeplearning4j.linalg.util.Shape;






import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Arrays;

import static org.deeplearning4j.linalg.util.ArrayUtil.calcStrides;
import static org.deeplearning4j.linalg.util.ArrayUtil.reverseCopy;

public class JCublasNDArray implements INDArray {
    private int[] shape;
    private int[] stride;
    private int offset = 0;
    public int rows;
    /** Number of columns. */
    public int columns;
    /** Total number of elements (for convenience). */
    public int length;
    /** The actual data stored by rows (that is, row 0, row 1...). */
    public double[] data = null; // rows are contiguous

    public void checkDimensions(JCublasNDArray other) {
        assert Arrays.equals(shape,other.shape()) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape());
        assert Arrays.equals(stride,other.stride()) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride());
        assert offset == other.offset() : "Offset of this array is " + offset + " but other was " + other.offset();

    }
    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public JCublasNDArray(int[] shape,int[] stride,int offset) {this(new double[ArrayUtil.prod(shape)],shape,stride,offset);}
    public JCublasNDArray(int[] shape,int offset) {
        this(shape,calcStrides(shape),offset);
    }
    public JCublasNDArray(int[] shape) { this(shape,0); }
    public JCublasNDArray(int[] shape,int[] stride){
        this(shape,stride,0);
    }
    public JCublasNDArray(double[] data,int[] shape) {
        this(data,shape,0);
    }
    public JCublasNDArray(double[] data,int[] shape,int offset) { this(data, shape, calcStrides(shape), offset); }
    public JCublasNDArray(int newRows, int newColumns) {
        this(newRows, newColumns, new double[newRows * newColumns]);
    }
    public JCublasNDArray(int newRows, int newColumns, double... newData) {
        rows = newRows;
        columns = newColumns;
        length = rows * columns;

        if (newData != null && newData.length != newRows * newColumns) {
            throw new IllegalArgumentException(
                    "Passed data must match matrix dimensions.");
        }

        data = newData;
        //System.err.printf("%d * %d matrix created\n", rows, columns);
    }
    public JCublasNDArray(double[] data,int[] shape,int[] stride,int offset) {
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");

        this.offset = offset;
        this.stride = stride;

        initShape(shape);

        if(data != null  && data.length > 0)
            this.data = data;
    }

    public JCublasNDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    public JCublasNDArray diviColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).divi(columnVector.getScalar(i));
        }
        return this;
    }

    @Override
    public JCublasNDArray getColumn(int c) {
        if(shape.length == 2)
            return new JCublasNDArray(
                    data,
                    new int[]{shape[0]},
                    new int[]{stride[0]},
                    offset + c
            );
        else
            throw new IllegalArgumentException("Unable to getFromOrigin column of non 2d matrix");
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

    public JCublasNDArray dup() {
        double[] dupData = new double[data.length];
        System.arraycopy(data,0,dupData,0,dupData.length);
        JCublasNDArray ret = new JCublasNDArray(dupData,shape,stride,offset);
        return ret;
    }


    public JCublasNDArray subColumnVector(INDArray columnVector) {
        return dup().subiColumnVector(columnVector);
    }

//    @Override
    public JCublasNDArray subiRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).subi(rowVector.getScalar(i));
        }
        return this;
    }

    @Override
    public JCublasNDArray subRowVector(INDArray rowVector) {
            return dup().subiRowVector(rowVector);
    }

    @Override
    public JCublasNDArray addiColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).addi(columnVector.getScalar(i));
        }
        return this;
    }

    @Override
    public JCublasNDArray addColumnVector(INDArray columnVector) {
        return dup().addiColumnVector(columnVector);
    }

    public JCublasNDArray subiColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).subi(columnVector.getScalar(i));
        }
        return this;
    }


    @Override
    public boolean isVector() {
        return shape.length == 1
                ||
                shape.length == 1  && shape[0] == 1
                ||
                shape.length == 2 && (shape[0] == 1 || shape[1] == 1);
    }

    @Override
    public JCublasNDArray subi(INDArray other) {
        return subi(other,this);
    }
    @Override
    public JCublasNDArray subi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(SubtractOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(SubtractOp.class)
                    .to(result).build().exec();
        return (JCublasNDArray) result;
    }

    @Override
    public int[] endsForSlices() {
        int[] ret = new int[slices()];
        int currOffset = offset + stride[0] - 1;
        for(int i = 0; i < slices(); i++) {
            ret[i] = currOffset;
            currOffset += stride[0];
        }
        return ret;
    }

    @Override
    public JCublasNDArray reduce(Ops.DimensionOp op,int dimension) {
        if(isScalar())
            return this;


        if(isVector())
            return JCublasNDArray.scalar(reduceVector(op, this));


        int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

        if(dimension == 0) {
            double[] data2 = new double[ArrayUtil.prod(shape)];
            int dataIter = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for(int offset = this.offset; offset < numTimes; offset++) {
                double reduce = op(dimension, offset, op);
                data2[dataIter++] = reduce;

            }

            return new JCublasNDArray(data2,shape);
        }

        else {
            double[] data2 = new double[ArrayUtil.prod(shape)];
            int dataIter = 0;
            //want the milestone to slice[1] and beyond
            int[] sliceIndices = endsForSlices();
            int currOffset = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for(int offset = this.offset; offset < numTimes; offset++) {
                if(dataIter >= data2.length || currOffset >= sliceIndices.length)
                    break;

                //do the operation,, and look for whether it exceeded the current slice
                IterationResult pair = op(dimension, offset, op,sliceIndices[currOffset]);
                //append the result
                double reduce = pair.getResult();
                data2[dataIter++] = reduce;

                //go to next slice and iterate over that
                if(pair.isNextSlice()) {
                    //will update to next step
                    offset = sliceIndices[currOffset];
                    numTimes +=  sliceIndices[currOffset];
                    currOffset++;
                }

            }

            return new JCublasNDArray(data2,shape);
        }


    }

    @Override
    public JCublasNDArray putSlice(int slice, INDArray put) {
        if(isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            put(0,put.getScalar(0));
            return this;
        }

        else if(isVector()) {
            assert put.isScalar() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            put(slice,put.getScalar(0));
            return this;
        }


        assertSlice(put,slice);


        JCublasNDArray view = slice(slice);

        if(put.isScalar())
            put(slice,put.getScalar(0));
        else if(put.isVector())
            for(int i = 0; i < put.length(); i++)
                view.put(i,put.getScalar(i));
        else if(put.shape().length == 2)
            for(int i = 0; i < put.rows(); i++)
                for(int j = 0; j < put.columns(); j++)
                    view.put(i,j,(double) put.getScalar(i,j).element());

        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for(int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i,view.slice(i));

        }

        return this;
    }

    private void assertSlice(INDArray put,int slice) {
        assert slice <= slices() : "Invalid slice specified " + slice;
        int[] sliceShape = put.shape();
        int[] requiredShape = ArrayUtil.removeIndex(shape(),0);

        //no need to compare for scalar; primarily due to shapes either being [1] or length 0
        if(put.isScalar())
            return;



        assert Shape.shapeEquals(sliceShape,requiredShape) : String.format("Invalid shape size of %s . Should have been %s ",Arrays.toString(sliceShape),Arrays.toString(requiredShape));

    }
    private double reduceVector(Ops.DimensionOp op,JCublasNDArray vector) {

        switch(op) {
            case SUM:
                return vector.sum();
            case MEAN:
                return vector.mean();
            case MIN:
                return vector.min();
            case MAX:
                return vector.max();
            case NORM_1:
                return vector.norm1();
            case NORM_2:
                return vector.norm2();
            case NORM_MAX:
                return vector.normmax();
            default: throw new IllegalArgumentException("Illegal operation");
        }
    }
    public boolean isEmpty() {
        return columns == 0 || rows == 0;
    }

    public double norm1() {
        double norm = 0.0;
        for (int i = 0; i < length; i++) {
            norm += Math.abs(get(i));
        }
        return norm;
    }
    public double norm2() {
        double norm = 0.0;
        for (int i = 0; i < length; i++) {
            norm += get(i) * get(i);
        }
        return (double) Math.sqrt(norm);
    }
    public double sum() {
        double s = 0.0;
        for (int i = 0; i < length; i++) {
            s += get(i);
        }
        return s;
    }

    @Override
    public INDArray sum(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return JCublasNDArray.scalar(reshape(new int[]{1,length}).sum());
        }

        else if(isVector()) {
            return JCublasNDArray.scalar(sum());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.sum(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.sum(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape).transpose();
        }
    }

    public double mean() {
        return sum() / length;
    }
    @Override
    public INDArray mean(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return JCublasNDArray.scalar(reshape(new int[]{1,length}).mean());
        }
        else if(isVector()) {
            return JCublasNDArray.scalar(sum() / length());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.mean(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.mean(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }
    public double min() {
        if (isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }
        double v = Double.POSITIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (!Double.isNaN(get(i)) && get(i) < v) {
                v = get(i);
            }
        }

        return v;
    }

    public double max() {
        if (isEmpty()) {
            return Double.NEGATIVE_INFINITY;
        }
        double v = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (!Double.isNaN(get(i)) && get(i) > v) {
                v = get(i);
            }
        }
        return v;
    }
    public boolean isMatrix() {
        return (shape().length == 2
                && (shape[0] != 1 && shape[1] != 1)) ||
                shape.length == 3 &&
                        (shape[0] == 1 || shape[1] == 1 || shape[2] == 1);
    }

    @Override
    public int length() {
        return length;
    }

    private void initShape(int[] shape) {
        this.shape = shape;

        if(this.shape.length == 1) {
            rows = 1;
            columns = this.shape[0];
        }
        else if(this.shape().length == 2) {
            if(shape[0] == 1) {
                this.shape = new int[1];
                this.shape[0] = shape[1];
                rows = 1;
                columns = shape[1];
            }
            else {
                rows = shape[0];
                columns = shape[1];
            }


        }

        //default row vector
        else if(this.shape.length == 1) {
            columns = this.shape[0];
            rows = 1;
        }



        this.length = ArrayUtil.prod(this.shape);
        if(this.stride == null)
            this.stride = ArrayUtil.calcStrides(this.shape);

        //recalculate stride: this should only happen with row vectors
        if(this.stride.length != this.shape.length) {
            this.stride = ArrayUtil.calcStrides(this.shape);
        }

    }

    /**
     * Number of columns (shape[1]), throws an exception when
     * called when not 2d
     * @return the number of columns in the array (only 2d)
     */
    public int columns() {
        if(isMatrix()) {
            if (shape().length > 2)
                return Shape.squeeze(shape)[1];
            else if (shape().length == 2)
                return shape[1];
        }
        if(isVector()) {
            if(isColumnVector())
                return 1;
            else
                return shape[0];
        }
        throw new IllegalStateException("Unable to getFromOrigin number of of rows for a non 2d matrix");
    }

    /**
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    public JCublasNDArray ravel() {
        JCublasNDArray ret = new JCublasNDArray(new int[]{1,length});
        List<JCublasNDArray> list = new ArrayList<>();
        sliceVectors(list);
        int count = 0;
        for(int i = 0; i < list.size(); i++) {
            for(int j = 0; j < list.get(i).length; j++)
                ret.put(count++,list.get(i).getScalar(j));
        }
        return ret;
    }

    /**
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    private void sliceVectors(java.util.List<JCublasNDArray> list) {
        if(isVector())
            list.add(this);
        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
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
    public JCublasNDArray slice(int slice) {

        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            return new JCublasNDArray(data,ArrayUtil.empty(),ArrayUtil.empty(),offset + slice * stride[0]);


            //slice of a matrix is a vector
        else if (shape.length == 2) {
            JCublasNDArray slice2 =  new JCublasNDArray(
                    data,
                    ArrayUtil.of(shape[1]),
                    Arrays.copyOfRange(stride,1,stride.length),
                    offset + slice * stride[0]
            );
            return slice2;

        }

        else
            return new JCublasNDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset + (slice * stride[0]));

    }


    /**
     * Returns the slice of this from the specified dimension
     * @param slice the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    public JCublasNDArray slice(int slice, int dimension) {
        if (slice == 0)
            return slice(dimension);
        if (shape.length == 2) {
            if (slice != 1)
                throw new IllegalArgumentException("Unable to retrieve dimension " + slice + " from a 2d array");
            return new JCublasNDArray(data,
                    ArrayUtil.of(shape[0]),
                    ArrayUtil.of(stride[0]),
                    offset + dimension * stride[1]
            );
        }

        return new JCublasNDArray (
                data,
                ArrayUtil.removeIndex(shape,dimension),
                ArrayUtil.removeIndex(stride,dimension),
                offset + dimension * stride[slice]
        );
    }
    public JCublasNDArray get(int[] indices) {
        JCublasNDArray result = new JCublasNDArray(data,new int[]{1,indices.length},stride,offset);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, getScalar(indices[i]));
        }

        return result;
    }

    public JCublasNDArray getScalar(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return JCublasNDArray.scalar(data[idx]);
    }

    @Override
    public JCublasNDArray put(int[] indices, INDArray element) {
        if(!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        int ix = offset;
        if (indices.length != shape.length)
            throw new IllegalArgumentException("Unable to applyTransformToDestination values: number of indices must be equal to the shape");

        for (int i = 0; i< shape.length; i++)
            ix += indices[i] * stride[i];


        data[ix] = (double) element.element();
        return this;

    }

    @Override
    public JCublasNDArray put(int i, int j, INDArray element) {
        {
            return put(new int[]{i, j}, element);
        }
    }

    //@Override
    public JCublasNDArray put(int i, INDArray element) {
            if(element == null)
                throw new IllegalArgumentException("Unable to insert null element");
            assert element.isScalar() : "Unable to insert non scalar element";

            put(i,(double) element.element());
            return this;
    }

    public int linearIndex(int i) {
        int realStride = getRealStrideForLinearIndex();
        int idx = offset + i * realStride;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }

    private int getRealStrideForLinearIndex() {
        if(stride == null || stride().length < 1)
            return 1;
        if(stride.length == 2 && shape[0] == 1)
            return stride[1];
        if(stride().length == 2 && shape[1] == 1)
            return stride[0];
        return stride[0];
    }

    public static JCublasNDArray scalar(JCublasNDArray from,int index) {
        return new JCublasNDArray(from.data,new int[]{1},new int[]{1},index);
    }

    public JCublasNDArray addi(INDArray other) {
        return addi(other,this);
    }

    /**
     * in place addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public JCublasNDArray addi(INDArray other, INDArray result) {
        new TwoArrayOps().from(this).other(other).op(AddOp.class)
                .to(result).build().exec();
        return (JCublasNDArray) result;
    }

    public static JCublasNDArray scalar(double num) {
        return new JCublasNDArray(new double[]{num},new int[]{1},new int[]{1},0);
    }

    public JCublasNDArray muliRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).muli(rowVector.getScalar(i));
        }
        return this;
    }

    @Override
    public JCublasNDArray getRow(int r) {
        if(shape.length == 2)
            return new JCublasNDArray(
                    data,
                    new int[]{shape[1]},
                    new int[]{stride[1]},
                    offset +  r * columns()
            );
        else
            throw new IllegalArgumentException("Unable to getFromOrigin row of non 2d matrix");
    }

    @Override
    public JCublasNDArray muliColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).muli(columnVector.getScalar(i));
        }
        return this;
    }

    @Override
    public JCublasNDArray mulColumnVector(INDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

    public JCublasNDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    public JCublasNDArray add(JCublasNDArray other) {
        return addi(other, new JCublasNDArray(rows, columns));
    }
    public JCublasNDArray addiRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).addi(rowVector.getScalar(i));
        }
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public JCublasNDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }

    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     * @param shape
     * @return
     */
    public JCublasNDArray reshape(int[] shape) {
        long ec = 1;
        for (int i = 0; i < shape.length; i++) {
            int si = shape[i];
            if (( ec * si ) != (((int) ec ) * si ))
                throw new IllegalArgumentException("Too many elements");
            ec *= shape[i];
        }
        int n = (int) ec;

        if (ec != n)
            throw new IllegalArgumentException("Too many elements");

        JCublasNDArray ndArray = new JCublasNDArray(data,shape,stride,offset);
        return ndArray;

    }

    public boolean multipliesWith(JCublasNDArray a) {
        return columns == a.rows;
    }


    /** Throws SizeException unless matrices can be multiplied with one another. */
    public void assertMultipliesWith(INDArray a) {
        if (!multipliesWith(a)) {
            throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
        }
    }
    public boolean sameSize(JCublasNDArray a) {
        return rows == a.rows && columns == a.columns;
    }
    /** Resize the matrix. All elements will be set to zero. */
    public void resize(int newRows, int newColumns) {
        rows = newRows;
        columns = newColumns;
        length = newRows * newColumns;
        data = new double[rows * columns];
    }
    /** Copy DoubleMatrix a to this. this a is resized if necessary. */
    public JCublasNDArray copy(JCublasNDArray a) {
        if (!sameSize(a)) {
            resize(a.rows, a.columns);
        }

        System.arraycopy(a.data, 0, data, 0, length);
        return a;
    }

    public double scalar() {
        return get(0);
    }

    public double get(int i) {
        int idx = linearIndex(i);
        return data[idx];
    }


    public double get(int i,int j) {
        if(isColumnVector()) {
            if(j > 0)
                throw new IllegalArgumentException("Trying to access column > " + columns() + " at " + j);
            return get(i);

        }
        else if(isRowVector()) {
            if(i > 0)
                throw new IllegalArgumentException("Trying to access row > " + rows() + " at " + i);
            return get(j);
        }


        return (double) get(new int[]{i,j}).element();

    }

    @Override
    public JCublasNDArray mmul(INDArray other) {
        int[] shape = {rows(),other.columns()};
        return mmuli(other,NDArrays.create(shape));
    }

    @Override
    public JCublasNDArray mmul(INDArray other, INDArray result) {
        return dup().mmuli(other,result);
    }

    @Override
    public JCublasNDArray mmul(JCublasNDArray other) {
        int[] shape = {rows(),other.columns()};
        return mmuli(other,NDArrays.create(shape));
    }



    public JCublasNDArray muli(INDArray other) {
        return muli(other,this);
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public JCublasNDArray muli(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(MultiplyOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(MultiplyOp.class)
                    .to(result).build().exec();
        return (JCublasNDArray) result;
    }

    //@Override
    public JCublasNDArray muli(double v, JCublasNDArray result) {
        new TwoArrayOps().from(this).to(this).scalar(JCublasNDArray.scalar(v))
                .op(MultiplyOp.class).build().exec();

        return this;
    }

    /** Matrix-matrix multiplication (in-place). */
    public JCublasNDArray mmuli(INDArray other, INDArray result) {
        JCublasNDArray otherArray = JCublasNDArray.wrap(other);
        JCublasNDArray resultArray = JCublasNDArray.wrap(result);

        if (other.isScalar()) {
            return muli(otherArray.scalar(), resultArray);
        }
        if (isScalar()) {
            return otherArray.muli(scalar(), resultArray);
        }

        /* check sizes and resize if necessary */
        assertMultipliesWith(other);


        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            JCublasNDArray temp = new JCublasNDArray(resultArray.shape(),ArrayUtil.calcStridesFortran(resultArray.shape()));

            if (otherArray.columns() == 1) {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();

                JCublas.cublasSetVector(
                        otherArray.length(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_A,
                        1);
                JCublas.cublasSetVector(
                        length(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_B,
                        1);

                JCublas.cublasDgemv(
                        'n',
                        otherArray.rows(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_A,
                        1,
                        1,
                        d_B,
                        1);

                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);

                //NDArrayBlas.gemv(1.0, this, otherArray, 0.0, temp);
            } else {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();
                Pointer d_C = new Pointer();

                JCublas.cublasSetMatrix(
                        columns(),
                        rows(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_A,
                        1
                        );
                JCublas.cublasSetMatrix(
                        other.columns(),
                        other.rows(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_B,
                        1
                );
                JCublas.cublasSgemm(
                        'n',
                        'n',
                        otherArray.rows(),
                        columns(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_B,
                        1,
                        0,
                        d_C,
                        1);
                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);
                //NDArrayBlas.gemm(1.0, this, otherArray, 0.0, temp);
            }

            JCublasNDArray.copy(temp, resultArray);


        } else {
            if (otherArray.columns() == 1) {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();

                JCublas.cublasSetVector(
                        otherArray.length(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_A,
                        1);
                JCublas.cublasSetVector(
                        length(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_B,
                        1);

                JCublas.cublasDgemv(
                        'n',
                        otherArray.rows(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_A,
                        1,
                        1,
                        d_B,
                        1);

                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);

                //NDArrayBlas.gemv(1.0, this, otherArray, 0.0, resultArray);
            }
            else {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();
                Pointer d_C = new Pointer();

                JCublas.cublasSetMatrix(
                        columns(),
                        rows(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_A,
                        1
                );
                JCublas.cublasSetMatrix(
                        other.columns(),
                        other.rows(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_B,
                        1
                );
                JCublas.cublasSgemm(
                        'n',
                        'n',
                        otherArray.rows(),
                        columns(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_B,
                        1,
                        0,
                        d_C,
                        1);
                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);
                //NDArrayBlas.gemm(1.0, this, otherArray, 0.0, resultArray);
            }
        }
        return resultArray;
    }
    //public JCublasNDArray mmuli(JCublasNDArray other) {return dup().mmuli(other,this); }
    public JCublasNDArray mmuli( INDArray result) {
        return JCublasNDArray.wrap(mmuli(result));
    }


    public JCublasNDArray mul(JCublasNDArray other, JCublasNDArray result) {return dup().muli(other,result);}
    public JCublasNDArray mul(INDArray other, INDArray result) {
        return dup().muli(other, result);
    }

    @Override
    public JCublasNDArray sub(INDArray other) {
        return dup().subi(other);
    }

    @Override
    public JCublasNDArray sub(INDArray other, INDArray result) {
        return dup().subi(other,result);
    }


    @Override
    public double[] data() {
        if(offset == 0)
            return data;

        INDArray linear = reshape(new int[]{1,length});
        double[] data = new double[length];
        int count = 0;
        for(int i = 0; i < length; i++) {
            data[count++] = (double) linear.getScalar(i).element();
        }
        return data;
    }

    public Object element() {
        if(!isScalar())
            throw new IllegalStateException("Unable to retrieve element from non scalar matrix");
        return data[0];
    }
    /**
     * Wrap toWrap with the specified shape, and dimensions from
     * the passed in ndArray
     * @param ndArray the way to wrap a matrix
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static JCublasNDArray wrap(JCublasNDArray ndArray,JCublasNDArray toWrap) {
        if(toWrap instanceof JCublasNDArray)
            return (JCublasNDArray) toWrap;
        int[] stride = ndArray.stride();
        JCublasNDArray ret = new JCublasNDArray(toWrap.data,ndArray.shape(),stride,ndArray.offset());
        return ret;
    }


    /**
     * Returns the shape(dimensions) of this array
     * @return the shape of this matrix
     */
    public int[] shape() {
        return shape;
    }

    /**
     * Wrap a matrix in to an ndarray
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static JCublasNDArray wrap(JCublasNDArray toWrap) {
        if(toWrap instanceof JCublasNDArray)
            return (JCublasNDArray) toWrap;
        int[]  shape = new int[]{toWrap.rows,toWrap.columns};
        JCublasNDArray ret = new JCublasNDArray(toWrap.data,shape);
        return ret;
    }

    @Override
    public JCublasNDArray div(INDArray other) {
        return dup().divi(other);
    }

    /**
     * copy (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public JCublasNDArray div(INDArray other, INDArray result) {
        return dup().divi(other,result);
    }

    public JCublasNDArray divi(INDArray other) {
        return divi(other,this);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public JCublasNDArray divi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(DivideOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(DivideOp.class)
                    .to(result).build().exec();
        return (JCublasNDArray) result;
    }

    public float[] floatData() {
        float[] ret = new float[data.length];
        for(int i = 0; i < ret.length ;i++)
            ret[i] = (float) data[i];
        return ret;
    }

    public JCublasNDArray diviRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).divi(rowVector.getScalar(i));
        }
        return this;
    }

    @Override
    public JCublasNDArray divRowVector(INDArray rowVector) {
        return dup().diviRowVector(rowVector);
    }

    public JCublasNDArray add(INDArray other) {
        return dup().addi(other);
    }

    @Override
    public INDArray add(INDArray other, INDArray result) {
        return dup().addi(other,result);
    }

    public JCublasNDArray mul(INDArray other) {
        return dup().muli(other);
    }

    public int rows() {
        if(isMatrix()) {
            if (shape().length > 2)
                return Shape.squeeze(shape)[0];
            else if (shape().length == 2)
                return shape[0];
        }
        else if(isVector()) {
            if(isRowVector())
                return 1;
            else
                return shape[0];
        }
        throw new IllegalStateException("Unable to getFromOrigin number of of rows for a non 2d matrix");
    }

    public boolean isRowVector() {
        if(shape().length == 1)
            return true;

        if(isVector())
            return shape()[0] == 1;

        return false;
    }

    /**
     * Checks whether the matrix is a column vector.
     */
    public boolean isColumnVector() {
        if(shape().length == 1)
            return false;

        if(isVector())
            return shape()[1] == 1;

        return false;

    }

    public JCublasNDArray normmax(int dimension) {
        if(isVector()) {
            return JCublasNDArray.scalar(normmax());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final JCublasNDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (JCublasNDArray) nd.getResult();
                    arr.put(i.get(),arr2.normmax(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */

                public void operate(JCublasNDArray nd) {
                    JCublasNDArray arr2 = nd;
                    arr.put(i.get(),arr2.normmax(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }

    @Override
    public INDArray norm2(int dimension) {
        if(isVector()) {
            return JCublasNDArray.scalar(norm2());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.norm2(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.norm2(0));
                    i.incrementAndGet();

                }
            }, false);

            return arr.reshape(shape);
        }
    }

    @Override
    public INDArray norm1(int dimension) {
        if(isVector()) {
            return JCublasNDArray.scalar(norm1());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.norm1(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.norm1(0));
                    i.incrementAndGet();

                }
            }, false);

            return arr.reshape(shape);
        }
    }

    @Override
    public INDArray prod(int dimension) {

        if(dimension == Integer.MAX_VALUE) {
            return NDArray.scalar(reshape(new int[]{1,length}).prod());
        }

        else if(isVector()) {
            return NDArray.scalar(prod());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                    arr.put(i.get(),arr2.prod(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.prod(0));
                    i.incrementAndGet();

                }
            }, false);

            return arr.reshape(shape);
        }
    }

    public double normmax() {
        double max = 0.0;
        for (int i = 0; i < length; i++) {
            double a = Math.abs(get(i));
            if (a > max) {
                max = a;
            }
        }
        return max;
    }
    /*
    public double normmax() {
        if(isVector())
            return super.normmax();
        return (double) JCublasNDArray.doSliceWise(NDArrayUtil.ScalarOp.NORM_MAX,this).element();

    }
    */

    //@Override
    public JCublasNDArray put(int i, int j, JCublasNDArray element) {
        return put(new int[]{i,j},element);
    }

    //@Override
    public JCublasNDArray muliColumnVector(JCublasNDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).muli(columnVector.getScalar(i));
        }
        return this;
    }

    //@Override
    public JCublasNDArray put(int[] indices, JCublasNDArray element) {
        if(!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        int ix = offset;
        if (indices.length != shape.length)
            throw new IllegalArgumentException("Unable to applyTransformToDestination values: number of indices must be equal to the shape");

        for (int i = 0; i< shape.length; i++)
            ix += indices[i] * stride[i];


        data[ix] = (double) element.element();
        return this;

    }
    public JCublasNDArray broadcast(int[] shape) {
        return null;
    }
    @Override
    public JCublasNDArray broadcasti(int[] shape) {
        return null;
    }
    @Override
    public JCublasNDArray repmat(int[] shape) {
        return null;
    }

    @Override
    public JCublasNDArray putRow(int row, INDArray toPut) {
        JCublasNDArray put = (JCublasNDArray) toPut;
        putRow(row,put);
        return this;
    }

    @Override
    public JCublasNDArray putColumn(int column, INDArray toPut) {
        JCublasNDArray put = (JCublasNDArray) toPut;
        putColumn(column, put);
        return this;
    }

    @Override
    public JCublasNDArray transpose() {
        if(isRowVector())
            return new JCublasNDArray(data,new int[]{shape[0],1},offset);
        else if(isColumnVector())
            return new JCublasNDArray(data,new int[]{shape[0]},offset);
        JCublasNDArray n = new JCublasNDArray(data,reverseCopy(shape),reverseCopy(stride),offset);
        return n;

    }

    public boolean isScalar() {
        if(shape.length == 0)
            return true;
        else if(shape.length == 1 && shape[0] == 1)
            return true;
        else if(shape.length >= 2) {
            for(int i = 0; i < shape.length; i++)
                if(shape[i] != 1)
                    return false;
        }

        return length == 1;
    }

    public JCublasNDArray getScalar(int row, int column) {
        return get(new int[]{row,column});
    }

    public int size(int dimension) {
        if(isScalar()) {
            if(dimension == 0)
                return length;
            else
                throw new IllegalArgumentException("Illegal dimension for scalar " + dimension);
        }

        else if(isVector()) {
            if(dimension == 0)
                return length;
            else if(dimension == 1)
                return 1;
        }

        return shape[dimension];
    }


    public JCublasNDArray mmul(JCublasNDArray other, JCublasNDArray result) {
        return dup().mmuli(other,result);
    }

    /**
     * Iterate over every row of every slice
     * @param op the operation to apply
     */
    public void iterateOverAllRows(SliceOp op) {
        if(isVector())
            op.operate(new DimensionSlice(false,this,null));
        else if(isMatrix()) {
            for(int i = 0; i < rows(); i++) {
                op.operate(new DimensionSlice(false,getRow(i),null));
            }
        }

        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).iterateOverAllRows(op);
            }
        }
    }

    @Override
    public INDArray getScalar(int... indexes) {
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return JCublasNDArray.scalar(data[ix]);
    }

    @Override
    public void checkDimensions(INDArray other) {
        assert Arrays.equals(shape,other.shape()) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape());
        assert Arrays.equals(stride,other.stride()) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride());
        assert offset == other.offset() : "Offset of this array is " + offset + " but other was " + other.offset();

    }

    public JCublasNDArray permute(int[] rearrange) {
        checkArrangeArray(rearrange);

        int[] newShape = doPermuteSwap(shape,rearrange);
        int[] newStride = doPermuteSwap(stride,rearrange);
        return new JCublasNDArray(data,newShape,newStride,offset);
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

    private int[] doPermuteSwap(int[] shape,int[] rearrange) {
        int[] ret = new int[shape.length];
        for(int i = 0; i < shape.length; i++) {
            ret[i] = shape[rearrange[i]];
        }
        return ret;
    }

    @Override
    public JCublasNDArray swapAxes(int dimension,int with) {
        int[] shape = ArrayUtil.range(0,shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
    }


    @Override
    public int slices() {
        if(shape.length < 1)
            return 0;
        return shape[0];
    }

    @Override
    public void iterateOverDimension(int dimension,SliceOp op,boolean modify) {
        if(dimension >= shape.length)
            throw new IllegalArgumentException("Unable to remove dimension  " + dimension + " was >= shape length");

        if(isScalar()) {
            if(dimension > 0)
                throw new IllegalArgumentException("Dimension must be 0 for a scalar");
            else {
                DimensionSlice slice = this.vectorForDimensionAndOffset(0,0);
                op.operate(slice);
                if(modify && slice.getIndices() != null) {
                    JCublasNDArray result = (JCublasNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = (double) result.getScalar(i).element();
                    }
                }
            }
        }

        else if(isVector()) {
            if(dimension == 0) {
                DimensionSlice slice = this.vectorForDimensionAndOffset(0,0);
                op.operate(slice);
                if(modify && slice.getIndices() != null) {
                    JCublasNDArray result = (JCublasNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = (double) result.getScalar(i).element();
                    }
                }
            }
            else if(dimension == 1) {
                for(int i = 0; i < length; i++) {
                    DimensionSlice slice = vectorForDimensionAndOffset(dimension,i);
                    op.operate(slice);
                    if(modify && slice.getIndices() != null) {
                        JCublasNDArray result = (JCublasNDArray) slice.getResult();
                        for(int j = 0; j < slice.getIndices().length; j++) {
                            data[slice.getIndices()[j]] = (double) result.getScalar(j).element();
                        }
                    }

                }
            }
            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);
        }


        else {

            int[] shape = ArrayUtil.removeIndex(this.shape,dimension);
            int[] stride = ArrayUtil.reverseCopy(ArrayUtil.removeIndex(this.stride,dimension));
            for(int currSlice = 0; currSlice < shape[0]; currSlice++) {
                JCublasNDArray ret = new JCublasNDArray(data,
                        shape,
                        stride,
                        currSlice);
                for(int j = 0; j < ret.slices(); j++) {
                    JCublasNDArray slice = ret.slice(j);
                    if(slice.isVector())
                        op.operate(slice);
                    else {
                        for(int i = 0; i < slice.slices(); i++) {
                            iterateOverDimension(dimension,op,modify);
                        }
                    }
                }
            }
        }
    }

    //getFromOrigin one result along one dimension based on the given offset
    public DimensionSlice vectorForDimensionAndOffset(int dimension, int offset) {
        if(isScalar() && dimension == 0 && offset == 0)
            return new DimensionSlice(false,getScalar(offset),new int[]{offset});


            //need whole vector
        else   if (isVector()) {
            if(dimension == 0) {
                int[] indices = new int[length];
                for(int i = 0; i < indices.length; i++)
                    indices[i] = i;
                return new DimensionSlice(false,dup(),indices);
            }
            else if(dimension == 1)
                return new DimensionSlice(false,getScalar(offset),new int[]{offset});
            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);

        }

        else {

            int count = 0;
            List<Integer> indices = new ArrayList<>();
            JCublasNDArray ret = new JCublasNDArray(new int[]{shape[dimension]});

            for(int j = offset; count < this.shape[dimension]; j+= this.stride[dimension]) {
                double d = data[j];
                ret.put(count++,d);
                indices.add(j);


            }

            return new DimensionSlice(false,ret,ArrayUtil.toArray(indices));

        }

    }

    //@Override
    public JCublasNDArray put(int i, double v) {
        if(!isVector() && !isScalar())
            throw new IllegalStateException("Unable to do linear indexing on a non vector");
        int idx = linearIndex(i);
        data[idx] = v;
        return this;
    }
    //@Override
    public JCublasNDArray mulColumnVector(JCublasNDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

}
