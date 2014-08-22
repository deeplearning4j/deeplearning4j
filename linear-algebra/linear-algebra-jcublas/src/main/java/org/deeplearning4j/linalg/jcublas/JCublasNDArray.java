package org.deeplearning4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.*;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.jcublas.util.JCublasNDArrayBlas;
import org.deeplearning4j.linalg.ops.TwoArrayOps;
import org.deeplearning4j.linalg.ops.elementwise.AddOp;
import org.deeplearning4j.linalg.ops.elementwise.DivideOp;
import org.deeplearning4j.linalg.ops.elementwise.MultiplyOp;
import org.deeplearning4j.linalg.ops.elementwise.SubtractOp;
import org.deeplearning4j.linalg.ops.reduceops.Ops;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.IterationResult;
import org.deeplearning4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.linalg.util.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.deeplearning4j.linalg.util.ArrayUtil.calcStrides;
import static org.deeplearning4j.linalg.util.ArrayUtil.reverseCopy;

public class JCublasNDArray implements INDArray {
    private int[] shape;
    private int[] stride;
    private int offset = 0;
    private boolean changedStride = false;
    private int[] oldStride;

    public int rows;
    /** Number of columns. */
    public int columns;
    /** Total number of elements (for convenience). */
    public int length;
    /** The actual data stored by rows (that is, row 0, row 1...). */
    public double[] data = null; // rows are contiguous

    public JCublasNDArray(float[] data, int[] shape, int[] stride, int offset) {
        this(ArrayUtil.doubleCopyOf(data),shape,stride,offset);

    }

    @Override
    public String toString() {
        if (isScalar()) {
            return element().toString();
        }
        else if(isVector()) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for(int i = 0; i < length; i++) {
                sb.append(getScalar(i));
                if(i < length - 1)
                    sb.append(',');
            }

            sb.append("]\n");
            return sb.toString();
        }



        StringBuilder sb = new StringBuilder();
        int length= shape[0];
        sb.append("[");
        if (length > 0) {
            sb.append(slice(0).toString());
            for (int i = 1; i < slices(); i++) {
                sb.append(slice(i).toString());
                if(i < length - 1)
                    sb.append(',');

            }
        }
        sb.append("]\n");
        return sb.toString();
    }

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
        initShape(new int[]{rows,columns});

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

    //getFromOrigin one result along one dimension based on the given offset
    private IterationResult op(int dimension, int offset, Ops.DimensionOp op,int currOffsetForSlice) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        boolean newSlice = false;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
            if(j >= currOffsetForSlice)
                newSlice = true;
        }

        return new IterationResult(reduceVector(op,new JCublasNDArray(dim)),newSlice);
    }

    //getFromOrigin one result along one dimension based on the given offset
    private double op(int dimension, int offset, Ops.DimensionOp op) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op,new JCublasNDArray(dim));
    }

    public JCublasNDArray(double[] newData) {
        this(newData.length);
        data = newData;
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

    //@Override
    public JCublasNDArray put(int row,int column,double value) {
        if (shape.length == 2)
            data[offset + row * stride[0]  + column * stride[1]] = value;

        else
            throw new UnsupportedOperationException("Invalid applyTransformToDestination for a non 2d array");
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

    @Override
    public void setStride(int[] stride) {
        this.oldStride = ArrayUtil.copy(this.stride);
        this.stride = stride;

    }

    @Override
    public INDArray subArray(int[] offsets, int[] shape, int[] stride) {
        return null;
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

    @Override
    public INDArray var(int dimension) {
        return null;
    }

    @Override
    public INDArray max(int dimension) {
        return null;
    }

    @Override
    public INDArray min(int dimension) {
        return null;
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

    public JCublasNDArray(List<INDArray> slices,int[] shape) {
        List<double[]> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++)
            list.add(slices.get(i).data());

        this.data = ArrayUtil.combine(list);

        initShape(shape);



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
        return (JCublasNDArray) getScalar(indices);
/*
        JCublasNDArray result = new JCublasNDArray(data,new int[]{1,indices.length},stride,offset);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, getScalar(indices[i]));
        }

        return result;
        */
    }

    public JCublasNDArray getScalar(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return JCublasNDArray.scalar(data[idx]);
    }

    @Override
    public double squaredDistance(INDArray other) {
        return 0;
    }

    @Override
    public double distance2(INDArray other) {
        return 0;
    }

    @Override
    public double distance1(INDArray other) {
        return 0;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, Number element) {
        return null;
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

    @Override
    public INDArray put(int i, int j, Number element) {
        return null;
    }

    //@Override
    public JCublasNDArray put(int i, INDArray element) {
            if(element == null)
                throw new IllegalArgumentException("Unable to insert null element");
            assert element.isScalar() : "Unable to insert non scalar element";

            put(i,(double) element.element());
            return this;
    }

    @Override
    public int vectorsAlongDimension(int dimension) {
        return length / size(dimension);
    }

    @Override
    public INDArray vectorAlongDimension(int index, int dimension) {
        assert dimension <= shape.length : "Invalid dimension " + dimension;
        if (shape.length == 2) {
            if (dimension == 1)
                return new JCublasNDArray(data,
                        new int[]{shape[dimension]}
                        , new int[]{stride[dimension]},
                        offset + index * stride[0]);
            else if (dimension == 0)
                return new JCublasNDArray(data,
                        new int[]{shape[dimension]}
                        , new int[]{stride[dimension]},
                        offset + index);
        }
        if(dimension == shape.length - 1)
            return new JCublasNDArray(data,
                    new int[]{1,shape[dimension]}
                    ,ArrayUtil.removeIndex(stride,0),
                    offset + index * stride[dimension - 1]);

        else if(dimension == 0)
            return new JCublasNDArray(data,
                    new int[]{shape[dimension],1}
                    ,new int[]{stride[dimension],1},
                    offset + index);

        return new JCublasNDArray(data,
                new int[]{shape[dimension],1}
                ,new int[]{stride[dimension],1},
                offset + index * stride[0]);

    }

    @Override
    public INDArray cumsumi(int dimension) {
        if(isVector()) {
            double s = 0.0;
            for (int i = 0; i < length; i++) {
                s += (double) getScalar(i).element();
                put(i, s);
            }
        }

        else if(dimension == Integer.MAX_VALUE || dimension == shape.length - 1) {
            INDArray flattened = ravel().dup();
            double prevVal = (double) flattened.getScalar(0).element();
            for(int i = 1; i < flattened.length(); i++) {
                double d = prevVal + (double) flattened.getScalar(i).element();
                flattened.putScalar(i,d);
                prevVal = d;
            }

            return flattened;
        }



        else {
            for(int i = 0; i < vectorsAlongDimension(dimension); i++) {
                INDArray vec = vectorAlongDimension(i,dimension);
                vec.cumsumi(0);

            }
        }


        return this;
    }

    @Override
    public INDArray cumsum(int dimension) {
        return dup().cumsumi(dimension);
    }

    @Override
    public INDArray assign(INDArray arr) {
        LinAlgExceptions.assertSameShape(this, arr);
        INDArray other = arr.ravel();
        INDArray thisArr = ravel();
        for(int i = 0; i < other.length(); i++)
            thisArr.put(i, other.getScalar(i));
        return this;
    }

    @Override
    public INDArray putScalar(int i, Number value) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] i, Number value) {
        return put(i,NDArrays.scalar(value));
    }

    @Override
    public INDArray lt(Number other) {
        return dup().lti(other);
    }

    @Override
    public INDArray lti(Number other) {
        return lti(NDArrays.scalar(other));
    }

    @Override
    public INDArray eq(Number other) {
        return dup().eqi(other);
    }

    @Override
    public INDArray eqi(Number other) {
        return eqi(NDArrays.scalar(other));
    }

    @Override
    public INDArray gt(Number other) {
        return dup().gti(other);
    }

    @Override
    public INDArray gti(Number other) {
        return gti(NDArrays.scalar(other));
    }

    @Override
    public INDArray lt(INDArray other) {
        return dup().lti(other);
    }

    @Override
    public INDArray lti(INDArray other) {
        return Transforms.lt(other);
    }

    @Override
    public INDArray eq(INDArray other) {
        return dup().eqi(other);
    }

    @Override
    public INDArray eqi(INDArray other) {
        return Transforms.eq(other);
    }

    @Override
    public INDArray gt(INDArray other) {
        return dup().gti(other);
    }

    @Override
    public INDArray gti(INDArray other) {
        return Transforms.gt(other);
    }

    @Override
    public INDArray neg() {
        return dup().negi();
    }

    @Override
    public INDArray negi() {
        return (JCublasNDArray) Transforms.neg(this);
    }

    @Override
    public INDArray rdiv(Number n) {
        return dup().rdivi(n);
    }

    @Override
    public INDArray rdivi(Number n) {
        return rdivi(NDArrays.valueArrayOf(shape(), n.doubleValue()));
    }

    @Override
    public INDArray rsub(Number n) {
        return dup().rsubi(n);
    }

    @Override
    public INDArray rsubi(Number n) {
        return rsubi(NDArrays.valueArrayOf(shape(),n.doubleValue()));
    }

    @Override
    public INDArray div(Number n) {
        return dup().divi(n);
    }

    @Override
    public INDArray divi(Number n) {
        return divi(NDArrays.scalar(n));
    }

    @Override
    public INDArray mul(Number n) {
        return dup().muli(n);
    }

    @Override
    public INDArray muli(Number n) {
        return muli(NDArrays.scalar(n));
    }

    @Override
    public INDArray sub(Number n) {
        return dup().subi(n);
    }

    @Override
    public INDArray subi(Number n) {
        return subi(NDArrays.scalar(n));
    }

    @Override
    public INDArray add(Number n) {
        return dup().addi(n);
    }

    @Override
    public INDArray addi(Number n) {
        return addi(NDArrays.scalar(n));
    }

    @Override
    public INDArray get(NDArrayIndex... indexes) {
        if(indexes.length < shape().length) {

        }

        return null;
    }

    @Override
    public JCublasNDArray getColumns(int[] cindices) {
        INDArray rows = NDArrays.create(rows(),cindices.length);
        for(int i = 0; i < cindices.length; i++) {
            rows.putColumn(i,getColumn(cindices[i]));
        }
        return (JCublasNDArray) rows;
    }

    @Override
    public INDArray getRows(int[] rindices) {
        INDArray rows = NDArrays.create(rindices.length,columns());
        for(int i = 0; i < rindices.length; i++) {
            rows.putRow(i,getRow(rindices[i]));
        }
        return (JCublasNDArray) rows;    }

    @Override
    public INDArray rdiv(INDArray other) {
        return dup().rdivi(other);
    }

    @Override
    public INDArray rdivi(INDArray other) {
        return rdivi(other,this);
    }

    @Override
    public INDArray rdiv(INDArray other, INDArray result) {
        return dup().rdivi(other,result);
    }

    @Override
    public INDArray rdivi(INDArray other, INDArray result) {
        return other.divi(this, result);
    }

    @Override
    public INDArray rsub(INDArray other, INDArray result) {
        return dup().rsubi(other,result);
    }

    @Override
    public INDArray rsub(INDArray other) {
        return dup().rsubi(other);
    }

    @Override
    public INDArray rsubi(INDArray other) {
        return rsubi(other,this);
    }

    @Override
    public INDArray rsubi(INDArray other, INDArray result) {
        return other.subi(this, result);
    }

    @Override
    public INDArray assign(Number value) {
        INDArray one = reshape(new int[]{1,length});
        for(int i = 0; i < one.length(); i++)
            one.put(i,NDArrays.scalar(value.doubleValue()));
        return one;
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
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(AddOp.class)
                    .to(result).build().exec();

        else
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
    public void assertMultipliesWith(JCublasNDArray a) {
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

    public static JCublasNDArray ones(int rows, int columns) {
        JCublasNDArray m = new JCublasNDArray(rows, columns);

        for (int i = 0; i < rows * columns; i++) {
            m.put(i, 1.0);
        }

        return m;
    }
    public static JCublasNDArray ones(int length) {
        return ones(length, 1);
    }

    public static JCublasNDArray linspace(int lower, int upper, int size) {
        JCublasNDArray result = new JCublasNDArray(size);
        for (int i = 0; i < size; i++) {
            double t = (double) i / (size - 1);
            result.put(i, lower * (1 - t) + t * upper);
        }
        return result;
    }

    public JCublasNDArray reshape(int newRows, int newColumns) {
        if (length != newRows * newColumns) {
            throw new IllegalArgumentException(
                    "Number of elements must not change.");
        }
        return reshape(new int[]{newRows,newColumns});
    }

    public double[] toArray() {
        double[] array = new double[length];

        System.arraycopy(data, 0, array, 0, length);

        return array;
    }
    public static JCublasNDArray zeros(int rows, int columns) {
        return new JCublasNDArray(rows, columns);
    }

    public int index(int rowIndex, int columnIndex) {
        return rowIndex + rows * columnIndex;
    }

    /** Creates a row vector of given length. */
    public static JCublasNDArray zeros(int length) {
        return zeros(length, 1);
    }
    public double get(int i,int j) {
        return data[index(i, j)];
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

    //@Override
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
        assertMultipliesWith(otherArray);

        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            if (otherArray.columns() == 1) {
                resultArray = JCublasNDArrayBlas.gemm(1.0f,this,otherArray,0.0,resultArray);
                //NDArrayBlas.gemv(1.0, this, otherArray, 0.0, temp);
            } else
            {
                resultArray = JCublasNDArrayBlas.gemm(1.0f,this,otherArray,0.0,resultArray);
                //NDArrayBlas.gemm(1.0, this, otherArray, 0.0, temp);
            }
        } else
        {
            if (otherArray.columns() == 1) {
                resultArray = JCublasNDArrayBlas.gemm(1.0f,this,otherArray,0.0,resultArray);
                //NDArrayBlas.gemv(1.0, this, otherArray, 0.0, resultArray);
            }
            else
            {
                resultArray = JCublasNDArrayBlas.gemm(1.0f,this,otherArray,0.0,resultArray);
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

    @Override
    public void setData(double[] data) {

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
    public static JCublasNDArray wrap(INDArray toWrap) {
        if(toWrap instanceof JCublasNDArray)
            return (JCublasNDArray) toWrap;
        int[]  shape = new int[]{toWrap.rows(),toWrap.columns()};
        JCublasNDArray ret = new JCublasNDArray(toWrap.data(),shape);
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

    @Override
    public void setData(float[] data) {

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

    public INDArray normmax(int dimension) {
        if(isVector()) {
            return JCublasNDArray.scalar(normmax());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
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

                public void operate(INDArray nd) {
                    INDArray arr2 = nd;
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

    public double std() {
        StandardDeviation dev = new StandardDeviation();
        double std = dev.evaluate(data());
        return std;
    }

    @Override
    public INDArray std(int dimension) {
        if(isVector()) {
            return JCublasNDArray.scalar(std());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.std(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.std(0));
                    i.incrementAndGet();

                }
            }, false);

            return arr.reshape(shape);
        }    }

    public static JCublasNDArray doSliceWise(MatrixOp op,JCublasNDArray arr) {
        int columns = isColumnOp(op) ? arr.columns() : arr.rows();
        int[] shape = {arr.slices(),columns};

        JCublasNDArray ret = new JCublasNDArray(shape);

        for(int i = 0; i < arr.slices(); i++) {
            switch(op) {
                case COLUMN_MIN:
                    ret.putSlice(i,arr.slice(i).columnMins());
                    break;
                case COLUMN_MAX:
                    ret.putSlice(i,arr.slice(i).columnMaxs());
                    break;
                case COLUMN_SUM:
                    ret.putSlice(i,arr.slice(i).columnSums());
                    break;
                case COLUMN_MEAN:
                    ret.putSlice(i,arr.slice(i).columnMeans());
                    break;
                case ROW_MIN:
                    ret.putSlice(i,arr.slice(i).rowMins());
                    break;
                case ROW_MAX:
                    ret.putSlice(i,arr.slice(i).rowMaxs());
                    break;
                case ROW_SUM:
                    ret.putSlice(i,arr.slice(i).rowSums());
                    break;
                case ROW_MEAN:
                    ret.putSlice(i,arr.slice(i).rowMeans());
                    break;
            }
        }


        return ret;
    }

    //@Override
    public double prod() {
        if(isVector() ) {
            double ret = 0.0;
            for(int i = 0; i < length; i++) {
                ret *= get(i);
            }
            return ret;
        }

        return (double) JCublasNDArray.doSliceWise(ScalarOp.PROD,this).element();

    }
    @Override
    public INDArray prod(int dimension) {

        if(dimension == Integer.MAX_VALUE) {
            return JCublasNDArray.scalar(reshape(new int[]{1,length}).prod());
        }

        else if(isVector()) {
            return JCublasNDArray.scalar(prod());
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

    public void putRow(int r, JCublasNDArray v) {
        JCublasNDArray n = JCublasNDArray.wrap(v);
        if(n.isVector() && n.length != columns())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");
        JCublasNDArray row = getRow(r);
        for(int i = 0; i < v.length; i++)
            row.put(i,v.get(i));


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

    public void putColumn(int c, JCublasNDArray v) {
        JCublasNDArray n = JCublasNDArray.wrap(v);
        if(n.isVector() && n.length != rows())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");
        JCublasNDArray column = getColumn(c);
        for(int i = 0; i < v.length; i++)
            column.put(i,v.get(i));


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

    public double[] getOffsetData() {
        return Arrays.copyOfRange(data, offset, offset+length);
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

    public static enum ScalarOp {
        SUM,
        MEAN,
        PROD,
        MAX,
        MIN,
        ARG_MAX,
        ARG_MIN,
        NORM_2,
        NORM_1,
        NORM_MAX
    }


    public static enum DimensionOp {
        SUM,
        MEAN,
        PROD,
        MAX,
        MIN,
        ARG_MIN,
        NORM_2,
        NORM_1,
        NORM_MAX,
        FFT
    }


    public static enum MatrixOp {
        COLUMN_MIN,
        COLUMN_MAX,
        COLUMN_SUM,
        COLUMN_MEAN,
        ROW_MIN,
        ROW_MAX,
        ROW_SUM,
        ROW_MEAN
    }

    private static boolean isRowOp(MatrixOp op) {
        return
                op == MatrixOp.ROW_MIN ||
                        op == MatrixOp.ROW_MAX ||
                        op == MatrixOp.ROW_SUM ||
                        op == MatrixOp.ROW_MEAN;
    }


    private static boolean isColumnOp(MatrixOp op) {
        return
                op == MatrixOp.COLUMN_MIN ||
                        op == MatrixOp.COLUMN_MAX ||
                        op == MatrixOp.COLUMN_SUM ||
                        op == MatrixOp.COLUMN_MEAN;
    }

    public JCublasNDArray supercolumnMins() {
        JCublasNDArray mins = new JCublasNDArray(1, columns);
        for (int c = 0; c < columns; c++) {
            mins.put(c, getColumn(c).min());
        }
        return mins;
    }

    //@Override
    public JCublasNDArray columnMins() {
        if(shape().length == 2) {
            return JCublasNDArray.wrap(supercolumnMins());

        }
        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.COLUMN_MIN,this);

    }

    public JCublasNDArray supercolumnSums() {
        if (rows == 1) {
            return dup();
        } else {
            JCublasNDArray v = new JCublasNDArray(1, columns);

            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    v.put(c, v.get(c) + get(r, c));
                }
            }

            return v;
        }
    }
    /**
     * Return a vector containing the sums of the columns (having number of columns many entries)
     */
    //@Override
    public JCublasNDArray columnSums() {
        if(shape().length == 2) {
            return JCublasNDArray.wrap(supercolumnSums());

        }
        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.COLUMN_SUM,this);

    }

    public JCublasNDArray supercolumnMaxs() {
        JCublasNDArray maxs = new JCublasNDArray(1, columns);
        for (int c = 0; c < columns; c++) {
            maxs.put(c, getColumn(c).max());
        }
        return maxs;
    }

    //@Override
    public JCublasNDArray columnMaxs() {
        if(shape().length == 2)
            return JCublasNDArray.wrap(supercolumnMaxs());

        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.COLUMN_MAX,this);

    }
    public static double asum(JCublasNDArray x) {
        Pointer d_A = new Pointer();
        JCublas.cublasSetVector(
                x.length(),
                Sizeof.DOUBLE,
                Pointer.to(x.data()),
                1,
                d_A,
                1);
        return JCublas.cublasDasum(x.length, d_A, 1);
    }
    public static double nrm2(JCublasNDArray x) {
        Pointer d_A = new Pointer();
        JCublas.cublasSetVector(
                x.length(),
                Sizeof.DOUBLE,
                Pointer.to(x.data()),
                1,
                d_A,
                1);
        return JCublas.cublasDnrm2(x.length, d_A, 1);
    }
    public static int iamax(JCublasNDArray x) {
        Pointer d_A = new Pointer();
        JCublas.cublasSetVector(
                x.length(),
                Sizeof.DOUBLE,
                Pointer.to(x.data()),
                1,
                d_A,
                1);
        return JCublas.cublasIdamax(x.length, d_A, 1) - 1;
    }

    //public static double asum(ComplexNDArray x) {
    //    return NativeBlas.dzasum(x.length, x.data, x.offset(), 1);
    //}


    public static INDArray doSliceWise(ScalarOp op,INDArray arr) {

        arr = arr.reshape(new int[]{1,arr.length()});

        JCublasNDArray cast = (JCublasNDArray) arr;

        if(op == ScalarOp.NORM_1) {
            return JCublasNDArray.scalar(JCublasNDArray.asum(cast));
        }

        else if(op == ScalarOp.NORM_2) {
            return JCublasNDArray.scalar(JCublasNDArray.nrm2(cast));

        }

        else if(op == ScalarOp.NORM_MAX) {
            int i = JCublasNDArray.iamax(cast);
            return arr.getScalar(i);
        }

        INDArray s = JCublasNDArray.scalar(0);
        for (int i = 0; i < arr.length(); i++) {
            switch (op) {
                case SUM:
                    s.addi(arr.getScalar(i));
                    break;
                case MEAN:
                    s.addi(arr.getScalar(i));
                    break;
                case MAX:
                    double curr = (double) arr.getScalar(i).element();
                    double sEle1 = (double) arr.getScalar(i).element();
                    if (curr > sEle1)
                        s = arr.getScalar(i);
                    break;
                case MIN:
                    double curr2 = (double) arr.getScalar(i).element();
                    double sEle2 = (double) arr.getScalar(i).element();
                    if (curr2 < sEle2)
                        s = arr.getScalar(i);

                    break;
                case PROD:
                    s.muli(arr.getScalar(i));
                    break;


            }


        }

        if(op == ScalarOp.MEAN)
            s.divi(JCublasNDArray.scalar(arr.length()));


        return s;
    }

    public JCublasNDArray supercolumnMeans() {
        return columnSums().divi(rows);
    }

    public JCublasNDArray columnMeans() {
        if(shape().length == 2) {
            return JCublasNDArray.wrap(supercolumnMeans());

        }

        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.COLUMN_MEAN,this);

    }

    public JCublasNDArray superrowMins() {
        JCublasNDArray mins = new JCublasNDArray(rows);
        for (int c = 0; c < rows; c++) {
            mins.put(c, getRow(c).min());
        }
        return mins;
    }
    public JCublasNDArray rowMins() {
        if(shape().length == 2) {
            return JCublasNDArray.wrap(superrowMins());

        }
        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.ROW_MIN,this);

    }
    public JCublasNDArray rowMaxs() {
        if(shape().length == 2) {
            return JCublasNDArray.wrap(superrowMaxs());

        }
        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.ROW_MAX,this);

    }
    public JCublasNDArray superrowMaxs() {
        JCublasNDArray maxs = new JCublasNDArray(rows);
        for (int c = 0; c < rows; c++) {
            maxs.put(c, getRow(c).max());
        }
        return maxs;
    }

    public JCublasNDArray rowSums() {
        if(shape().length == 2) {
            return JCublasNDArray.wrap(superrowSums());

        }

        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.ROW_SUM,this);
    }

    public JCublasNDArray superrowSums() {
        if (columns == 1) {
            return dup();
        } else {
            JCublasNDArray v = new JCublasNDArray(rows);

            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    v.put(r, v.get(r) + get(r, c));
                }
            }

            return v;
        }
    }
    public JCublasNDArray superrowMeans() {
        return rowSums().divi(columns);
    }

    public JCublasNDArray rowMeans() {
        if(shape().length == 2) {
            return JCublasNDArray.wrap(superrowMeans());

        }
        else
            return JCublasNDArray.doSliceWise(JCublasNDArray.MatrixOp.ROW_MEAN,this);

    }

    public JCublasNDArray divi(double v) {
        return divi(v, this);
    }

    /** Elementwise divide by a scalar. */
    public JCublasNDArray div(double v) {
        return divi(v, new JCublasNDArray(rows, columns));
    }
    public JCublasNDArray divi(double a, JCublasNDArray result) {
        ensureResultLength(null, result);

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) / a);
        }
        return result;
    }
    public boolean sameLength(JCublasNDArray a) {
        return length == a.length;
    }

    private void ensureResultLength(JCublasNDArray other, JCublasNDArray result) {
        if (!sameLength(result)) {
            if (result == this || result == other) {
                throw new SizeException("Cannot resize result matrix because it is used in-place.");
            }
            result.resize(rows, columns);
        }
    }

    public JCublasNDArray(int len) {
        this(len, 1, new double[len]);
    }

    @Override
    public boolean equals(Object o) {
        JCublasNDArray n = null;
        if(o instanceof  JCublasNDArray && !(o instanceof JCublasNDArray)) {
            JCublasNDArray d = (JCublasNDArray) o;
            //chance for comparison of the matrices if the shape of this matrix is 2
            if(shape().length > 2)
                return false;

            else
                n = JCublasNDArray.wrap(d);


        }
        else if(!o.getClass().isAssignableFrom(JCublasNDArray.class))
            return false;

        if(n == null)
            n = (JCublasNDArray) o;

        //epsilon equals
        if(isScalar() && n.isScalar()) {
            double val = (double) element();
            double val2 = (double) n.element();
            return Math.abs(val - val2) < 1e-6;
        }
        else if(isVector() && n.isVector()) {
            for(int i = 0; i < length; i++) {
                double curr = (double) getScalar(i).element();
                double comp = (double) n.getScalar(i).element();
                if(Math.abs(curr - comp) > 1e-6)
                    return false;
            }

            if(!Shape.shapeEquals(shape(),n.shape()))
                return false;

            return true;

        }


        if(!Shape.shapeEquals(shape(),n.shape()))
            return false;



        if(slices() != n.slices())
            return false;

        for (int i = 0; i < slices(); i++) {
            JCublasNDArray slice = slice(i);
            JCublasNDArray nSlice = n.slice(i);

            if (!slice.equals(nSlice))
                return false;
        }

        return true;


    }
}
