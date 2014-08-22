package org.deeplearning4j.linalg.jblas.complex;


import static  org.deeplearning4j.linalg.util.ArrayUtil.calcStrides;
import static  org.deeplearning4j.linalg.util.ArrayUtil.reverseCopy;

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.DimensionSlice;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.jblas.NDArray;
import org.deeplearning4j.linalg.jblas.util.ComplexNDArrayUtil;
import org.deeplearning4j.linalg.jblas.util.MatrixUtil;
import org.deeplearning4j.linalg.jblas.util.NDArrayBlas;
import org.deeplearning4j.linalg.ops.TwoArrayOps;
import org.deeplearning4j.linalg.ops.elementwise.AddOp;
import org.deeplearning4j.linalg.ops.elementwise.DivideOp;
import org.deeplearning4j.linalg.ops.elementwise.MultiplyOp;
import org.deeplearning4j.linalg.ops.elementwise.SubtractOp;
import org.deeplearning4j.linalg.ops.reduceops.Ops;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.linalg.util.*;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.NativeBlas;
import org.jblas.SimpleBlas;
import org.jblas.exceptions.SizeException;
import org.jblas.ranges.Range;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * ComplexNDArray for complex numbers.
 *
 *
 * Note that the indexing scheme for a complex ndarray is 2 * length
 * not length.
 *
 * The reason for this is the fact that imaginary components have
 * to be stored alongside realComponent components.
 *
 * @author Adam Gibson
 */
public class ComplexNDArray extends ComplexDoubleMatrix implements IComplexNDArray {
    private int[] shape;
    private int[] stride;
    private int offset = 0;



    /**
     * Create a new matrix with <i>newRows</i> rows, <i>newColumns</i> columns
     * using <i>newData></i> as the data. The length of the data is not checked!
     *
     * @param newRows
     * @param newColumns
     * @param newData
     */
    public ComplexNDArray(int newRows, int newColumns, double... newData) {
        super(newRows, newColumns, newData);
    }

    /**
     * Creates a new <tt>ComplexDoubleMatrix</tt> of size 0 times 0.
     */
    public ComplexNDArray() {
        this.length = 0;
        this.shape = new int[0];
        this.data = new double[0];

    }


    /** Construct a complex matrix from a realComponent matrix. */
    public ComplexNDArray(NDArray m) {
        this(m.shape());
        //NativeBlas.dcopy(m.length, m.data, m.offset(), 1, data, offset, 2);
        NDArray flattened = m.reshape(new int[]{1,m.length});
        ComplexNDArray flatten = reshape(1,length);
        for(int i = 0; i < length; i++) {
            flatten.put(i, flattened.getScalar(i));
        }
    }


    /** Construct a complex matrix from a realComponent matrix. */
    public ComplexNDArray(INDArray m) {
        this(m,ArrayUtil.calcStrides(m.shape()));
    }


    public ComplexNDArray(INDArray m,int[] stride) {
        this(m.shape(),stride);
        //NativeBlas.dcopy(m.length, m.data, m.offset(), 1, data, offset, 2);
        INDArray flattened = m.reshape(new int[]{1,m.length()});
        ComplexNDArray flatten = reshape(1,length);
        for(int i = 0; i < length; i++) {
            flatten.put(i, flattened.getScalar(i));
        }
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     * @param slices the slices of the ndarray
     * @param shape the final shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public ComplexNDArray(List<IComplexNDArray> slices,int[] shape,int[] stride) {
        super(new double[ArrayUtil.prod(shape) * 2]);
        List<ComplexDouble> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++) {
            IComplexNDArray flattened = slices.get(i).ravel();
            for(int j = 0; j < flattened.length(); j++)
                list.add((ComplexDouble) flattened.getScalar(j).element());
        }


        this.data = new double[ArrayUtil.prod(shape) * 2 ];
        this.stride = stride;
        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            data[count] = list.get(i).realComponent();
            data[count + 1] = list.get(i).imaginaryComponent();
            count += 2;
        }

        initShape(shape);



    }


    /**
     * Create an ndarray from the specified slices
     * and the given shape
     * @param slices the slices of the ndarray
     * @param shape the final shape of the ndarray
     */
    public ComplexNDArray(List<IComplexNDArray> slices,int[] shape) {
        this(slices,shape,ArrayUtil.calcStrides(shape,2));


    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     */
    public ComplexNDArray(IComplexNumber[] newData,int[] shape) {
        super(new double[ArrayUtil.prod(shape) * 2]);
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i,(ComplexDouble) newData[i].asDouble());

    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     */
    public ComplexNDArray(IComplexNumber[] newData,int[] shape,int[] stride) {
        super(new double[ArrayUtil.prod(shape) * 2]);
        this.stride = stride;
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i,(ComplexDouble) newData[i].asDouble());

    }


    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     */
    public ComplexNDArray(org.jblas.ComplexDouble[] newData,int[] shape) {
        super(new double[ArrayUtil.prod(shape) * 2]);
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i,newData[i]);

    }


    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     */
    public ComplexNDArray(ComplexDouble[] newData,int[] shape) {
        super(new double[ArrayUtil.prod(shape) * 2]);
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i,newData[i]);

    }

    public ComplexNDArray(double[] data,int[] shape,int[] stride) {
        this(data,shape,stride,0);
    }




    public ComplexNDArray(double[] data,int[] shape,int[] stride,int offset) {
        super(data);
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");

        this.stride = stride;
        initShape(shape);



        this.offset = offset;



        if(data != null  && data.length > 0)
            this.data = data;
    }



    public ComplexNDArray(double[] data,int[] shape) {
        this(data,shape,0);
    }

    public ComplexNDArray(double[] data,int[] shape,int offset) {
        this(data,shape,calcStrides(shape,2),offset);
    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public ComplexNDArray(int[] shape,int[] stride,int offset) {
        this(new double[ArrayUtil.prod(shape) * 2],shape,stride,offset);
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public ComplexNDArray(int[] shape,int[] stride){
        this(shape,stride,0);
    }

    public ComplexNDArray(int[] shape,int offset) {
        this(shape,calcStrides(shape,2),offset);
    }

    /**
     * Create with the specified shape
     * and an offset of 0
     * @param shape the shape of the ndarray
     */
    public ComplexNDArray(int[] shape) {
        this(shape,0);
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public ComplexNDArray(int newRows, int newColumns) {
        this(new int[]{newRows,newColumns});
    }



    public ComplexNDArray(float[] data, int[] shape, int[] stride, int offset) {
        this(ArrayUtil.doubleCopyOf(data),shape,stride,offset);
    }

    @Override
    public ComplexNDArray dup() {
        double[] dupData = new double[data.length];
        System.arraycopy(data,0,dupData,0,dupData.length);
        ComplexNDArray ret = new ComplexNDArray(dupData,shape,stride,offset);
        return ret;
    }



    /**
     * Returns the squared (Euclidean) distance.
     */
    public double squaredDistance(INDArray other) {
        double sd = 0.0;
        for (int i = 0; i < length; i++) {
            IComplexNumber diff = (IComplexNumber) getScalar(i).sub(other.getScalar(i)).element();
            double d = (double) diff.absoluteValue();
            sd += d * d;
        }
        return sd;
    }

    /**
     * Returns the (euclidean) distance.
     */
    public double distance2(INDArray other) {
        return  Math.sqrt(squaredDistance(other));
    }

    /**
     * Returns the (1-norm) distance.
     */
    public double distance1(INDArray other) {
        double d = 0.0;
        for (int i = 0; i < length; i++) {
            IComplexNumber n = (IComplexNumber) getScalar(i).sub(other.getScalar(i)).element();
            d += n.absoluteValue().doubleValue();
        }
        return d;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        return null;
    }


    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, int j, Number element) {
        return put(i,j,NDArrays.scalar(element));
    }


    @Override
    public ComplexNDArray put(int rowIndex, int columnIndex, org.jblas.ComplexDouble value) {
        int i =  index(rowIndex, columnIndex);
        data[i] = value.real();
        data[i+1] = value.imag();
        return this;
    }


    @Override
    public ComplexNDArray put(int row,int column,double value) {
        if (shape.length == 2)
            put(row,column,new ComplexDouble(value));

        else
            throw new UnsupportedOperationException("Invalid applyTransformToDestination for a non 2d array");
        return this;
    }

    @Override
    public int index(int row,int column) {
        if(!isMatrix())
            throw new IllegalStateException("Unable to getFromOrigin row/column from a non matrix");

        return row *  stride[0]  + column * stride[1];
    }

    private int[] nonOneStride() {
        int index = -1;
        for(int i = 0; i < shape().length; i++)
            if(shape()[i] == 1) {
                index = i;
                break;
            }

        return ArrayUtil.removeIndex(stride,index);
    }







    @Override
    public ComplexNDArray put(int i, org.jblas.ComplexDouble v) {
        if(i > length)
            throw new IllegalArgumentException("Unable to insert element " + v + " at index " + i + " with length " + length);
        int linearIndex = linearIndex(i);
        data[linearIndex] = v.real();
        data[linearIndex + 1] = v.imag();
        return this;
    }


    /**
     * Copy a column back into the matrix.
     *
     * @param c
     * @param v
     */
    @Override
    public void putColumn(int c, ComplexDoubleMatrix v) {
        ComplexNDArray n = ComplexNDArray.wrap(this,v);
        if(n.isVector() && n.length != rows())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");
        ComplexNDArray column = getColumn(c);

        for(int i = 0; i < v.length; i++)
            column.put(i,v.get(i));


    }
    /**
     *
     * Copy a row back into the matrix.
     *
     * @param r
     * @param v
     */
    @Override
    public void putRow(int r, ComplexDoubleMatrix v) {
        ComplexNDArray n = ComplexNDArray.wrap(v);
        if(!n.isVector())
            throw new IllegalArgumentException("Unable to insert matrix, wrong shape " + Arrays.toString(n.shape()));

        if(n.isVector() && n.length != columns())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");

        ComplexNDArray row = getRow(r);

        for(int i = 0; i < v.length; i++)
            row.put(i,v.get(i));


    }
    /**
     * Test whether a matrix is scalar.
     */
    @Override
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

    /**
     *
     * @param indexes
     * @param value
     * @return
     */
    @Override
    public ComplexNDArray put(int[] indexes, double value) {
        int ix = offset;
        if (indexes.length != shape.length)
            throw new IllegalArgumentException("Unable to applyTransformToDestination values: number of indices must be equal to the shape");

        for (int i = 0; i< shape.length; i++)
            ix += indexes[i] * stride[i];


        data[ix] = value;
        return this;
    }


    /**
     * Assigns the given matrix (put) to the specified slice
     * @param slice the slice to assign
     * @param put the slice to applyTransformToDestination
     * @return this for chainability
     */
    public ComplexNDArray putSlice(int slice,ComplexNDArray put) {
        if(isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            put(0,put.get(0));
            return this;
        }

        else if(isVector()) {
            assert put.isScalar() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            put(slice,put.get(0));
            return this;
        }


        assertSlice(put,slice);


        ComplexNDArray view = slice(slice);

        if(put.isScalar())
            put(slice,put.get(0));
        else if(put.isVector())
            for(int i = 0; i < put.length; i++)
                view.put(i,put.get(i));
        else if(put.shape().length == 2)
            for(int i = 0; i < put.rows(); i++)
                for(int j = 0; j < put.columns(); j++)
                    view.put(i,j,put.get(i,j));

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

        assert Shape.shapeEquals(sliceShape, requiredShape) : String.format("Invalid shape size of %s . Should have been %s ",Arrays.toString(sliceShape),Arrays.toString(requiredShape));

    }


    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     * @param dimension the dimension to swap
     * @param with the one to swap it with
     * @return the swapped axes view
     */
    public ComplexNDArray swapAxes(int dimension,int with) {
        int[] shape = ArrayUtil.range(0,shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
    }

    /**
     * Returns true if this ndarray is 2d
     * or 3d with a singleton element
     * @return true if the element is a matrix, false otherwise
     */
    public boolean isMatrix() {
        return shape().length == 2 ||
                shape.length == 3
                        && (shape[0] == 1 || shape[1] == 1 || shape[2] == 1);
    }


    /**
     * Return the first element of the matrix
     */
    @Override
    public ComplexDouble scalar() {
        return new ComplexDouble(super.scalar());
    }

    /**
     * Compute complex conj (in-place).
     */
    @Override
    public ComplexNDArray conji() {
        ComplexNDArray reshaped = reshape(1,length);
        ComplexDouble c = new ComplexDouble(0.0);
        for (int i = 0; i < length; i++)
            reshaped.put(i, reshaped.get(i, c).conji());
        return this;
    }

    @Override
    public ComplexNDArray hermitian() {
        ComplexNDArray result = new ComplexNDArray(shape());

        ComplexDouble c = new ComplexDouble(0);

        for (int i = 0; i < slices(); i++)
            for (int j = 0; j < columns; j++)
                result.put(j, i, get(i, j, c).conji());
        return result;
    }

    /**
     * Compute complex conj.
     */
    @Override
    public ComplexNDArray conj() {
        return dup().conji();
    }

    @Override
    public NDArray getReal() {
        int[] stride = ArrayUtil.copy(stride());
        for(int i = 0; i < stride.length; i++)
            stride[i] /= 2;
        NDArray result = new NDArray(shape(),stride);
        NativeBlas.dcopy(length, data, offset, 2, result.data, 0, 1);
        return result;
    }

    @Override
    public double getImag(int i) {
        return data[2 * i + offset + 1];
    }

    @Override
    public double getReal(int i) {
        return data[2 * i  + offset];
    }

    @Override
    public ComplexNDArray putReal(int rowIndex, int columnIndex, double value) {
        data[2*index(rowIndex, columnIndex) + offset] = value;
        return this;
    }

    @Override
    public ComplexNDArray putImag(int rowIndex, int columnIndex, double value) {
        data[2*index(rowIndex, columnIndex) + 1 + offset] = value;
        return this;
    }

    @Override
    public ComplexNDArray putReal(int i, double v) {
        return ComplexNDArray.wrap(this, super.putReal(i, v));
    }

    @Override
    public ComplexNDArray putImag(int i, double v) {
        data[2 * i + 1 + offset] = v;
        return this;
    }

    /**
     * Get realComponent part of the matrix.
     */
    @Override
    public NDArray real() {
        NDArray ret = new NDArray(shape());
        NativeBlas.dcopy(length, data, 0, 2, ret.data, 0, 1);
        return ret;
    }

    /**
     * Get imaginary part of the matrix.
     */
    @Override
    public NDArray imag() {
        NDArray ret = new NDArray(shape());
        NativeBlas.dcopy(length, data, 1, 2, ret.data, 0, 1);
        return ret;
    }

    //For fftn, it was found that it operated over every element
    //for some reason, when iterating over a "dimension"
    //. Apply every will pick n elements as a view
    //and run an operation. Confirm this behavior
    //tomorrow.


    public void applyEvery(int n,SliceOp op) {
        for(int i = 0; i < length; i += n) {
            op.operate(getNextN(i,n));
        }
    }


    public DimensionSlice getNextN(int from,int num) {
        List<Integer> list = new ArrayList<>();
        for(int i = from; i < from + num + 1; i++) {
            //realComponent and imaginary
            list.add(i);
            list.add(i + 1);
        }

        return new DimensionSlice(
                false,
                new ComplexNDArray(data,new int[]{2},new int[]{1}),
                ArrayUtil.toArray(list));
    }


    /**
     * Iterate along a dimension.
     * This encapsulates the process of sum, mean, and other processes
     * take when iterating over a dimension.
     * @param dimension the dimension to iterate over
     * @param op the operation to apply
     * @param modify whether to modify this array or not based on the results
     */
    public void iterateOverDimension(int dimension,SliceOp op,boolean modify) {
        if(isScalar()) {
            if(dimension > 1)
                throw new IllegalArgumentException("Dimension must be 0 for a scalar");
            else {
                DimensionSlice slice = this.vectorForDimensionAndOffset(0,0);
                op.operate(slice);

                if(modify && slice.getIndices() != null) {
                    ComplexNDArray result = (ComplexNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.get(i).realComponent();
                        data[slice.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                    }
                }
            }
        }



        else if(isVector()) {
            if(dimension == 0) {
                DimensionSlice slice = vectorForDimensionAndOffset(0,0);
                op.operate(slice);
                if(modify && slice.getIndices() != null) {
                    ComplexNDArray result = (ComplexNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.get(i).realComponent();
                        data[slice.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                    }
                }
            }
            else if(dimension == 1) {
                for(int i = 0; i < length; i++) {
                    DimensionSlice slice = vectorForDimensionAndOffset(dimension,i);
                    op.operate(slice);
                    if(modify && slice.getIndices() != null) {
                        ComplexNDArray result = (ComplexNDArray) slice.getResult();
                        for(int j = 0; j < slice.getIndices().length; j++) {
                            data[slice.getIndices()[j]] = result.get(j).realComponent();
                            data[slice.getIndices()[j] + 1] = result.get(j).imaginaryComponent();
                        }
                    }
                }
            }
            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);
        }

        else {
            if(dimension >= shape.length)
                throw new IllegalArgumentException("Unable to remove dimension  " + dimension + " was >= shape length");


            int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

            if(dimension == 0) {
                //iterating along the dimension is relative to the number of slices
                //in the return dimension
                int numTimes = ArrayUtil.prod(shape);
                //note difference here from ndarray, the offset is incremented by 2 every time
                //note also numtimes is multiplied by 2, this is due to the complex and imaginary components
                for(int offset = this.offset; offset < numTimes ; offset += 2) {
                    DimensionSlice vector = vectorForDimensionAndOffset(dimension,offset);
                    op.operate(vector);
                    if(modify && vector.getIndices() != null) {
                        ComplexNDArray result = (ComplexNDArray) vector.getResult();
                        for(int i = 0; i < vector.getIndices().length; i++) {
                            data[vector.getIndices()[i]] = result.get(i).realComponent();
                            data[vector.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                        }
                    }

                }

            }

            else {
                //needs to be 2 * shape: this is due to both realComponent and imaginary components
                double[] data2 = new double[ArrayUtil.prod(shape) ];
                int dataIter = 0;
                //want the milestone to slice[1] and beyond
                int[] sliceIndices = endsForSlices();
                int currOffset = 0;

                //iterating along the dimension is relative to the number of slices
                //in the return dimension
                //note here the  and +=2 this is for iterating over realComponent and imaginary components
                for(int offset = this.offset;;) {
                    if(dataIter >= data2.length || currOffset >= sliceIndices.length)
                        break;

                    //do the operation,, and look for whether it exceeded the current slice
                    DimensionSlice pair = vectorForDimensionAndOffsetPair(dimension, offset,sliceIndices[currOffset]);
                    //append the result
                    op.operate(pair);


                    if(modify && pair.getIndices() != null) {
                        ComplexNDArray result = (ComplexNDArray) pair.getResult();
                        for(int i = 0; i < pair.getIndices().length; i++) {
                            data[pair.getIndices()[i]] = result.get(i).realComponent();
                            data[pair.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                        }
                    }

                    //go to next slice and iterate over that
                    if(pair.isNextSlice()) {
                        //DO NOT CHANGE
                        currOffset++;
                        if(currOffset >= sliceIndices.length)
                            break;
                        //will update to next step
                        offset = sliceIndices[currOffset];
                    }

                }

            }


        }


    }



    //getFromOrigin one result along one dimension based on the given offset
    public DimensionSlice vectorForDimensionAndOffsetPair(int dimension, int offset,int currOffsetForSlice) {
        int count = 0;
        ComplexNDArray ret = new ComplexNDArray(new int[]{shape[dimension]});
        boolean newSlice = false;
        List<Integer> indices = new ArrayList<>();
        for(int j = offset; count < this.shape[dimension]; j+= this.stride[dimension] ) {
            ComplexDouble d = new ComplexDouble(data[j],data[j + 1]);
            indices.add(j);
            ret.put(count++,d);
            if(j >= currOffsetForSlice)
                newSlice = true;

        }

        return new DimensionSlice(newSlice,ret,ArrayUtil.toArray(indices));
    }


    //getFromOrigin one result along one dimension based on the given offset
    public DimensionSlice vectorForDimensionAndOffset(int dimension, int offset) {
        if(isScalar() && dimension == 0 && offset == 0)
            return new DimensionSlice(false,ComplexNDArray.scalar(get(offset)),new int[]{offset});


            //need whole vector
        else  if (isVector()) {
            if(dimension == 0) {
                int[] indices = new int[length];
                for(int i = 0; i < indices.length; i++)
                    indices[i] = i;
                return new DimensionSlice(false,dup(),indices);
            }

            else if(dimension == 1) {
                return new DimensionSlice(false,ComplexNDArray.scalar(get(offset)),new int[]{offset});
            }

            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);
        }


        else {
            int count = 0;
            ComplexNDArray ret = new ComplexNDArray(new int[]{shape[dimension]});
            List<Integer> indices = new ArrayList<>();
            for (int j = offset; count < this.shape[dimension]; j += this.stride[dimension] ) {
                ComplexDouble d = new ComplexDouble(data[j], data[j + 1]);
                ret.put(count++, d);
                indices.add(j);
            }

            return new DimensionSlice(false, ret, ArrayUtil.toArray(indices));
        }

    }




    //getFromOrigin one result along one dimension based on the given offset
    private ComplexIterationResult op(int dimension, int offset, Ops.DimensionOp op,int currOffsetForSlice) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        boolean newSlice = false;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
            if(j >= currOffsetForSlice)
                newSlice = true;
        }

        ComplexDoubleMatrix r = new ComplexDoubleMatrix(dim);
        ComplexNDArray wrapped = ComplexNDArray.wrap(r);
        ComplexDouble r2 = reduceVector(op,wrapped);
        return new ComplexIterationResult(newSlice,r2);
    }


    //getFromOrigin one result along one dimension based on the given offset
    private ComplexDouble op(int dimension, int offset, Ops.DimensionOp op) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op,ComplexNDArray.wrap(new ComplexDoubleMatrix(dim)));
    }



    private ComplexDouble reduceVector(Ops.DimensionOp op,ComplexNDArray vector) {

        switch(op) {
            case SUM:
                return (ComplexDouble) vector.sum(0).element();
            case MEAN:
                return (ComplexDouble) vector.mean(0).element();
            case NORM_1:
                return new ComplexDouble(vector.norm1());
            case NORM_2:
                return new ComplexDouble(vector.norm2());
            case NORM_MAX:
                return new ComplexDouble(vector.normmax());
            case FFT:
            default: throw new IllegalArgumentException("Illegal operation");
        }
    }

    public ComplexNDArray get(Object...o) {


        int[] shape = shapeFor(shape(),o,true);
        int[] indexingShape = shapeFor(shape(),o,false);
        int[] stride = calcStrides(shape);
        int[] query = queryForObject(shape(),o);

        if(query.length == 1)
            return ComplexNDArray.scalar(this,query[0]);



        //promising
        int index = offset + indexingShape[0] * stride[0];
        //int[] baseLineIndices = new int[]
        return new ComplexNDArray(data,
                shape,
                stride,
                index);
    }


    /**
     * Fetch a particular number on a multi dimensional scale.
     *
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    @Override
    public IComplexNDArray getScalar(int... indexes) {
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return NDArrays.scalar(NDArrays.createDouble(data[ix],data[ix + 1]));
    }

    /**
     * Validate dimensions are equal
     *
     * @param other the other ndarray to compare
     */
    @Override
    public void checkDimensions(INDArray other) {

    }

    /**
     * Gives the indices for the ending of each slice
     * @return the off sets for the beginning of each slice
     */
    public int[] endsForSlices() {
        int[] ret = new int[slices()];
        int currOffset = offset;
        for(int i = 0; i < slices(); i++) {
            ret[i] = (currOffset );
            currOffset += stride[0];
        }
        return ret;
    }

    /**
     * http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduce.html
     *
     * @param op        the operation to do
     * @param dimension the dimension to return from
     * @return the results of the reduce (applying the operation along the specified
     * dimension)t
     */
    @Override
    public IComplexNDArray reduce(Ops.DimensionOp op, int dimension) {
        if(isScalar())
            return this;


        if(isVector())
            return NDArrays.scalar(reduceVector(op, this));


        int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

        if(dimension == 0) {
            double[] data2 = new double[ArrayUtil.prod(shape) * 2];
            int dataIter = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for(int offset = this.offset; offset < numTimes; offset++) {
                ComplexDouble reduce = op(dimension, offset, op);
                data2[dataIter++] = reduce.real();
                data2[dataIter++] = reduce.imag();


            }

            return NDArrays.createComplex(data2,shape);
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
                ComplexIterationResult pair = op(dimension, offset, op,sliceIndices[currOffset]);
                //append the result
                IComplexNumber reduce = pair.getNumber();
                data2[dataIter++] = reduce.realComponent().doubleValue();
                data2[dataIter++] = reduce.imaginaryComponent().doubleValue();
                //go to next slice and iterate over that
                if(pair.isNextIteration()) {
                    //will update to next step
                    offset = sliceIndices[currOffset];
                    numTimes +=  sliceIndices[currOffset];
                    currOffset++;
                }

            }

            return NDArrays.createComplex(data2,shape);
        }

    }

    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to applyTransformToDestination
     * @return this for chainability
     */
    @Override
    public IComplexNDArray putSlice(int slice, INDArray put) {
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


        INDArray view = slice(slice);

        if(put.isScalar())
            put(slice,put.getScalar(0));
        else if(put.isVector())
            for(int i = 0; i < put.length(); i++)
                view.put(i,put.getScalar(i));
        else if(put.shape().length == 2)
            for(int i = 0; i < put.rows(); i++)
                for(int j = 0; j < put.columns(); j++) {
                    view.put(i,j,NDArrays.scalar((IComplexNumber) put.getScalar(i, j).element()));

                }

        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for(int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i,view.slice(i));

        }

        return this;
    }

    /**
     * Gives the indices for the beginning of each slice
     * @return the off sets for the beginning of each slice
     */
    public int[] offsetsForSlices() {
        int[] ret = new int[slices()];
        int currOffset = offset;
        for(int i = 0; i < slices(); i++) {
            ret[i] = currOffset;
            currOffset += stride[0] ;
        }
        return ret;
    }


    public ComplexNDArray subArray(int[] shape) {
        return subArray(offsetsForSlices(),shape);
    }


    /**
     * Number of slices: aka shape[0]
     * @return the number of slices
     * for this nd array
     */
    @Override
    public int slices() {
        if(shape.length < 1)
            return 0;
        return shape[0];
    }



    public ComplexNDArray subArray(int[] offsets, int[] shape,int[] stride) {
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

        return new ComplexNDArray(
                data
                , Arrays.copyOf(shape,shape.length)
                , stride
                ,offset + ArrayUtil.dotProduct(offsets, stride)
        );
    }




    public ComplexNDArray subArray(int[] offsets, int[] shape) {
        return subArray(offsets,shape,stride);
    }



    public static int[] queryForObject(int[] shape,Object[] o) {
        //allows us to put it in to shape format
        Object[] copy =  o;
        int[] ret = new int[copy.length];
        for(int i = 0; i < copy.length; i++) {
            //give us the whole thing
            if(copy[i] instanceof  Character &&   (char)copy[i] == ':')
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
            if((char)copy[i] == ':')
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
            if((char)copy[i] == ':')
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
            if((char)copy[i] == ':')
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
    public ComplexNDArray put(int i, double v) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");

        data[linearIndex(i)] = v;
        return this;
    }

    @Override
    public ComplexDouble get(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return new ComplexDouble(data[idx],data[idx + 1]);
    }


    /**
     * Linear getScalar ignoring linear restrictions
     * @param i the index of the element to getScalar
     * @return the item at the given index
     */
    public ComplexDouble unSafeGet(int i) {
        int idx = unSafeLinearIndex(i);
        return new ComplexDouble(data[idx],data[idx + 1]);
    }


    public int unSafeLinearIndex(int i) {
        int realStride = stride[0];
        int idx = offset + i;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }




    /**
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public IComplexNDArray put(int[] indices, INDArray element) {
        if(!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        int ix = offset;
        if (indices.length != shape.length)
            throw new IllegalArgumentException("Unable to applyTransformToDestination values: number of indices must be equal to the shape");

        for (int i = 0; i< shape.length; i++)
            ix += indices[i] * stride[i];

        if(element instanceof IComplexNDArray) {
            IComplexNumber element2 = (IComplexNumber) element.element();
            data[ix] = (double) element2.realComponent();
            data[ix + 1]= (double) element2.imaginaryComponent();
        }
        else {
            double element2 = (double) element.element();
            data[ix] = element2;
            data[ix + 1]= 0;
        }

        return this;

    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public IComplexNDArray put(int i, int j, INDArray element) {
        return put(new int[]{i,j},element);
    }


    @Override
    public int linearIndex(int i) {
        int realStride = getRealStrideForLinearIndex();
        int idx = offset + i * realStride;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }

    private int getRealStrideForLinearIndex() {
        if(stride.length != shape.length)
            throw new IllegalStateException("Stride and shape not equal length");
        if(shape.length == 1)
            return stride[0];
        if(shape.length == 2) {
            if(shape[0] == 1)
                return stride[1];
            if(shape[1] == 1)
                return stride[0];
        }
        return stride[0];
    }



    /**
     * Returns the specified slice of this matrix.
     * In matlab, this would be equivalent to (given a 2 x 2 x 2):
     * A(x,:,:) where x is the slice you want to return.
     *
     * The slice is always relative to the final dimension of the matrix.
     *
     * @param dimension the slice to return
     * @return the specified slice of this matrix
     */
    public ComplexNDArray dim(int dimension) {
        int[] shape = ArrayUtil.copy(shape());
        int[] stride = ArrayUtil.reverseCopy(this.stride);
        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d ComplexNDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            return new ComplexNDArray(data,new int[]{},new int[]{},offset + dimension * stride[0]);

            //slice of a matrix is a vector
        else if (shape.length == 2) {
            int st = stride[0];
            if (st == 1) {
                return new ComplexNDArray(
                        data,
                        ArrayUtil.of(shape[1]),
                        ArrayUtil.of(1),
                        offset + dimension * stride[0]);
            }

            else {

                return new ComplexNDArray(
                        data,
                        ArrayUtil.of(shape[1]),
                        ArrayUtil.of(stride[1]),
                        offset + dimension * stride[0]
                );
            }
        }

        else {
            return new ComplexNDArray(data,
                    shape,
                    stride,
                    offset + dimension * stride[0]);
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
    public ComplexNDArray slice(int slice) {
        int offset = this.offset + (slice * stride[0]  );
        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d ComplexNDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            return new ComplexNDArray(
                    data,
                    ArrayUtil.empty(),
                    ArrayUtil.empty(),
                    offset);


            //slice of a matrix is a vector
        else if (shape.length == 2) {
            return new ComplexNDArray(
                    data,
                    ArrayUtil.of(shape[1]),
                    Arrays.copyOfRange(stride,1,stride.length),
                    offset

            );

        }

        else {
            if(offset >= data.length)
                throw new IllegalArgumentException("Offset index is > data.length");
            return new ComplexNDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset);
        }
    }


    /**
     * Returns the slice of this from the specified dimension
     * @param slice the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    public ComplexNDArray slice(int slice, int dimension) {
        int offset = this.offset + dimension * stride[slice];
        if(this.offset == 0)
            offset *= 2;

        if (shape.length == 2) {
            int st = stride[1];
            if (st == 1) {
                return new ComplexNDArray(
                        data,
                        new int[]{shape[dimension]},
                        offset);
            } else {
                return new ComplexNDArray(
                        data,
                        new int[]{shape[dimension]},
                        new int[]{st},
                        offset);
            }


        }

        if (slice == 0)
            return slice(dimension);


        return new ComplexNDArray (
                data,
                ArrayUtil.removeIndex(shape,dimension),
                ArrayUtil.removeIndex(stride,dimension),
                offset
        );
    }



    /**
     * Fetch a particular number on a multi dimensional scale.
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    public double getMulti(int... indexes) {
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return data[ix];
    }


    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    public IComplexNDArray repmat(int[] shape) {
        int[] newShape = ArrayUtil.copy(shape());
        assert shape.length <= newShape.length : "Illegal shape: The passed in shape must be <= the current shape length";
        for(int i = 0; i < shape.length; i++)
            newShape[i] *= shape[i];
        IComplexNDArray result = NDArrays.createComplex(newShape);
        //nd copy
        if(isScalar()) {
            for(int i = 0; i < result.length(); i++) {
                result.put(i,getScalar(0));

            }
        }

        else if(isMatrix()) {

            for (int c = 0; c < shape()[1]; c++) {
                for (int r = 0; r < shape()[0]; r++) {
                    for (int i = 0; i < rows(); i++) {
                        for (int j = 0; j < columns(); j++) {
                            result.put(r * rows() + i, c * columns() + j, getScalar(i, j));
                        }
                    }
                }
            }

        }

        else {
            int[] sliceRepmat = ArrayUtil.removeIndex(shape,0);
            for(int i = 0; i < result.slices(); i++) {
                result.putSlice(i,repmat(sliceRepmat));
            }
        }

        return  result;
    }



    /**
     * Assign all of the elements in the given
     * ndarray to this nedarray
     *
     * @param arr the elements to assign
     * @return this
     */
    @Override
    public IComplexNDArray assign(IComplexNDArray arr) {
        LinAlgExceptions.assertSameShape(this, arr);
        INDArray other = arr.ravel();
        INDArray thisArr = ravel();
        for(int i = 0; i < other.length(); i++)
            thisArr.put(i,other.getScalar(i));
        return this;
    }
    /**
     * Get whole rows from the passed indices.
     *
     * @param rindices
     */
    @Override
    public ComplexNDArray getRows(int[] rindices) {
        INDArray rows = NDArrays.create(rindices.length,columns());
        for(int i = 0; i < rindices.length; i++) {
            rows.putRow(i,getRow(rindices[i]));
        }
        return (ComplexNDArray) rows;
    }


    public double var() {
        double mean = (double) mean(Integer.MAX_VALUE).element();
        return StatUtils.variance(data(), mean);
    }

    /**
     * Returns the overall variance of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray var(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArray.scalar(reshape(new int[]{1,length}).var());
        }
        else if(isVector()) {
            return NDArray.scalar(var());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.var(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.var(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }


    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNumber element) {
        return null;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, Number element) {
        return null;
    }

    @Override
    public IComplexNDArray putScalar(int i, IComplexNumber value) {
        return put(i, NDArrays.scalar(value));
    }

    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    @Override
    public int vectorsAlongDimension(int dimension) {
        return length / size(dimension);
    }


    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to get
     * @param dimension the dimension to get the vector from
     * @return the vector along a particular dimension
     */
    @Override
    public IComplexNDArray vectorAlongDimension(int index, int dimension) {
        return new ComplexNDArray(data,
                new int[]{shape[dimension]}
                ,new int[]{stride[dimension]},
                offset + index);
    }

    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public IComplexNDArray cumsumi(int dimension) {
        if(isVector()) {
            IComplexNumber s = NDArrays.createDouble(0,0);
            for (int i = 0; i < length; i++) {
                s .addi((IComplexNumber) getScalar(i).element());
                putScalar(i, s);
            }
        }

        else if(dimension == Integer.MAX_VALUE || dimension == shape.length - 1) {
            IComplexNDArray flattened = ravel().dup();
            IComplexNumber prevVal = (IComplexNumber) flattened.getScalar(0).element();
            for(int i = 1; i < flattened.length(); i++) {
                IComplexNumber d = prevVal.add((IComplexNumber) flattened.getScalar(i).element());
                flattened.putScalar(i,d);
                prevVal = d;
            }

            return flattened;
        }



        else {
            for(int i = 0; i < vectorsAlongDimension(dimension); i++) {
                IComplexNDArray vec = vectorAlongDimension(i,dimension);
                vec.cumsumi(0);

            }
        }


        return this;
    }

    /**
     * Cumulative sum along a dimension (in place)
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public IComplexNDArray cumsum(int dimension) {
        return dup().cumsumi(dimension);
    }

    /**
     * Assign all of the elements in the given
     * ndarray to this nedarray
     *
     * @param arr the elements to assign
     * @return this
     */
    @Override
    public INDArray assign(INDArray arr) {
        return null;
    }

    @Override
    public IComplexNDArray putScalar(int i, Number value) {
        return put(i, NDArrays.scalar(value));
    }

    @Override
    public INDArray putScalar(int[] i, Number value) {
        return null;
    }

    @Override
    public ComplexNDArray lt(Number other) {
        return dup().lti(other);
    }

    @Override
    public ComplexNDArray lti(Number other) {
        return lti(NDArrays.scalar(other));
    }

    @Override
    public ComplexNDArray eq(Number other) {
        return dup().eqi(other);
    }

    @Override
    public ComplexNDArray eqi(Number other) {
        return eqi(NDArrays.scalar(other));
    }

    @Override
    public ComplexNDArray gt(Number other) {
        return dup().gti(other);
    }

    @Override
    public ComplexNDArray gti(Number other) {
        return gti(NDArrays.scalar(other));
    }

    @Override
    public ComplexNDArray lt(INDArray other) {
        return dup().lti(other);
    }

    @Override
    public ComplexNDArray lti(INDArray other) {
        return (ComplexNDArray) Transforms.lt(other);
    }

    @Override
    public ComplexNDArray eq(INDArray other) {
        return dup().eqi(other);
    }

    @Override
    public ComplexNDArray eqi(INDArray other) {
        return (ComplexNDArray) Transforms.eq(other);
    }

    @Override
    public ComplexNDArray gt(INDArray other) {
        return dup().gti(other);
    }

    @Override
    public ComplexNDArray gti(INDArray other) {
        return (ComplexNDArray) Transforms.gt(other);
    }

    /**
     * Negate each element.
     */
    @Override
    public ComplexNDArray neg() {
        return dup().negi();
    }

    /**
     * Negate each element (in-place).
     */
    @Override
    public ComplexNDArray negi() {
        return (ComplexNDArray) Transforms.neg(this);
    }

    @Override
    public IComplexNDArray rdiv(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(Number n) {
        return null;
    }


    @Override
    public IComplexNDArray div(Number n) {
        return dup().divi(n);
    }

    @Override
    public IComplexNDArray divi(Number n) {
        return divi(NDArrays.scalar(n));
    }

    @Override
    public IComplexNDArray mul(Number n) {
        return dup().muli(n);
    }

    @Override
    public IComplexNDArray muli(Number n) {
        return muli(NDArrays.scalar(n));
    }

    @Override
    public IComplexNDArray sub(Number n) {
        return dup().subi(n);
    }

    @Override
    public IComplexNDArray subi(Number n) {
        return subi(NDArrays.scalar(n));
    }

    @Override
    public IComplexNDArray add(Number n) {
        return dup().addi(n);
    }

    @Override
    public IComplexNDArray addi(Number n) {
        return addi(NDArrays.scalar(n));
    }

    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    public IComplexNDArray get(NDArrayIndex... indexes) {
        return null;
    }

    /**
     * Get whole columns from the passed indices.
     *
     * @param cindices
     */
    @Override
    public ComplexNDArray getColumns(int[] cindices) {
        INDArray rows = NDArrays.create(rows(),cindices.length);
        for(int i = 0; i < cindices.length; i++) {
            rows.putColumn(i,getColumn(cindices[i]));
        }
        return (ComplexNDArray) rows;
    }


    /**
     * Insert a row in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param row   the row insert into
     * @param toPut the row to insert
     * @return this
     */
    @Override
    public IComplexNDArray putRow(int row, INDArray toPut) {
        ComplexNDArray n = (ComplexNDArray) toPut;
        ComplexDoubleMatrix n2 = n;
        putRow(row,n2);
        return this;
    }

    /**
     * Insert a column in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param column the column to insert
     * @param toPut  the array to put
     * @return this
     */
    @Override
    public IComplexNDArray putColumn(int column, INDArray toPut) {
        ComplexNDArray n = (ComplexNDArray) toPut;
        ComplexDoubleMatrix n2 = n;
        putColumn(column,n2);
        return this;
    }

    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    @Override
    public IComplexNDArray getScalar(int row, int column) {
        return ComplexNDArray.scalar(get(row,column));
    }

    /** Retrieve matrix element */
    @Override
    public ComplexDouble get(int rowIndex, int columnIndex) {
        int index = offset +  index(rowIndex,columnIndex);
        return new ComplexDouble(data[index],data[index + 1]);
    }



    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public IComplexNDArray getScalar(int i) {
        return ComplexNDArray.scalar(get(i));
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public IComplexNDArray put(int i, INDArray element) {
        if(element == null)
            throw new IllegalArgumentException("Unable to insert null element");
        assert element.isScalar() : "Unable to insert non scalar element";
        if(element instanceof  IComplexNDArray) {
            put(i,(org.jblas.ComplexDouble) element.element());
        }
        else
            put(i,(double) element.element());
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray diviColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).divi(columnVector.getScalar(i));
        }
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray diviRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).divi(rowVector.getScalar(i));
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
    public IComplexNDArray divRowVector(INDArray rowVector) {
        return dup().diviRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray muliColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).muli(columnVector.getScalar(i));
        }
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray mulColumnVector(INDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray muliRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).muli(rowVector.getScalar(i));
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
    public IComplexNDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray subiColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).subi(columnVector.getScalar(i));
        }
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray subColumnVector(INDArray columnVector) {
        return dup().subiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray subiRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).subi(rowVector.getScalar(i));
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
    public IComplexNDArray subRowVector(INDArray rowVector) {
        return dup().subiRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray addiColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).addi(columnVector.getScalar(i));
        }
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray addColumnVector(INDArray columnVector) {
        return dup().addiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray addiRowVector(INDArray rowVector) {
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
    public IComplexNDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public IComplexNDArray mmul(INDArray other) {
        int[] shape = {rows(),other.columns()};
        return mmuli(other,NDArrays.create(shape));
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public IComplexNDArray mmul(INDArray other, INDArray result) {
        return dup().mmuli(other,result);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public IComplexNDArray div(INDArray other) {
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
    public IComplexNDArray div(INDArray other, INDArray result) {
        return dup().divi(other,result);
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray mul(INDArray other) {
        return dup().muli(other);
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public IComplexNDArray mul(INDArray other, INDArray result) {
        return dup().muli(other,result);
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray sub(INDArray other) {
        return dup().subi(other);
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public IComplexNDArray sub(INDArray other, INDArray result) {
        return dup().subi(other,result);
    }

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray add(INDArray other) {
        return dup().addi(other);
    }

    /**
     * copy addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray add(INDArray other, INDArray result) {
        return dup().addi(other,result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public IComplexNDArray mmuli(INDArray other) {
        return mmuli(other,this);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public IComplexNDArray mmuli(INDArray other, INDArray result) {
        if (other.isScalar())
            return muli(other.getScalar(0), result);


        ComplexNDArray otherArray = new ComplexNDArray(other);
        ComplexNDArray resultArray = new ComplexNDArray(result);


		/* check sizes and resize if necessary */
        assertMultipliesWith(other);


        if (result == this || result == other) {
			/* actually, blas cannot do multiplications in-place. Therefore, we will fake by
			 * allocating a temporary object on the side and copy the result later.
			 */
            otherArray = otherArray.ravel().reshape(otherArray.shape);

            ComplexNDArray temp = new ComplexNDArray(resultArray.shape(),ArrayUtil.calcStridesFortran(resultArray.shape()));
            NDArrayBlas.gemm(org.jblas.ComplexDouble.UNIT, this, otherArray, org.jblas.ComplexDouble.ZERO, temp);

            NDArrayBlas.copy(temp, resultArray);

        }
        else {
            otherArray = otherArray.ravel().reshape(otherArray.shape);
            ComplexNDArray thisInput =  this.ravel().reshape(shape());
            NDArrayBlas.gemm(org.jblas.ComplexDouble.UNIT, thisInput, otherArray, org.jblas.ComplexDouble.ZERO, resultArray);
        }





        return resultArray;
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public IComplexNDArray divi(INDArray other) {
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
    public IComplexNDArray divi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(DivideOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(DivideOp.class)
                    .to(result).build().exec();
        return (IComplexNDArray) result;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray muli(INDArray other) {
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
    public IComplexNDArray muli(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(MultiplyOp.class)
                    .to(result).build().exec();

        else
            new TwoArrayOps().from(this).other(other).op(MultiplyOp.class)
                    .to(result).build().exec();
        return (IComplexNDArray) result;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray subi(INDArray other) {
        return subi(other,this);
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public IComplexNDArray subi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(SubtractOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(SubtractOp.class)
                    .to(result).build().exec();
        return (IComplexNDArray) result;
    }

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray addi(INDArray other) {
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
    public IComplexNDArray addi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(AddOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(AddOp.class)
                    .to(result).build().exec();
        return (IComplexNDArray) result;
    }



    @Override
    public ComplexNDArray get(int[] indices) {
        ComplexNDArray result = new ComplexNDArray(data,new int[]{1,indices.length},stride,offset);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, get(indices[i]));
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


    private void ensureSameShape(ComplexNDArray arr1,ComplexNDArray arr2) {
        assert true == Shape.shapeEquals(arr1.shape(), arr2.shape());

    }







    /**
     * Return a vector containing the means of all columns.
     */
    @Override
    public ComplexNDArray columnMeans() {
        if(shape().length == 2) {
            return ComplexNDArray.wrap(super.columnMeans());

        }

        else
            return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.MatrixOp.COLUMN_MEAN,this);

    }


    /**
     * Return a vector containing the sums of the columns (having number of columns many entries)
     */
    @Override
    public ComplexNDArray columnSums() {
        if(shape().length == 2) {
            return ComplexNDArray.wrap(super.columnSums());

        }
        else
            return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.MatrixOp.COLUMN_SUM,this);

    }


    /**
     * Return a vector containing the means of the rows for each slice.
     */
    @Override
    public ComplexNDArray rowMeans() {
        if(shape().length == 2) {
            return ComplexNDArray.wrap(super.rowMeans());

        }
        else
            return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.MatrixOp.ROW_MEAN,this);

    }



    /**
     * Return a matrix with the row sums for each slice
     */
    @Override
    public ComplexNDArray rowSums() {
        if(shape().length == 2) {
            return ComplexNDArray.wrap(super.rowSums());

        }

        else
            return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.MatrixOp.ROW_SUM,this);

    }


    /**
     * Add two matrices.
     *
     * @param other
     * @param result
     */
    @Override
    public ComplexNDArray addi(ComplexDoubleMatrix other, ComplexDoubleMatrix result) {
        if (other.isScalar())
            return addi(new ComplexDouble(other.scalar()), ComplexNDArray.wrap(result));

        assertSameLength(other);

        if (result == this)
            SimpleBlas.axpy(ComplexDouble.UNIT, other, result);
        else if (result == other)
            SimpleBlas.axpy(ComplexDouble.UNIT, this, result);
        else {
            SimpleBlas.copy(this, result);
            SimpleBlas.axpy(ComplexDouble.UNIT, other, result);
        }

        return ComplexNDArray.wrap(result);
    }



    /**
     * Add a scalar to a matrix.
     *
     * @param v
     * @param result
     */
    @Override
    public ComplexNDArray addi(org.jblas.ComplexDouble v, ComplexDoubleMatrix result) {
        ComplexNDArray r = ComplexNDArray.wrap(result);
        new TwoArrayOps().from(this).to(r).scalar(ComplexNDArray.scalar(new ComplexDouble(v))).op(AddOp.class)
                .build().exec();
        return r;
    }

    @Override
    public ComplexNDArray addi(double v, ComplexDoubleMatrix result) {
        ComplexNDArray r = ComplexNDArray.wrap(result);
        new TwoArrayOps().from(this).to(r).scalar(ComplexNDArray.scalar(v)).op(AddOp.class)
                .build().exec();
        return r;
    }





    /** Add two matrices (in-place). */
    public ComplexNDArray addi(ComplexNDArray other, ComplexNDArray result) {
        if (other.isScalar())
            return addi(other.scalar(), result);

        assertSameLength(other);

        if (result == this)
            SimpleBlas.axpy(ComplexDouble.UNIT, other, result);
        else if (result == other)
            SimpleBlas.axpy(ComplexDouble.UNIT, this, result);
        else {
            SimpleBlas.copy(this, result);
            SimpleBlas.axpy(ComplexDouble.UNIT, other, result);
        }

        return result;
    }


    /**
     * Add a scalar to a matrix (in-place).
     * @param other
     */
    @Override
    public ComplexNDArray addi(ComplexDoubleMatrix other) {
        ComplexNDArray r = ComplexNDArray.wrap(other);
        new TwoArrayOps().from(this).to(this).other(r).op(AddOp.class)
                .build().exec();
        return this;
    }

    /**
     * Subtract two matrices (in-place).
     *
     * @param other
     */
    @Override
    public ComplexNDArray subi(ComplexDoubleMatrix other) {
        ComplexNDArray r = ComplexNDArray.wrap(other);
        new TwoArrayOps().from(this).to(this).other(r).op(SubtractOp.class).build().exec();
        return this;
    }


    /** Subtract a scalar from a matrix */
    @Override
    public ComplexNDArray subi(org.jblas.ComplexDouble v, ComplexDoubleMatrix result) {
        ComplexNDArray wrapped = ComplexNDArray.wrap(result);
        new TwoArrayOps().from(this).to(wrapped).op(SubtractOp.class).scalar(ComplexNDArray.scalar(new ComplexDouble(v))).build().exec();
        return wrapped;
    }

    @Override
    public ComplexNDArray subi(org.jblas.ComplexDouble v) {
        return subi(new ComplexDouble(v), this);
    }



    /**
     * Elementwise multiply by a scalar.
     *
     * @param v
     */
    @Override
    public ComplexNDArray mul(double v) {
        return dup().muli(v);
    }


    /**
     * Elementwise multiplication (in-place).
     *
     * @param result
     */
    @Override
    public ComplexNDArray muli(ComplexDoubleMatrix result) {
        new TwoArrayOps().from(this).to(this).other(ComplexNDArray.wrap(result))
                .op(MultiplyOp.class).build().exec();
        return this;
    }


    /**
     * Matrix-matrix multiplication (in-place).
     * @param result
     */
    @Override
    public ComplexNDArray mmuli( ComplexDoubleMatrix result) {
        super.mmuli(result);
        return this;
    }


    @Override
    public ComplexNDArray muli(org.jblas.ComplexDouble v) {
        return muli(v, this);
    }

    /**
     * Elementwise division (in-place).
     *
     * @param result
     */
    @Override
    public ComplexNDArray divi(ComplexDoubleMatrix result) {
        ComplexNDArray flatten = ComplexNDArray.wrap(result);
        new TwoArrayOps().from(this).to(this).other(flatten)
                .op(DivideOp.class).build().exec();
        return this;
    }


    /** (Elementwise) division with a scalar */
    @Override
    public ComplexNDArray divi(org.jblas.ComplexDouble a, ComplexDoubleMatrix result) {
        ComplexNDArray flatten = ComplexNDArray.wrap(result);
        new TwoArrayOps().from(this).to(flatten).scalar(ComplexNDArray.scalar(new ComplexDouble(a))).build().exec();
        return flatten;
    }



    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public ComplexNDArray addi(double v) {
        new TwoArrayOps().from(this).op(AddOp.class)
                .scalar(ComplexNDArray.scalar(v)).build().exec();
        return this;
    }

    /**
     * Compute elementwise logical and against a scalar.
     *
     * @param value
     */
    @Override
    public ComplexNDArray andi(double value) {
        super.andi(value);
        return this;
    }

    /**
     * Elementwise divide by a scalar (in place).
     *
     * @param v
     */
    @Override
    public ComplexNDArray divi(double v) {
        new TwoArrayOps().from(this).scalar(ComplexNDArray.scalar(v)).op(DivideOp.class).build().exec();
        return this;
    }

    /**
     * Matrix-multiply by a scalar.
     *
     * @param v
     */
    @Override
    public ComplexNDArray mmul(double v) {
        return muli(v);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public ComplexNDArray subi(double v) {
        new TwoArrayOps().from(this).scalar(ComplexNDArray.scalar(v)).op(SubtractOp.class).build().exec();
        return this;
    }




    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public ComplexNDArray transpose() {
        //transpose of row vector is column vector
        if(isRowVector())
            return new ComplexNDArray(data,new int[]{shape[0],1},offset);
            //transpose of a column vector is row vector
        else if(isColumnVector())
            return new ComplexNDArray(data,new int[]{shape[0]},offset);

        ComplexNDArray n = new ComplexNDArray(data,reverseCopy(shape),reverseCopy(stride),offset);
        return n;

    }


    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     * @param shape
     * @return
     */
    public ComplexNDArray reshape(int[] shape) {
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

        ComplexNDArray ndArray = new ComplexNDArray(data,shape,stride,offset);
        return ndArray;

    }

    /** (Elementwise) Multiplication with a scalar */
    @Override
    public ComplexNDArray muli(org.jblas.ComplexDouble v, ComplexDoubleMatrix result) {
        ComplexNDArray wrap = ComplexNDArray.wrap(result);
        new TwoArrayOps().from(this).to(wrap).scalar(ComplexNDArray.scalar(new ComplexDouble(v))).op(MultiplyOp.class).build().exec();
        return this;
    }

    @Override
    public ComplexNDArray mul(org.jblas.ComplexDouble v) {
        return dup().muli(v, new ComplexDoubleMatrix(rows, columns));
    }


    public void checkDimensions(ComplexNDArray other) {
        assert Arrays.equals(shape,other.shape) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape);
        assert Arrays.equals(stride,other.stride) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride);
        assert offset == other.offset : "Offset of this array is " + offset + " but other was " + other.offset;

    }


    @Override
    public ComplexNDArray div(double v) {
        return divi(new ComplexDouble(v), new ComplexDoubleMatrix(rows, columns));
    }


    @Override
    public ComplexNDArray mmul(ComplexDoubleMatrix other) {
        ComplexNDArray n = ComplexNDArray.wrap(other);
        return dup().mmuli(other, new ComplexDoubleMatrix(rows(), n.columns()));
    }



    /**
     * Check whether this can be multiplied with a.
     *
     * @param a right-hand-side of the multiplication.
     * @return true iff <tt>this.columns == a.rows</tt>
     */
    @Override
    public boolean multipliesWith(ComplexDoubleMatrix a) {
        return columns() == ComplexNDArray.wrap(a).rows();
    }



    /**
     * Check whether this can be multiplied with a.
     *
     * @param a right-hand-side of the multiplication.
     * @return true iff <tt>this.columns == a.rows</tt>
     */

    public boolean multipliesWith(INDArray a) {
        return columns() == a.rows();
    }

    public void assertMultipliesWith(ComplexDoubleMatrix a) {
        if (!multipliesWith(a))
            throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
    }

    public void assertMultipliesWith(INDArray a) {
        if (!multipliesWith(a))
            throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
    }

    /** Matrix-Matrix Multiplication */
    @Override
    public ComplexNDArray mmuli(ComplexDoubleMatrix other, ComplexDoubleMatrix result) {
        if (other.isScalar())
            return ComplexNDArray.wrap(muli(other.scalar(), result));


        ComplexNDArray otherArray = ComplexNDArray.wrap(other);
        ComplexNDArray resultArray = ComplexNDArray.wrap(result);


		/* check sizes and resize if necessary */
        assertMultipliesWith(other);
        if (result.rows != rows || result.columns != other.columns) {
            if (result != this && result != other)
                result.resize(rows, other.columns);
            else
                throw new SizeException("Cannot resize result matrix because it is used in-place.");
        }

        if (result == this || result == other) {
			/* actually, blas cannot do multiplications in-place. Therefore, we will fake by
			 * allocating a temporary object on the side and copy the result later.
			 */
            otherArray = otherArray.ravel().reshape(otherArray.shape);

            ComplexNDArray temp = new ComplexNDArray(resultArray.shape(),ArrayUtil.calcStridesFortran(resultArray.shape()));
            NDArrayBlas.gemm(ComplexDouble.UNIT, this, otherArray, ComplexDouble.ZERO, temp);

            NDArrayBlas.copy(temp, resultArray);

        }
        else {
            otherArray = otherArray.ravel().reshape(otherArray.shape);
            ComplexNDArray thisInput =  this.ravel().reshape(shape());
            NDArrayBlas.gemm(ComplexDouble.UNIT, thisInput, otherArray, ComplexDouble.ZERO, resultArray);
        }





        return resultArray;
    }


    /**
     * Returns a copy of
     * all of the data in this array in order
     * @return all of the data in order
     */
    public double[] data() {
        double[] ret = new double[length * 2];
        ComplexNDArray flattened = ravel();
        int count = 0;
        for(int i = 0; i < flattened.length; i++) {
            ret[count++] = flattened.get(i).realComponent();
            ret[count++] = flattened.get(i).imaginaryComponent();
        }

        return ret;
    }

    @Override
    public void setData(double[] data) {
        this.data = data;
    }

    /**
     * Returns a linear float array representation of this ndarray
     *
     * @return the linear float array representation of this ndarray
     */
    @Override
    public float[] floatData() {
        return ArrayUtil.floatCopyOf(data);
    }

    @Override
    public void setData(float[] data) {
        this.data = ArrayUtil.doubleCopyOf(data);
    }


    /**
     * Checks whether the matrix is a row vector.
     */
    @Override
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
    @Override
    public boolean isColumnVector() {
        if(shape().length == 1)
            return false;

        if(isVector())
            return shape()[1] == 1;

        return false;

    }



    /**
     * Add a matrix (in place).
     *
     * @param o
     */
    @Override
    public ComplexNDArray add(ComplexDoubleMatrix o) {
        return dup().addi(o);
    }

    /**
     * Add a scalar.
     *
     * @param v
     */
    @Override
    public ComplexNDArray add(double v) {
        return dup().addi(v);

    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public ComplexNDArray div(ComplexDoubleMatrix other) {
        return dup().divi(other);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public ComplexNDArray sub(ComplexDoubleMatrix other) {
        return dup().subi(other);
    }

    /**
     * Subtract a scalar.
     *
     * @param v
     */
    @Override
    public ComplexNDArray sub(double v) {
        return dup().subi(v);
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public ComplexNDArray mul(ComplexDoubleMatrix other) {
        return dup().muli(other);
    }

    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public ComplexNDArray muli(double v) {
        new TwoArrayOps().from(this).scalar(ComplexNDArray.scalar(v)).op(MultiplyOp.class).build().exec();
        return this;
    }


    private void initShape(int[] shape) {

        this.shape = shape;

        if(this.shape.length == 2) {
            if(this.shape[0] == 1) {
                this.shape = new int[1];
                this.shape[0] = shape[1];
            }

            if(this.shape.length == 1) {
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
        if(this.stride == null) {
            this.stride = ArrayUtil.calcStrides(this.shape,2);

        }

        if(this.stride.length != this.shape.length) {
            this.stride = ArrayUtil.calcStrides(this.shape,2);

        }

    }



    /**
     * Computes the sum of all elements of the matrix.
     */
    @Override
    public ComplexDouble sum() {
        ComplexDouble d = new ComplexDouble(0);

        if(isVector()) {
            for(int i = 0; i < length(); i++) {
                org.jblas.ComplexDouble d2 = (ComplexDouble) getScalar(i).element();
                d.addi(d2);
            }
        }
        else {
            ComplexNDArray reshape = reshape(new int[]{1,length()});
            for(int i = 0; i < reshape.length(); i++) {
                org.jblas.ComplexDouble d2 = (ComplexDouble) reshape.getScalar(i).element();
                d.addi(d2);
            }
        }

        return d;
    }

    /**
     * The 1-norm of the matrix as vector (sum of absolute values of elements).
     */
    @Override
    public double norm1() {
        if(isVector())
            return super.norm2();
        return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.ScalarOp.NORM_1,this).real();

    }




    public ComplexDouble mean() {
        ComplexDouble d = new ComplexDouble(0);

        if(isVector()) {
            for(int i = 0; i < length(); i++) {
                org.jblas.ComplexDouble d2 = (ComplexDouble) getScalar(i).element();
                d.addi(d2);
            }
        }
        else {
            ComplexNDArray reshape = reshape(new int[]{1,length()});
            for(int i = 0; i < reshape.length(); i++) {
                org.jblas.ComplexDouble d2 = (ComplexDouble) reshape.getScalar(i).element();
                d.addi(d2);
            }
        }


        d.divi(new org.jblas.ComplexDouble(length(),length()));

        return d;

    }


    public ComplexDouble prod() {
        ComplexDouble d = new ComplexDouble(1);

        if(isVector()) {
            for(int i = 0; i < length(); i++) {
                org.jblas.ComplexDouble d2 = (ComplexDouble) getScalar(i).element();
                d.muli(d2);
            }
        }
        else {
            ComplexNDArray reshape = reshape(new int[]{1,length()});
            for(int i = 0; i < reshape.length(); i++) {
                org.jblas.ComplexDouble d2 = (ComplexDouble) reshape.getScalar(i).element();
                d.muli(d2);
            }
        }

        return d;

    }


    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    @Override
    public IComplexNDArray prod(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return ComplexNDArray.scalar(reshape(new int[]{1,length}).prod());
        }

        else if(isVector()) {
            return ComplexNDArray.scalar(sum().divi(length));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
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

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public IComplexNDArray mean(int dimension) {
        if(isVector()) {
            return ComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
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

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    @Override
    public IComplexNDArray assign(Number value) {
        IComplexNDArray one = reshape(new int[]{1,length});
        for(int i = 0; i < one.length(); i++)
            one.put(i,NDArrays.complexScalar(value));
        return one;
    }


    /**
     * Reverse division
     *
     * @param other the matrix to divide from
     * @return
     */
    @Override
    public IComplexNDArray rdiv(INDArray other) {
        return dup().rdivi(other);
    }

    /**
     * Reverse divsion (in place)
     *
     * @param other
     * @return
     */
    @Override
    public IComplexNDArray rdivi(INDArray other) {
        return rdivi(other,this);
    }

    /**
     * Reverse division
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public IComplexNDArray rdiv(INDArray other, INDArray result) {
        return dup().rdivi(other,result);
    }

    /**
     * Reverse division (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public IComplexNDArray rdivi(INDArray other, INDArray result) {
        return (IComplexNDArray) other.divi(this, result);
    }

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public IComplexNDArray rsub(INDArray other, INDArray result) {
        return dup().rsubi(other,result);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public IComplexNDArray rsub(INDArray other) {
        return dup().rsubi(other);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public IComplexNDArray rsubi(INDArray other) {
        return rsubi(other,this);
    }

    /**
     * Reverse subtraction (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public IComplexNDArray rsubi(INDArray other, INDArray result) {
        return (IComplexNDArray) other.subi(this, result);
    }


    public IComplexNumber max() {
        IComplexNDArray reshape = ravel();
        IComplexNumber max = (IComplexNumber) reshape.getScalar(0).element();

        for(int i = 1; i < reshape.length(); i++) {
            IComplexNumber curr = (IComplexNumber) reshape.getScalar(i).element();
            double val = curr.realComponent().doubleValue();
            if(val > curr.realComponent().doubleValue())
                max = curr;

        }
        return max;
    }


    public IComplexNumber min() {
        IComplexNDArray reshape = ravel();
        IComplexNumber min = (IComplexNumber) reshape.getScalar(0).element();
        for(int i = 1; i < reshape.length(); i++) {
            IComplexNumber curr = (IComplexNumber) reshape.getScalar(i).element();
            double val = curr.realComponent().doubleValue();
            if(val < curr.realComponent().doubleValue())
                min = curr;

        }
        return min;
    }

    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public IComplexNDArray max(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArrays.scalar(reshape(new int[]{1,length}).max());
        }

        else if(isVector()) {
            return NDArrays.scalar(max());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.max(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.max(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape).transpose();
        }
    }

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public IComplexNDArray min(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArrays.scalar(reshape(new int[]{1,length}).min());
        }

        else if(isVector()) {
            return NDArrays.scalar(min());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.min(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.min(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape).transpose();
        }
    }


    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public IComplexNDArray normmax(int dimension) {
        if(isVector()) {
            return ComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                    arr.put(i.get(),arr2.normmax(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.normmax(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }


    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    @Override
    public IComplexNDArray sum(int dimension) {
        if(dimension == Integer.MAX_VALUE)
            return ComplexNDArray.scalar(sum());
        if(isVector()) {
            return ComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
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

            return arr.reshape(shape);
        }
    }

    @Override
    public void setStride(int[] stride) {
        this.stride = stride;
    }

    /**
     * The Euclidean norm of the matrix as vector, also the Frobenius
     * norm of the matrix.
     */
    @Override
    public double norm2() {
        if(isVector())
            return super.norm2();
        return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.ScalarOp.NORM_2,this).real();

    }

    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public IComplexNDArray norm1(int dimension) {
        if(dimension == Integer.MAX_VALUE)
            return ComplexNDArray.scalar(norm1());

        else if(isVector()) {
            return ComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
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

    public ComplexDouble std() {
        StandardDeviation dev = new StandardDeviation();
        INDArray real = getReal();
        NDArray imag = imag();
        double std = dev.evaluate(real.data());
        double std2 = dev.evaluate(imag.data());
        return new ComplexDouble(std,std2);
    }

    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to get the std along
     * @return the standard deviation along a particular dimension
     */
    @Override
    public INDArray std(int dimension) {
        if(dimension == Integer.MAX_VALUE)
            return ComplexNDArray.scalar(std());
        if(isVector()) {
            return ComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
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
        }
    }

    /**
     * The maximum norm of the matrix (maximal absolute value of the elements).
     */
    @Override
    public double normmax() {
        if(isVector() )
            return super.normmax();
        return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.ScalarOp.NORM_MAX,this).real();

    }

    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    public IComplexNDArray norm2(int dimension) {
        if(dimension == Integer.MAX_VALUE)
            return ComplexNDArray.scalar(norm2());
        if(isVector()) {
            return ComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
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


    /**
     * Checks whether the matrix is empty.
     */
    @Override
    public boolean isEmpty() {
        return length == 0;
    }





    /**
     * Converts the matrix to a one-dimensional array of doubles.
     */
    @Override
    public ComplexDouble[] toArray() {
        length = ArrayUtil.prod(shape);
        ComplexDouble[] ret = new ComplexDouble[length];
        for(int i = 0; i < ret.length; i++)
            ret[i] = get(i);
        return ret;
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
     * Returns the number of rows
     * in the array (only 2d) throws an exception when
     * called when not 2d
     * @return the number of rows in the matrix
     */
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




    /**
     * Reshape the matrix. Number of elements must not change.
     *
     * @param newRows
     * @param newColumns
     */
    @Override
    public ComplexNDArray reshape(int newRows, int newColumns) {
        return reshape(new int[]{newRows,newColumns});
    }


    /**
     * Iterate over every row of every slice
     * @param op the operation to apply
     */
    public void iterateOverAllRows(SliceOp op) {

        if(isVector())
            op.operate(new DimensionSlice(false,this,null));

        else {
            for(int i = 0; i < slices(); i++) {
                ComplexNDArray slice = slice(i);
                slice.iterateOverAllRows(op);
            }
        }

    }

    /**
     * Get the specified column
     *
     * @param c
     */
    @Override
    public ComplexNDArray getColumn(int c) {
        if(shape.length == 2) {
            int offset = this.offset + c * 2;
            return new ComplexNDArray(
                    data,
                    new int[]{shape[0], 1},
                    new int[]{stride[0], 2},
                    offset
            );
        }

        else
            throw new IllegalArgumentException("Unable to getFromOrigin row of non 2d matrix");

    }





    /**
     * Get a copy of a row.
     *
     * @param r
     */
    @Override
    public ComplexNDArray getRow(int r) {
        if(shape.length == 2)
            return new ComplexNDArray(
                    data,
                    new int[]{shape[1]},
                    new int[]{stride[1]},
                    offset +  (r * 2) * columns()
            );
        else
            throw new IllegalArgumentException("Unable to getFromOrigin row of non 2d matrix");

    }

    /**
     * Compare two matrices. Returns true if and only if other is also a
     * ComplexDoubleMatrix which has the same size and the maximal absolute
     * difference in matrix elements is smaller than 1e-6.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        ComplexNDArray n = null;
        if(o instanceof  ComplexDoubleMatrix && !(o instanceof ComplexNDArray)) {
            ComplexDoubleMatrix d = (ComplexDoubleMatrix) o;
            //chance for comparison of the matrices if the shape of this matrix is 2
            if(shape().length > 2)
                return false;

            else
                n = ComplexNDArray.wrap(d);


        }
        else if(!o.getClass().isAssignableFrom(ComplexNDArray.class))
            return false;

        if(n == null)
            n = (ComplexNDArray) o;

        //epsilon equals
        if(isScalar() && n.isScalar()) {
            org.jblas.ComplexDouble c = n.get(0);
            return Math.abs(get(0).sub(c).real()) < 1e-6;
        }
        else if(isVector() && n.isVector()) {
            for(int i = 0; i < length; i++) {
                double curr = get(i).realComponent();
                double comp = n.get(i).realComponent();
                double currImag = get(i).imaginaryComponent();
                double compImag = n.get(i).imaginaryComponent();
                if(Math.abs(curr - comp) > 1e-6 || Math.abs(currImag - compImag) > 1e-6)
                    return false;
            }

            return true;

        }

        if(!Shape.shapeEquals(shape(),n.shape()))
            return false;
        //epsilon equals
        if(isScalar()) {
            org.jblas.ComplexDouble c = n.get(0);
            return get(0).sub(c).abs() < 1e-6;
        }
        else if(isVector()) {
            for(int i = 0; i < length; i++) {
                ComplexDouble curr = get(i);
                org.jblas.ComplexDouble comp = n.get(i);
                if(curr.sub(comp).abs() > 1e-6)
                    return false;
            }

            return true;


        }

        for (int i = 0; i< slices(); i++) {
            if (!(slice(i).equals(n.slice(i))))
                return false;
        }

        return true;


    }




    /**
     * Get elements from specified rows and columns.
     *
     * @param rs
     * @param cs
     */
    public ComplexNDArray get(Range rs, Range cs) {
        rs.init(0, rows());
        cs.init(0, columns());
        ComplexNDArray result = new ComplexNDArray(rs.length(), cs.length());

        for (; rs.hasMore(); rs.next()) {
            cs.init(0, columns());
            for (; cs.hasMore(); cs.next()) {
                result.put(rs.index(), cs.index(), get(rs.value(), cs.value()));
            }
        }

        return result;
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
     * Note that for edge cases (including vectors and scalars)
     * {@link IllegalArgumentException}
     *
     * are thrown
     *
     *
     * @param dimension the dimension to return from
     * @return the shape of the specified dimension
     */
    public int size(int dimension) {
        if(isScalar()) {
            if(dimension == 0)
                return length;
            else
                throw new IllegalArgumentException("Illegal dimension for scalar " + dimension);
        }

        else if(isVector()) {
            if(dimension == 0 || dimension == 1)
                return length;
            else
                throw new IllegalArgumentException("No dimension for vector " + dimension);
        }


        return shape[dimension];
    }

    /**
     * Returns the total number of elements in the ndarray
     *
     * @return the number of elements in the ndarray
     */
    @Override
    public int length() {
        return length;
    }

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public IComplexNDArray broadcast(int[] shape) {
        return null;
    }

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public IComplexNDArray broadcasti(int[] shape) {
        return null;
    }

    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     *
     * @return the individual item in this ndarray
     */
    @Override
    public Object element() {
        if(!isScalar())
            throw new IllegalStateException("Unable to get the element of a non scalar");
        return get(0);
    }


    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    public ComplexNDArray permute(int[] rearrange) {
        checkArrangeArray(rearrange);
        int[] newDims = doPermuteSwap(shape,rearrange);
        int[] newStrides = doPermuteSwap(stride,rearrange);

        ComplexNDArray ret = new ComplexNDArray(data,newDims,newStrides,offset);

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
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    public ComplexNDArray ravel() {
        ComplexNDArray ret = new ComplexNDArray(new int[]{1,length});
        List<ComplexNDArray> list = new ArrayList<>();
        sliceVectors(list);
        int count = 0;
        for(int i = 0; i < list.size(); i++) {
            for(int j = 0; j < list.get(i).length; j++)
                ret.put(count++,list.get(i).get(j));
        }
        return ret;
    }

    /**
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    private void sliceVectors(List<ComplexNDArray> list) {
        if(isVector())
            list.add(this);
        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
    }


    /**
     * Checks whether the matrix is a vector.
     */
    @Override
    public boolean isVector() {
        return shape.length == 1
                ||
                shape.length == 1  && shape[0] == 1
                ||
                shape.length == 2 && (shape[0] == 1 || shape[1] == 1) && !isScalar();
    }

    /** Generate string representation of the matrix. */
    @Override
    public String toString() {

        StringBuilder sb = new StringBuilder();

        if (isScalar()) {
            return String.valueOf(get(0));
        }
        else if(isVector()) {
            sb.append("[ ");
            for(int i = 0; i < length; i++) {
                sb.append(get(i));
                if(i < length - 1)
                    sb.append(" ,");
            }

            sb.append("]\n");
            return sb.toString();
        }


        int length = shape[0];
        sb.append("[");
        if (length > 0) {
            sb.append(slice(0).toString());
            for (int i = 1; i < slices(); i++) {
                sb.append(slice(i).toString());
                if(i < slices() - 1)
                    sb.append(',');

            }
        }
        sb.append("]\n");
        return sb.toString();
    }



    public static ComplexNDArray scalar(ComplexNDArray from,int index) {
        return new ComplexNDArray(from.data,new int[]{1},new int[]{1},index);
    }


    /**
     * Create a scalar ndarray with the specified number as the realComponent
     * component and 0 as an imaginary
     * @param num the number to use
     * @return a scalar ndarray
     */
    public static ComplexNDArray scalar(double num) {
        return new ComplexNDArray(new double[]{num,0},new int[]{1},new int[]{1},0);
    }

    public static ComplexNDArray scalar(ComplexDouble num) {
        return new ComplexNDArray(new double[]{num.realComponent(),num.imaginaryComponent()},new int[]{1},new int[]{1},0);
    }

    /**
     * Wrap toWrap with the specified shape, and dimensions from
     * the passed in ndArray
     * @param ndArray the way to wrap a matrix
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static ComplexNDArray wrap(ComplexNDArray ndArray,ComplexDoubleMatrix toWrap) {
        if(toWrap instanceof ComplexNDArray)
            return (ComplexNDArray) toWrap;
        int[] stride = ndArray.stride();
        ComplexNDArray ret = new ComplexNDArray(toWrap.data,ndArray.shape(),stride,ndArray.offset());
        return ret;
    }



    public static ComplexNDArray zeros(int[] shape) {
        return new ComplexNDArray(shape);
    }

    /**
     * Wrap a matrix in to an ndarray
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static ComplexNDArray wrap(ComplexDoubleMatrix toWrap) {
        if(toWrap instanceof ComplexNDArray)
            return (ComplexNDArray) toWrap;
        int[] shape;
        if(toWrap.isColumnVector())
            shape = new int[]{toWrap.columns};
        else if(toWrap.isRowVector())
            shape = new int[]{ toWrap.rows};
        else
            shape = new int[]{toWrap.rows,toWrap.columns};
        ComplexNDArray ret = new ComplexNDArray(toWrap.data,shape);
        return ret;
    }


    public static ComplexNDArray linspace(int lower,int upper,int num) {
        return new ComplexNDArray(new NDArray(DoubleMatrix.linspace(lower, upper, num).data,new int[]{num}));
    }



}
