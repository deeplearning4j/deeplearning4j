package org.deeplearning4j.nn.linalg;

import org.apache.commons.math3.complex.Complex;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.linalg.elementwise.complex.ComplexElementWiseOp;
import org.deeplearning4j.nn.linalg.elementwise.complex.ops.AddOp;
import org.deeplearning4j.nn.linalg.elementwise.complex.ops.DivideOp;
import org.deeplearning4j.nn.linalg.elementwise.complex.ops.MultiplyOp;
import org.deeplearning4j.nn.linalg.elementwise.complex.ops.SubtractOp;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.NDArrayBlas;
import org.jblas.*;
import org.jblas.exceptions.SizeException;
import org.jblas.ranges.Range;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.deeplearning4j.util.ArrayUtil.calcStrides;
import static org.deeplearning4j.util.ArrayUtil.reverseCopy;

/**
 * ComplexNDArray for complex numbers.
 *
 *
 * Note that the indexing scheme for a complex ndarray is 2 * length
 * not length.
 *
 * The reason for this is the fact that imaginary components have
 * to be stored alongside real components.
 *
 * @author Adam Gibson
 */
public class ComplexNDArray extends ComplexDoubleMatrix {
    private int[] shape;
    private int[] stride;
    private int offset = 0;



    /**
     * Creates a new <tt>ComplexDoubleMatrix</tt> of size 0 times 0.
     */
    public ComplexNDArray() {
        this.length = 0;
        this.shape = new int[0];
        this.data = new double[0];

    }


    /** Construct a complex matrix from a real matrix. */
    public ComplexNDArray(NDArray m) {
        this(m.shape());
        NativeBlas.dcopy(m.length, m.data, m.offset(), 1, data, offset, 2);

    }


    /**
     * Create an ndarray from the specified slices
     * and the given shape
     * @param slices the slices of the ndarray
     * @param shape the final shape of the ndarray
     */
    public ComplexNDArray(List<ComplexNDArray> slices,int[] shape) {
        super(new double[ArrayUtil.prod(shape) * 2]);
        List<ComplexDouble> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++) {
            ComplexNDArray flattened = slices.get(i).flatten();
            for(int j = 0; j < flattened.length; j++)
                list.add(flattened.get(j));
        }


        this.data = new double[ArrayUtil.prod(shape) * 2 ];

        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            data[count] = list.get(i).real();
            data[count + 1] = list.get(i).imag();
            count += 2;
        }

        initShape(shape);



    }


    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     */
    public ComplexNDArray(ComplexDouble[] newData,int[] shape) {
        super(newData);
        initShape(shape);

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

    @Override
    public ComplexNDArray dup() {
        double[] dupData = new double[data.length];
        System.arraycopy(data,0,dupData,0,dupData.length);
        ComplexNDArray ret = new ComplexNDArray(dupData,shape,stride,offset);
        return ret;
    }



    @Override
    public ComplexNDArray put(int rowIndex, int columnIndex, ComplexDouble value) {
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
    public ComplexNDArray put(int i, ComplexDouble v) {
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


    private void assertSlice(ComplexNDArray put,int slice) {
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
     *
     * http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduce.html
     * @param op the operation to do
     * @param dimension the dimension to return from
     * @return the results of the reduce (applying the operation along the specified
     * dimension)t
     */
    public ComplexNDArray reduce(ComplexNDArrayUtil.DimensionOp op,int dimension) {
        if(isScalar())
            return this;


        if(isVector())
            return ComplexNDArray.scalar(reduceVector(op, this));


        int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

        if(dimension == 0) {
            ComplexDouble[] data2 = new ComplexDouble[ArrayUtil.prod(shape)];
            int dataIter = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for(int offset = this.offset; offset < numTimes; offset++) {
                ComplexDouble reduce = op(dimension, offset, op);
                data2[dataIter++] = reduce;

            }

            return new ComplexNDArray(data2,shape);
        }

        else {
            ComplexDouble[] data2 = new ComplexDouble[ArrayUtil.prod(shape)];
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
                Pair<ComplexDouble,Boolean> pair = op(dimension, offset, op,sliceIndices[currOffset]);
                //append the result
                ComplexDouble reduce = pair.getFirst();
                data2[dataIter++] = reduce;

                //go to next slice and iterate over that
                if(pair.getSecond()) {
                    //will update to next step
                    offset = sliceIndices[currOffset];
                    numTimes +=  sliceIndices[currOffset];
                    currOffset++;
                }

            }

            return new ComplexNDArray(data2,shape);
        }


    }


    /**
     * Return the first element of the matrix
     */
    @Override
    public ComplexDouble scalar() {
        return super.scalar();
    }

    /**
     * Compute complex conjugate (in-place).
     */
    @Override
    public ComplexNDArray conji() {
        ComplexDouble c = new ComplexDouble(0.0);
        for (int i = 0; i < length; i++)
            put(i, get(i, c).conji());
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
     * Compute complex conjugate.
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
     * Get real part of the matrix.
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
            //real and imaginary
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
                        data[slice.getIndices()[i]] = result.get(i).real();
                        data[slice.getIndices()[i] + 1] = result.get(i).imag();
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
                        data[slice.getIndices()[i]] = result.get(i).real();
                        data[slice.getIndices()[i] + 1] = result.get(i).imag();
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
                            data[slice.getIndices()[j]] = result.get(j).real();
                            data[slice.getIndices()[j] + 1] = result.get(j).imag();
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
                            data[vector.getIndices()[i]] = result.get(i).real();
                            data[vector.getIndices()[i] + 1] = result.get(i).imag();
                        }
                    }

                }

            }

            else {
                //needs to be 2 * shape: this is due to both real and imaginary components
                double[] data2 = new double[ArrayUtil.prod(shape) ];
                int dataIter = 0;
                //want the milestone to slice[1] and beyond
                int[] sliceIndices = endsForSlices();
                int currOffset = 0;

                //iterating along the dimension is relative to the number of slices
                //in the return dimension
                //note here the  and +=2 this is for iterating over real and imaginary components
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
                            data[pair.getIndices()[i]] = result.get(i).real();
                            data[pair.getIndices()[i] + 1] = result.get(i).imag();
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
    private Pair<ComplexDouble,Boolean> op(int dimension, int offset, ComplexNDArrayUtil.DimensionOp op,int currOffsetForSlice) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        boolean newSlice = false;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
            if(j >= currOffsetForSlice)
                newSlice = true;
        }

        return new Pair<>(reduceVector(op,new ComplexDoubleMatrix(dim)),newSlice);
    }


    //getFromOrigin one result along one dimension based on the given offset
    private ComplexDouble op(int dimension, int offset, ComplexNDArrayUtil.DimensionOp op) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op,new ComplexDoubleMatrix(dim));
    }



    private ComplexDouble reduceVector(ComplexNDArrayUtil.DimensionOp op,ComplexDoubleMatrix vector) {

        switch(op) {
            case SUM:
                return vector.sum();
            case MEAN:
                return vector.mean();
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
    public int slices() {
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



    public int unSafeLinearIndex(int i) {
        int realStride = getRealStrideForLinearIndex();
        int idx = offset + i;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }

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
     * Iterate over a dimension. In the linear indexing context, we
     * can think of this as the following:
     * //number of operations per op
     int num = from.shape()[dimension];

     //how to isolate blocks from the matrix
     double[] d = new double[num];
     int idx = 0;
     for(int k = 0; k < d.length; k++) {
     d[k] = from.data[idx];
     idx += num;
     }

     *
     * With respect to a 4 3 2, if we are iterating over dimension 0
     * bump the index by 4
     *
     * The output for this is a matrix of num slices by number of columns
     *
     * @param dim the dimension to iterate along
     * @return the matrix containing the elements along
     * this dimension
     */
    public ComplexNDArray dimension(int dim) {
        return slice(1,dim);
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


    /** Retrieve matrix element */
    @Override
    public ComplexDouble get(int rowIndex, int columnIndex) {
        int index = offset +  index(rowIndex,columnIndex);
        return new ComplexDouble(data[index],data[index + 1]);
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





    /** Add a scalar to a matrix (in-place). */
    public ComplexNDArray addi(double v, ComplexNDArray result) {
        new AddOp(this,result,new ComplexDouble(v)).exec();
        return this;
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

        return ComplexNDArray.wrap(result);
    }



    /**
     * Add a scalar to a matrix.
     *
     * @param v
     * @param result
     */
    @Override
    public ComplexNDArray addi(ComplexDouble v, ComplexDoubleMatrix result) {
        ComplexNDArray r = ComplexNDArray.wrap(result);
        for(int i = 0; i < length; i++) {
            result.data[r.unSafeLinearIndex(i)] = data[unSafeLinearIndex(i)] + v.real();
            result.data[unSafeLinearIndex(i) + 1] = data[unSafeLinearIndex(i)] + v.imag();

        }

        return r;
    }

    @Override
    public ComplexNDArray addi(double v, ComplexDoubleMatrix result) {
        ComplexNDArray r = ComplexNDArray.wrap(result);
        for(int i = 0; i < length; i++) {
            result.data[r.unSafeLinearIndex(i)] = data[unSafeLinearIndex(i)] + v;

        }

        return r;
    }

    /** Add a scalar to a matrix. */
    public ComplexNDArray addi(ComplexDouble v, ComplexNDArray result) {
        ComplexNDArray r = ComplexNDArray.wrap(result);
        for(int i = 0; i < length; i++) {
            result.data[r.unSafeLinearIndex(i)] = data[unSafeLinearIndex(i)] + v.real();
            result.data[unSafeLinearIndex(i) + 1] = data[unSafeLinearIndex(i)] + v.imag();

        }

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
        new AddOp(this,this,r).exec();
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
        new SubtractOp(this,this,r).exec();
        return this;
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
        new MultiplyOp(this,this,ComplexNDArray.wrap(result)).exec();
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



    /**
     * Elementwise division (in-place).
     *
     * @param result
     */
    @Override
    public ComplexNDArray divi(ComplexDoubleMatrix result) {
        ComplexNDArray flatten = ComplexNDArray.wrap(result);
        new DivideOp(this,this,flatten).exec();
        return this;
    }

    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public ComplexNDArray addi(double v) {
        new AddOp(this,new ComplexDouble(v)).exec();
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
        ComplexNDArray flatten = flatten();
        for(int i = 0; i < flatten.length; i++)
            flatten.put(i,flatten.get(i).divi(v));
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
        new SubtractOp(this,new ComplexDouble(v)).exec();
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
            return new ComplexNDArray(data,new int[]{shape[1]},offset);

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

        ComplexNDArray ndArray = new ComplexNDArray(data,shape,offset);
        return ndArray;

    }

    /** (Elementwise) Multiplication with a scalar */
    @Override
    public ComplexNDArray muli(ComplexDouble v, ComplexDoubleMatrix result) {
        ComplexNDArray wrap = ComplexNDArray.wrap(result);
        new MultiplyOp(this,wrap,v).exec();
        return this;
    }

    @Override
    public ComplexNDArray mul(ComplexDouble v) {
        return dup().muli(v, new ComplexDoubleMatrix(rows, columns));
    }


    public void checkDimensions(ComplexNDArray other) {
        assert Arrays.equals(shape,other.shape) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape);
        assert Arrays.equals(stride,other.stride) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride);
        assert offset == other.offset : "Offset of this array is " + offset + " but other was " + other.offset;

    }



    @Override
    public ComplexNDArray mmul(ComplexDoubleMatrix other) {
        ComplexNDArray n = ComplexNDArray.wrap(other);
        return dup().mmuli(other, new ComplexDoubleMatrix(rows(), n.columns()));
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

            ComplexNDArray temp = new ComplexNDArray(resultArray.shape());
            NDArrayBlas.gemm(ComplexDouble.UNIT, this, otherArray, ComplexDouble.ZERO, temp);
            NDArrayBlas.copy(temp, resultArray);
        }
        else {
            otherArray = otherArray.flatten().reshape(otherArray.shape);
            ComplexNDArray thisInput =  this.flatten().reshape(shape());


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
        ComplexNDArray flattened = flatten();
        int count = 0;
        for(int i = 0; i < flattened.length; i++) {
            ret[count++] = flattened.get(i).real();
            ret[count++] = flattened.get(i).imag();
        }

        return ret;
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
        new MultiplyOp(this,new ComplexDouble(v)).exec();
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
     * Computes the sum of all elements of the matrix.
     */
    @Override
    public ComplexDouble sum() {
        if(isVector())
            return super.sum();
        return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.ScalarOp.SUM,this);
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
     * The maximum norm of the matrix (maximal absolute value of the elements).
     */
    @Override
    public double normmax() {
        if(isVector() )
            return super.normmax();
        return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.ScalarOp.NORM_MAX,this).real();

    }



    /**
     * Checks whether the matrix is empty.
     */
    @Override
    public boolean isEmpty() {
        return length == 0;
    }



    /**
     * Computes the mean value of all elements in the matrix,
     * that is, <code>x.sum() / x.length</code>.
     */
    @Override
    public ComplexDouble mean() {
        if(isVector())
            return super.mean();
        return ComplexNDArrayUtil.doSliceWise(ComplexNDArrayUtil.ScalarOp.MEAN, this);

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
        if(isMatrix()) {
            for(int i = 0; i < slices(); i++) {
                ComplexNDArray row = slice(i);
                op.operate(new DimensionSlice(false,row,null));

            }
        }

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
        if(isScalar() && n.isScalar())
            return Math.abs(get(0).sub(n.get(0)).real()) < 1e-6;
        else if(isVector() && n.isVector()) {
            for(int i = 0; i < length; i++) {
                double curr = get(i).real();
                double comp = n.get(i).real();
                double currImag = get(i).imag();
                double compImag = n.get(i).imag();
                if(Math.abs(curr - comp) > 1e-6 || Math.abs(currImag - compImag) > 1e-6)
                    return false;
            }

            return true;

        }

        if(!Shape.shapeEquals(shape(),n.shape()))
            return false;
        //epsilon equals
        if(isScalar())
            return get(0).sub(n.get(0)).abs() < 1e-6;
        else if(isVector()) {
            for(int i = 0; i < length; i++) {
                ComplexDouble curr = get(i);
                ComplexDouble comp = n.get(i);
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
     * {@link java.lang.IllegalArgumentException}
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
    public ComplexNDArray flatten() {
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
                shape.length == 2 && (shape[0] == 1 || shape[1] == 1);
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
     * Create a scalar ndarray with the specified number as the real
     * component and 0 as an imaginary
     * @param num the number to use
     * @return a scalar ndarray
     */
    public static ComplexNDArray scalar(double num) {
        return new ComplexNDArray(new double[]{num,0},new int[]{1},new int[]{1},0);
    }

    public static ComplexNDArray scalar(ComplexDouble num) {
        return new ComplexNDArray(new double[]{num.real(),num.imag()},new int[]{1},new int[]{1},0);
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
        return new ComplexNDArray(new NDArray(DoubleMatrix.linspace(lower,upper,num).data,new int[]{num}));
    }



}
