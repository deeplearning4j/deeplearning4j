package org.deeplearning4j.nn.linalg;

import static org.deeplearning4j.util.ArrayUtil.calcStrides;
import static org.deeplearning4j.util.ArrayUtil.reverseCopy;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.linalg.elementwise.ops.AddOp;
import org.deeplearning4j.nn.linalg.elementwise.ops.DivideOp;
import org.deeplearning4j.nn.linalg.elementwise.ops.MultiplyOp;
import org.deeplearning4j.nn.linalg.elementwise.ops.SubtractOp;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.NDArrayBlas;
import org.deeplearning4j.util.NDArrayUtil;
import org.jblas.*;
import org.jblas.ranges.Range;

import java.io.PrintWriter;
import java.io.StringWriter;
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
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     * @param slices the slices to merge
     * @param shape the shape of the ndarray
     */
    public NDArray(List<NDArray> slices,int[] shape) {
        List<double[]> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++)
            list.add(slices.get(i).data);

        this.data = ArrayUtil.combine(list);

        initShape(shape);



    }



    public NDArray(double[] data,int[] shape,int[] stride) {
        this(data,shape,stride,0);
    }


    public NDArray(double[] data,int[] shape,int[] stride,int offset) {
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");



        this.offset = offset;
        this.stride = stride;

        initShape(shape);

        if(data != null  && data.length > 0)
            this.data = data;



    }

    /**
     * Create this ndarray with the given data and shape and 0 offset
     * @param data the data to use
     * @param shape the shape of the ndarray
     */
    public NDArray(double[] data,int[] shape) {
        this(data,shape,0);
    }

    public NDArray(double[] data,int[] shape,int offset) {
        this(data,shape,calcStrides(shape),offset);

    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public NDArray(int[] shape,int[] stride,int offset) {
        this(new double[ArrayUtil.prod(shape)],shape,stride,offset);
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public NDArray(int[] shape,int[] stride){
        this(shape,stride,0);
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
        initShape(new int[]{newRows,newColumns});
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
            throw new UnsupportedOperationException("Invalid applyTransformToDestination for a non 2d array");
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
        NDArray n = NDArray.wrap(v);
        if(n.isVector() && n.length != columns())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");
        NDArray row = getRow(r);
        for(int i = 0; i < v.length; i++)
            row.put(i,v.get(i));


    }

    /**
     * Copy a column back into the matrix.
     *
     * @param c
     * @param v
     */
    @Override
    public void putColumn(int c, DoubleMatrix v) {
        NDArray n = NDArray.wrap(v);
        if(n.isVector() && n.length != rows())
            throw new IllegalArgumentException("Unable to put row, mis matched columns");
        NDArray column = getColumn(c);
        for(int i = 0; i < v.length; i++)
            column.put(i,v.get(i));


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
    public NDArray put(int[] indexes, double value) {
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

    public NDArray putSlice(int slice,NDArray put) {
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


        NDArray view = slice(slice);

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


    private void assertSlice(NDArray put,int slice) {
        assert slice <= slices() : "Invalid slice specified " + slice;
        int[] sliceShape = put.shape();
        int[] requiredShape = ArrayUtil.removeIndex(shape(),0);

        //no need to compare for scalar; primarily due to shapes either being [1] or length 0
        if(put.isScalar())
            return;



        assert Shape.shapeEquals(sliceShape,requiredShape) : String.format("Invalid shape size of %s . Should have been %s ",Arrays.toString(sliceShape),Arrays.toString(requiredShape));

    }


    /**
     * Returns true if this ndarray is 2d
     * or 3d with a singleton element
     * @return true if the element is a matrix, false otherwise
     */
    public boolean isMatrix() {
        return (shape().length == 2
                && (shape[0] != 1 && shape[1] != 1)) ||
                shape.length == 3 &&
                        (shape[0] == 1 || shape[1] == 1 || shape[2] == 1);
    }

    /**
     *
     * http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduce.html
     * @param op the operation to do
     * @param dimension the dimension to return from
     * @return the results of the reduce (applying the operation along the specified
     * dimension)t
     */
    public NDArray reduce(NDArrayUtil.DimensionOp op,int dimension) {
        if(isScalar())
            return this;


        if(isVector())
            return NDArray.scalar(reduceVector(op, this));


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

            return new NDArray(data2,shape);
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
                Pair<Double,Boolean> pair = op(dimension, offset, op,sliceIndices[currOffset]);
                //append the result
                double reduce = pair.getFirst();
                data2[dataIter++] = reduce;

                //go to next slice and iterate over that
                if(pair.getSecond()) {
                    //will update to next step
                    offset = sliceIndices[currOffset];
                    numTimes +=  sliceIndices[currOffset];
                    currOffset++;
                }

            }

            return new NDArray(data2,shape);
        }


    }



    //getFromOrigin one result along one dimension based on the given offset
    public DimensionSlice vectorForDimensionAndOffsetPair(int dimension, int offset,int currOffsetForSlice) {
        int count = 0;
        NDArray ret = new NDArray(new int[]{shape[dimension]});
        boolean newSlice = false;
        List<Integer> indices = new ArrayList<>();
        for(int j = offset; count < this.shape[dimension]; j+= this.stride[dimension]) {
            double d = data[j];
            ret.put(count++,d);
            if(j >= currOffsetForSlice)
                newSlice = true;
            indices.add(j);

        }



        return new DimensionSlice(newSlice,ret,ArrayUtil.toArray(indices));
    }

    //getFromOrigin one result along one dimension based on the given offset
    public DimensionSlice vectorForDimensionAndOffset(int dimension, int offset) {
        if(isScalar() && dimension == 0 && offset == 0)
            return new DimensionSlice(false,ComplexNDArray.scalar(get(offset)),new int[]{offset});


            //need whole vector
        else   if (isVector()) {
            if(dimension == 0) {
                int[] indices = new int[length];
                for(int i = 0; i < indices.length; i++)
                    indices[i] = i;
                return new DimensionSlice(false,dup(),indices);
            }
            else if(dimension == 1)
                return new DimensionSlice(false,NDArray.scalar(get(offset)),new int[]{offset});
            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);

        }

        else {

            int count = 0;
            List<Integer> indices = new ArrayList<>();
            NDArray ret = new NDArray(new int[]{shape[dimension]});

            for(int j = offset; count < this.shape[dimension]; j+= this.stride[dimension]) {
                double d = data[j];
                ret.put(count++,d);
                indices.add(j);


            }

            return new DimensionSlice(false,ret,ArrayUtil.toArray(indices));

        }

    }


    //getFromOrigin one result along one dimension based on the given offset
    private Pair<Double,Boolean> op(int dimension, int offset, NDArrayUtil.DimensionOp op,int currOffsetForSlice) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        boolean newSlice = false;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
            if(j >= currOffsetForSlice)
                newSlice = true;
        }

        return new Pair<>(reduceVector(op,new DoubleMatrix(dim)),newSlice);
    }


    //getFromOrigin one result along one dimension based on the given offset
    private double op(int dimension, int offset, NDArrayUtil.DimensionOp op) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op,new DoubleMatrix(dim));
    }



    private double reduceVector(NDArrayUtil.DimensionOp op,DoubleMatrix vector) {

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


    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     * @param dimension the dimension to swap
     * @param with the one to swap it with
     * @return the swapped axes view
     */
    public NDArray swapAxes(int dimension,int with) {
        int[] shape = ArrayUtil.range(0,shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
    }



    /**
     * Gives the indices for the ending of each slice
     * @return the off sets for the beginning of each slice
     */
    public int[] endsForSlices() {
        int[] ret = new int[slices()];
        int currOffset = offset + stride[0] - 1;
        for(int i = 0; i < slices(); i++) {
            ret[i] = currOffset;
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
            currOffset += stride[0] + 1;
        }
        return ret;
    }


    public double[] data() {
        if(offset == 0)
            return data;


        double[] data = new double[length * 2];
        int count = 0;
        for(int i = 0; i < length; i++) {
            data[count++] = get(i);
        }
        return data;
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



    /**
     * Iterate along a dimension.
     * This encapsulates the process of sum, mean, and other processes
     * take when iterating over a dimension.
     * @param dimension the dimension to iterate over
     * @param op the operation to apply
     * @param modify whether to modify this array while iterating
     */
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
                    NDArray result = (NDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.get(i);
                    }
                }
            }
        }

        else if(isVector()) {
            if(dimension == 0) {
                DimensionSlice slice = this.vectorForDimensionAndOffset(0,0);
                op.operate(slice);
                if(modify && slice.getIndices() != null) {
                    NDArray result = (NDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.get(i);
                    }
                }
            }
            else if(dimension == 1) {
                for(int i = 0; i < length; i++) {
                    DimensionSlice slice = vectorForDimensionAndOffset(dimension,i);
                    op.operate(slice);
                    if(modify && slice.getIndices() != null) {
                        NDArray result = (NDArray) slice.getResult();
                        for(int j = 0; j < slice.getIndices().length; j++) {
                            data[slice.getIndices()[j]] = result.get(j);
                        }
                    }

                }
            }
            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);
        }


        else {

            int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

            if(dimension == 0) {
                //iterating along the dimension is relative to the number of slices
                //in the return dimension
                int numTimes = ArrayUtil.prod(shape);
                for(int offset = this.offset; offset < numTimes; offset++) {
                    DimensionSlice vector = vectorForDimensionAndOffset(dimension,offset);
                    op.operate(vector);
                    if(modify && vector.getIndices() != null) {
                        NDArray result = (NDArray) vector.getResult();
                        for(int i = 0; i < vector.getIndices().length; i++) {
                            data[vector.getIndices()[i]] = result.get(i);
                        }
                    }

                }

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

                    //do the operation, and look for whether it exceeded the current slice
                    DimensionSlice dimensionResult = vectorForDimensionAndOffsetPair(dimension, offset,sliceIndices[currOffset]);
                    //append the result
                    op.operate(dimensionResult);

                    if(modify && dimensionResult.getIndices() != null) {
                        NDArray result = (NDArray) dimensionResult.getResult();
                        for(int i = 0; i < dimensionResult.getIndices().length; i++) {
                            data[dimensionResult.getIndices()[i]] = result.get(i);
                        }
                    }

                    //go to next slice and iterate over that
                    if(dimensionResult.isNextSlice()) {
                        //will update to next step
                        offset = sliceIndices[currOffset];
                        numTimes +=  sliceIndices[currOffset];
                        currOffset++;
                    }

                }

            }

        }



    }


    /**
     * Get elements from specified rows and columns.
     *
     * @param rs
     * @param cs
     */
    @Override
    public NDArray get(Range rs, Range cs) {
        rs.init(0, rows());
        cs.init(0, columns());
        NDArray result = new NDArray(rs.length(), cs.length());

        for (; rs.hasMore(); rs.next()) {
            cs.init(0, columns());
            for (; cs.hasMore(); cs.next()) {
                result.put(rs.index(), cs.index(), get(rs.value(), cs.value()));
            }
        }

        return result;
    }

    public void setStride(int[] stride) {
        this.stride = stride;
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
            if(copy[i] instanceof  Character && (char)copy[i] == ':')
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


    @Override
    public NDArray put(int i, double v) {
        if(!isVector() && !isScalar())
            throw new IllegalStateException("Unable to do linear indexing on a non vector");
        int idx = linearIndex(i);
        data[idx] = v;
        return this;
    }

    @Override
    public double get(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return data[idx];
    }


    /**
     * Linear get ignoring linear restrictions
     * @param i the index of the element to get
     * @return the item at the given index
     */
    public double unSafeGet(int i) {
        int idx = unSafeLinearIndex(i);
        return data[idx];
    }




    public int unSafeLinearIndex(int i) {
        int realStride = getRealStrideForLinearIndex();
        int idx = offset + i;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }

    private int linearIndex(int i) {
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
    public NDArray dim(int dimension) {
        int[] shape = ArrayUtil.copy(shape());
        int[] stride = ArrayUtil.reverseCopy(this.stride);
        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            return new NDArray(data,new int[]{},new int[]{},offset + dimension * stride[0]);

            //slice of a matrix is a vector
        else if (shape.length == 2) {
            int st = stride[0];
            if (st == 1) {
                return new NDArray(
                        data,
                        ArrayUtil.of(shape[1]),
                        ArrayUtil.of(1),
                        offset + dimension * stride[0]);
            }

            else {

                return new NDArray(
                        data,
                        ArrayUtil.of(shape[1]),
                        ArrayUtil.of(stride[1]),
                        offset + dimension * stride[0]
                );
            }
        }

        else {
            return new NDArray(data,
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
    public NDArray slice(int slice) {

        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            return new NDArray(data,ArrayUtil.empty(),ArrayUtil.empty(),offset + slice * stride[0]);


            //slice of a matrix is a vector
        else if (shape.length == 2) {
            NDArray slice2 =  new NDArray(
                    data,
                    ArrayUtil.of(shape[1]),
                    Arrays.copyOfRange(stride,1,stride.length),
                    offset + slice * stride[0]
            );
            return slice2;

        }

        else
            return new NDArray(data,
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
    public NDArray slice(int slice, int dimension) {
        if (slice == 0)
            return slice(dimension);
        if (shape.length == 2) {
            if (slice != 1)
                throw new IllegalArgumentException("Unable to retrieve dimension " + slice + " from a 2d array");
            return new NDArray(data,
                    ArrayUtil.of(shape[0]),
                    ArrayUtil.of(stride[0]),
                    offset + dimension * stride[1]
            );
        }

        return new NDArray (
                data,
                ArrayUtil.removeIndex(shape,dimension),
                ArrayUtil.removeIndex(stride,dimension),
                offset + dimension * stride[slice]
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
    public NDArray dimension(int dim) {
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
    public double get(int rowIndex, int columnIndex) {
        return data[offset + (rowIndex * stride[0] + columnIndex)];
    }

    @Override
    public NDArray get(int[] indices) {
        NDArray result = new NDArray(data,new int[]{1,indices.length},stride,offset);

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


    private void ensureSameShape(NDArray arr1,NDArray arr2) {
        assert true == Shape.shapeEquals(arr1.shape(),arr2.shape());

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
            throw new IllegalStateException("Unable to getFromOrigin column mins for dimensions more than 2");
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
        new AddOp(this,result,v).exec();
        return this;
    }

    /** Add two matrices (in-place). */
    public NDArray addi(NDArray other, NDArray result) {
        new AddOp(other,result,this).exec();
        return result;
    }


    /**
     * Add a scalar to a matrix (in-place).
     * @param other
     */
    @Override
    public NDArray addi(DoubleMatrix other) {
        NDArray ret = NDArray.wrap(other);
        new AddOp(this,this,ret).exec();
        return this;
    }

    /**
     * Subtract two matrices (in-place).
     *
     * @param result
     */
    @Override
    public NDArray subi(DoubleMatrix result) {
        NDArray ret = NDArray.wrap(result);
        new SubtractOp(this,this,ret).exec();
        return this;
    }

    /**
     * Elementwise multiplication (in-place).
     *
     * @param result
     */
    @Override
    public NDArray muli(DoubleMatrix result) {
        NDArray ret = NDArray.wrap(result);
        new MultiplyOp(this,this,ret).exec();
        return this;

    }
    /**
     * Elementwise division (in-place).
     *
     * @param result
     */
    @Override
    public NDArray divi(DoubleMatrix result) {
        NDArray r = NDArray.wrap(result);
        new DivideOp(this,r,this).exec();
        return r;

    }




    /**
     * Elementwise multiply by a scalar.
     *
     * @param v
     */
    @Override
    public NDArray mul(double v) {
        return dup().muli(v);
    }





    /**
     * Matrix-matrix multiplication (in-place).
     * @param result
     */
    @Override
    public NDArray mmuli( DoubleMatrix result) {
        return NDArray.wrap(super.mmuli(result));
    }




    /**
     * Add a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray addi(double v) {
        new AddOp(this,v).exec();
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
    public NDArray divi(double v) {
        new DivideOp(this,v).exec();
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
        new SubtractOp(this,v).exec();
        return this;
    }

    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public NDArray transpose() {
        if(isRowVector())
            return new NDArray(data,new int[]{shape[0],1},offset);
        else if(isColumnVector())
            return new NDArray(data,new int[]{shape[0]},offset);
        NDArray n = new NDArray(data,reverseCopy(shape),reverseCopy(stride),offset);
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
        int n = (int) ec;

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


    /** Elementwise multiplication with a scalar (in-place). */
    @Override
    public NDArray muli(double v, DoubleMatrix result) {
        new MultiplyOp(NDArray.wrap(result),v).exec();
        return NDArray.wrap(result);
    }



    /** Matrix-matrix multiplication (in-place). */
    @Override
    public NDArray mmuli(DoubleMatrix other, DoubleMatrix result) {
        NDArray otherArray = NDArray.wrap(other);
        NDArray resultArray = NDArray.wrap(result);

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
            NDArray temp = new NDArray(resultArray.shape(),ArrayUtil.calcStridesFortran(resultArray.shape()));

            if (otherArray.columns() == 1) {
                NDArrayBlas.gemv(1.0, this, otherArray, 0.0, temp);
            } else {
                NDArrayBlas.gemm(1.0, this, otherArray, 0.0, temp);
            }

            NDArrayBlas.copy(temp, resultArray);


        } else {
            if (otherArray.columns() == 1)
                NDArrayBlas.gemv(1.0, this, otherArray, 0.0, resultArray);
            else
                NDArrayBlas.gemm(1.0, this, otherArray, 0.0, resultArray);

        }
        return resultArray;
    }


    @Override
    public NDArray mmul(DoubleMatrix a) {
        int[] shape = {rows(),NDArray.wrap(a).columns()};
        return mmuli(a,new NDArray(shape));
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


    @Override
    public int index(int row,int column) {
        return row * stride[0]  + column * stride[1];
    }


    /**
     * Add a matrix (in place).
     *
     * @param o
     */
    @Override
    public NDArray add(DoubleMatrix o) {
        return dup().addi(o);
    }

    /**
     * Add a scalar.
     *
     * @param v
     */
    @Override
    public NDArray add(double v) {
        return dup().addi(v);

    }

    /**
     * Elementwise divide by a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray div(DoubleMatrix other) {
        return dup().divi(other);
    }

    /**
     * Subtract a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray sub(DoubleMatrix other) {
        return dup().subi(other);
    }

    /**
     * Subtract a scalar.
     *
     * @param v
     */
    @Override
    public NDArray sub(double v) {
        return dup().subi(NDArray.scalar(v));
    }

    /**
     * Elementwise multiply by a matrix (in place).
     *
     * @param other
     */
    @Override
    public NDArray mul(DoubleMatrix other) {
        return dup().muli(other);
    }

    /**
     * Elementwise multiply by a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray muli(double v) {
        new MultiplyOp(this,v).exec();
        return this;
    }
    /**
     * Computes the sum of all elements of the matrix.
     */
    @Override
    public double sum() {
        if(isVector()) {
            double ret = 0.0;
            for(int i = 0; i < length; i++)
                ret += get(i);
            return ret;
        }

        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.SUM,this);
    }

    /**
     * The 1-norm of the matrix as vector (sum of absolute values of elements).
     */
    @Override
    public double norm1() {
        if(isVector())
            return super.norm2();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.NORM_1,this);

    }

    /**
     * The Euclidean norm of the matrix as vector, also the Frobenius
     * norm of the matrix.
     */
    @Override
    public double norm2() {
        if(isVector())
            return super.norm2();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.NORM_2,this);

    }

    /**
     * The maximum norm of the matrix (maximal absolute value of the elements).
     */
    @Override
    public double normmax() {
        if(isVector() )
            return super.normmax();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.NORM_MAX,this);

    }

    /**
     * Computes the product of all elements of the matrix
     */
    @Override
    public double prod() {
        if(isVector()) {
            double ret = 0.0;
            for(int i = 0; i < length; i++)
                ret *= get(i);
            return ret;
        }
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.PROD,this);

    }

    /**
     * Checks whether the matrix is empty.
     */
    @Override
    public boolean isEmpty() {
        return length == 0;
    }

    /**
     * Returns the maximal element of the matrix.
     */
    @Override
    public double max() {
        if(isVector() )
            return super.max();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MAX,this);

    }

    /**
     * Returns the minimal element of the matrix.
     */
    @Override
    public double min() {
        if(isVector() )
            return super.min();
        return NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MIN,this);

    }

    /**
     * Computes the mean value of all elements in the matrix,
     * that is, <code>x.sum() / x.length</code>.
     */
    @Override
    public double mean() {
        if(isVector() )
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
        if(isVector() )
            return super.argmax();
        return (int) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.ARG_MAX,this);

    }

    /**
     * Returns the linear index of the minimal element. If there are
     * more than one elements with this value, the first one is returned.
     */
    @Override
    public int argmin() {
        if(isVector() )
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
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    public NDArray ravel() {
        NDArray ret = new NDArray(new int[]{1,length});
        List<NDArray> list = new ArrayList<>();
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
    private void sliceVectors(List<NDArray> list) {
        if(isVector())
            list.add(this);
        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
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
                    new int[]{stride[0]},
                    offset + c
            );
        else
            throw new IllegalArgumentException("Unable to getFromOrigin column of non 2d matrix");
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
                    new int[]{stride[1]},
                    offset +  r * columns()
            );
        else
            throw new IllegalArgumentException("Unable to getFromOrigin row of non 2d matrix");
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
        if(o instanceof  DoubleMatrix && !(o instanceof NDArray)) {
            DoubleMatrix d = (DoubleMatrix) o;
            //chance for comparison of the matrices if the shape of this matrix is 2
            if(shape().length > 2)
                return false;

            else
                n = NDArray.wrap(d);


        }
        else if(!o.getClass().isAssignableFrom(NDArray.class))
            return false;

        if(n == null)
            n = (NDArray) o;

        //epsilon equals
        if(isScalar() && n.isScalar())
            return Math.abs(get(0) - n.get(0)) < 1e-6;
        else if(isVector() && n.isVector()) {
            for(int i = 0; i < length; i++) {
                double curr = get(i);
                double comp = n.get(i);
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
            NDArray slice = slice(i);
            NDArray nSlice = n.slice(i);

            if (!slice.equals(nSlice))
                return false;
        }

        return true;


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


    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    public NDArray permute(int[] rearrange) {
        checkArrangeArray(rearrange);

        int[] newShape = doPermuteSwap(shape,rearrange);
        int[] newStride = doPermuteSwap(stride,rearrange);
        return new NDArray(data,newShape,newStride,offset);
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
        return shape.length == 1
                ||
                shape.length == 1  && shape[0] == 1
                ||
                shape.length == 2 && (shape[0] == 1 || shape[1] == 1);
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

    /** Generate string representation of the matrix. */
    @Override
    public String toString() {
        if (isScalar()) {
            return Double.toString(get(0));
        }
        else if(isVector()) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for(int i = 0; i < length; i++) {
                sb.append(get(i));
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

    public ComplexNDArray add(ComplexDouble d) {
        return new ComplexNDArray(this).muli(d);
    }

    public ComplexNDArray mul(ComplexDouble d) {
        return new ComplexNDArray(this).muli(d);
    }

    public ComplexNDArray sub(ComplexDouble d) {
        return new ComplexNDArray(this).subi(d);
    }

    public ComplexNDArray div(ComplexDouble d) {
        return new ComplexNDArray(this).muli(d);
    }

    public static NDArray scalar(NDArray from,int index) {
        return new NDArray(from.data,new int[]{1},new int[]{1},index);
    }


    public static NDArray scalar(double num) {
        return new NDArray(new double[]{num},new int[]{1},new int[]{1},0);
    }

    /**
     * Wrap toWrap with the specified shape, and dimensions from
     * the passed in ndArray
     * @param ndArray the way to wrap a matrix
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static NDArray wrap(NDArray ndArray,DoubleMatrix toWrap) {
        if(toWrap instanceof NDArray)
            return (NDArray) toWrap;
        int[] stride = ndArray.stride();
        NDArray ret = new NDArray(toWrap.data,ndArray.shape(),stride,ndArray.offset());
        return ret;
    }


    /**
     * Wrap a matrix in to an ndarray
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static NDArray wrap(DoubleMatrix toWrap) {
        if(toWrap instanceof NDArray)
            return (NDArray) toWrap;
        int[]  shape = new int[]{toWrap.rows,toWrap.columns};
        NDArray ret = new NDArray(toWrap.data,shape);
        return ret;
    }



    public static NDArray linspace(int lower,int upper,int num) {
        return new NDArray(DoubleMatrix.linspace(lower,upper,num).data,new int[]{num});
    }


    public static NDArray arange(double begin, double end) {
        return NDArray.wrap(new DoubleMatrix(ArrayUtil.toDoubles(ArrayUtil.range((int) begin,(int)end))).transpose());
    }



}