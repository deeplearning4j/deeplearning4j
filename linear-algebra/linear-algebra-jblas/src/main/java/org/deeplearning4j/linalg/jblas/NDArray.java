package org.deeplearning4j.linalg.jblas;


import static org.deeplearning4j.linalg.util.ArrayUtil.*;

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.DimensionSlice;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.factory.NDArrayFactory;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.jblas.complex.ComplexNDArray;
import org.deeplearning4j.linalg.jblas.util.MatrixUtil;
import org.deeplearning4j.linalg.jblas.util.NDArrayBlas;
import org.deeplearning4j.linalg.jblas.util.NDArrayUtil;
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
import org.jblas.ComplexDouble;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.Range;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * NDArray: (think numpy)
 * @author Adam Gibson
 */
public class NDArray extends DoubleMatrix implements INDArray {



    private int[] shape;
    private int[] stride;
    private int offset = 0;
    private boolean changedStride = false;
    private int[] oldStride;

    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     * @param slices the slices to merge
     * @param shape the shape of the ndarray
     */
    public NDArray(List<INDArray> slices,int[] shape) {
        List<double[]> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++)
            list.add(slices.get(i).data());

        this.data = ArrayUtil.combine(list);

        initShape(shape);



    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     * @param slices the slices to merge
     * @param shape the shape of the ndarray
     */
    public NDArray(List<INDArray> slices,int[] shape,int[] stride) {
        List<double[]> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++)
            list.add(slices.get(i).data());

        this.data = ArrayUtil.combine(list);
        this.stride = stride;
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

    public NDArray(float[] data, int[] shape, int[] stride, int offset) {
        this(ArrayUtil.doubleCopyOf(data),shape,stride,offset);
    }


    /**
     * Returns a linear float array representation of this ndarray
     *
     * @return the linear float array representation of this ndarray
     */
    @Override
    public float[] floatData() {
        return ArrayUtil.floatCopyOf(data());
    }

    @Override
    public void setData(float[] data) {
        this.data = ArrayUtil.doubleCopyOf(data);
    }


    public NDArray(DoubleMatrix d) {
        this(d.data,new int[]{d.rows,d.columns});
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
    public INDArray vectorAlongDimension(int index, int dimension) {
        assert dimension <= shape.length : "Invalid dimension " + dimension;
        if(shape.length == 2) {
            if(dimension == 1)
                return new NDArray(data,
                        new int[]{shape[dimension]}
                        ,new int[]{stride[dimension]},
                        offset + index * stride[0]);
            else if(dimension == 0)
                return new NDArray(data,
                        new int[]{shape[dimension]}
                        ,new int[]{stride[dimension]},
                        offset + index);


        }

        if(dimension == shape.length - 1)
            return new NDArray(data,
                    new int[]{1,shape[dimension]}
                    ,ArrayUtil.removeIndex(stride,0),
                    offset + index * stride[dimension - 1]);

        else if(dimension == 0)
            return new NDArray(data,
                    new int[]{shape[dimension],1}
                    ,new int[]{stride[dimension],1},
                    offset + index);



        return new NDArray(data,
                new int[]{shape[dimension],1}
                ,new int[]{stride[dimension],1},
                offset + index * stride[0]);

    }

    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
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

    /**
     * Cumulative sum along a dimension (in place)
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public INDArray cumsum(int dimension) {
        return dup().cumsumi(dimension);
    }

    /**
     * Assign all of the elements in the given
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    @Override
    public INDArray assign(INDArray arr) {
        LinAlgExceptions.assertSameShape(this,arr);
        INDArray other = arr.ravel();
        INDArray thisArr = ravel();
        for(int i = 0; i < other.length(); i++)
            thisArr.put(i, other.getScalar(i));
        return this;
    }

    @Override
    public INDArray putScalar(int i, Number value) {
        return put(i,NDArrays.scalar(value));
    }

    @Override
    public INDArray putScalar(int[] i, Number value) {
        return null;
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

    /**
     * Negate each element.
     */
    @Override
    public NDArray neg() {
        return dup().negi();
    }

    /**
     * Negate each element (in-place).
     */
    @Override
    public NDArray negi() {
        return (NDArray) Transforms.neg(this);
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
    public INDArray getScalar(int row, int column) {
        return get(new int[]{row,column});
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
     * Insert the element at the specified position
     * @param indexes the index to insert into
     * @param value the value to insert
     * @return the ndarray with the element
     * inserted
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
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int[] indices, INDArray element) {
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

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, int j, INDArray element) {
        return put(new int[]{i,j},element);
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


    /**
     * Assigns the given matrix (put) to the specified slice
     * @param slice the slice to assign
     * @param put the slice to applyTransformToDestination
     * @return this for chainability
     */
    @Override
    public NDArray putSlice(int slice,INDArray put) {
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


        NDArray view = slice(slice);

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
     * dimension)
     */
    @Override
    public NDArray reduce(Ops.DimensionOp op,int dimension) {
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
            

            for(int offset = this.offset; offset < numTimes; ) {
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

            return new NDArray(data2,shape);
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
    private IterationResult op(int dimension, int offset, Ops.DimensionOp op,int currOffsetForSlice) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        boolean newSlice = false;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
        	  
        	if(j >= currOffsetForSlice){
                  newSlice = true;
                  break;
        	}
        	
        	double d = data[j];
            dim[count++] = d;
          
        }

        return new IterationResult(reduceVector(op,new DoubleMatrix(dim)),newSlice);
    }


    //getFromOrigin one result along one dimension based on the given offset
    private double op(int dimension, int offset, Ops.DimensionOp op) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op,new DoubleMatrix(dim));
    }



    private double reduceVector(Ops.DimensionOp op,DoubleMatrix vector) {

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



    /**
     * Returns the squared (Euclidean) distance.
     */
    public double squaredDistance(INDArray other) {
        double sd = 0.0;
        for (int i = 0; i < length; i++) {
            double d = get(i) - (double) other.getScalar(i).element();
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
            d += Math.abs((double) getScalar(i).sub(other.getScalar(i)).element());
        }
        return d;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, Number element) {
        return null;
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
    @Override
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


    @Override
    public double[] data() {
        return data;
    }

    @Override
    public void setData(double[] data) {
        this.data = data;
    }


    public NDArray subArray(int[] shape) {
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



    @Override
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
                    NDArray result = (NDArray) slice.getResult();
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
                    NDArray result = (NDArray) slice.getResult();
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
                        NDArray result = (NDArray) slice.getResult();
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
            for(int i = 0; i < vectorsAlongDimension(dimension); i++) {
                INDArray vector = vectorAlongDimension(i,dimension);
                op.operate(vector);
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


    @Override
    public void setStride(int[] stride) {
        this.oldStride = ArrayUtil.copy(this.stride);
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
    public NDArray getScalar(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return NDArray.scalar(data[idx]);
    }



    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public NDArray put(int i, INDArray element) {
        if(element == null)
            throw new IllegalArgumentException("Unable to insert null element");
        assert element.isScalar() : "Unable to insert non scalar element";

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
    public NDArray diviColumnVector(INDArray columnVector) {
        assert columnVector.isColumnVector() : "Must only add a column vector";
        assert columnVector.length() == rows() : "Illegal column vector must have the same length as the number of column in this ndarray";

        for(int i = 0; i < columns(); i++) {
            getColumn(i).divi(columnVector);
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
    public NDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public NDArray diviRowVector(INDArray rowVector) {
        assert rowVector.isRowVector() : "Must only add a row vector";
        assert rowVector.length() == columns() : "Illegal row vector must have the same length as the number of rows in this ndarray";
        for(int j = 0; j< rows(); j++) {
            getRow(j).divi(rowVector);
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
    public NDArray divRowVector(INDArray rowVector) {
        return dup().diviRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public NDArray muliColumnVector(INDArray columnVector) {
        assert columnVector.isColumnVector() : "Must only add a column vector";
        assert columnVector.length() == rows() : "Illegal column vector must have the same length as the number of column in this ndarray";

        for(int i = 0; i < columns(); i++) {
            getColumn(i).muli(columnVector);
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
    public NDArray mulColumnVector(INDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public NDArray muliRowVector(INDArray rowVector) {
        assert rowVector.isRowVector() : "Must only add a row vector";
        assert rowVector.length() == columns() : "Illegal row vector must have the same length as the number of rows in this ndarray";
        for(int j = 0; j< rows(); j++) {
            getRow(j).muli(rowVector);
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
    public NDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public NDArray subiColumnVector(INDArray columnVector) {
        assert columnVector.isColumnVector() : "Must only add a column vector";
        assert columnVector.length() == rows() : "Illegal column vector must have the same length as the number of column in this ndarray";

        for(int i = 0; i < columns(); i++) {
            getColumn(i).subi(columnVector);
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
    public NDArray subColumnVector(INDArray columnVector) {
        return dup().subiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public NDArray subiRowVector(INDArray rowVector) {
        assert rowVector.isRowVector() : "Must only add a row vector";
        assert rowVector.length() == columns() : "Illegal row vector must have the same length as the number of rows in this ndarray";
        for(int j = 0; j< rows(); j++) {
            getRow(j).subi(rowVector);
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
    public NDArray subRowVector(INDArray rowVector) {
        return dup().subiRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public NDArray addiColumnVector(INDArray columnVector) {
        assert columnVector.isColumnVector() : "Must only add a column vector";
        assert columnVector.length() == rows() : "Illegal column vector must have the same length as the number of column in this ndarray";

        for(int i = 0; i < columns(); i++) {
            getColumn(i).addi(columnVector);
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
    public NDArray addColumnVector(INDArray columnVector) {
        return dup().addiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public NDArray addiRowVector(INDArray rowVector) {
        assert rowVector.isRowVector() : "Must only add a row vector";
        assert rowVector.length() == columns() : "Illegal row vector must have the same length as the number of rows in this ndarray";
        for(int j = 0; j< rows(); j++) {
            getRow(j).addi(rowVector);
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
    public NDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public NDArray mmul(INDArray other) {
        int[] shape = {rows(),other.columns()};
        char order = NDArrays.factory().order();
        boolean switchedOrder = false;
        if(order != NDArrayFactory.FORTRAN) {
            NDArrays.factory().setOrder(NDArrayFactory.C);
           switchedOrder = true;
        }

        INDArray result = NDArrays.create(shape);

        if(switchedOrder)
            NDArrays.factory().setOrder(NDArrayFactory.C);

        return mmuli(other,result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public NDArray mmul(INDArray other, INDArray result) {
        return dup().mmuli(other,result);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public NDArray div(INDArray other) {
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
    public NDArray div(INDArray other, INDArray result) {
        return dup().divi(other,result);
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public NDArray mul(INDArray other) {
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
    public NDArray mul(INDArray other, INDArray result) {
        return dup().muli(other,result);
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public NDArray sub(INDArray other) {
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
    public NDArray sub(INDArray other, INDArray result) {
        return dup().subi(other,result);
    }

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public NDArray add(INDArray other) {
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
    public NDArray add(INDArray other, INDArray result) {
        return dup().addi(other,result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public NDArray mmuli(INDArray other) {
        return dup().mmuli(other,this);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public NDArray mmuli(INDArray other, INDArray result) {
        NDArray otherArray = (NDArray) other;
        NDArray resultArray = (NDArray) result;

        if (other.isScalar()) {
            return muli(otherArray.scalar(), resultArray);
        }
        if (isScalar()) {
            return otherArray.muli(scalar(), resultArray);
        }

        /* check sizes and resize if necessary */
        //assertMultipliesWith(other);


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

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public NDArray divi(INDArray other) {
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
    public NDArray divi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(DivideOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(DivideOp.class)
                    .to(result).build().exec();
        return (NDArray) result;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public NDArray muli(INDArray other) {
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
    public NDArray muli(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(MultiplyOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(MultiplyOp.class)
                    .to(result).build().exec();
        return (NDArray) result;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public NDArray subi(INDArray other) {
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
    public NDArray subi(INDArray other, INDArray result) {

        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(SubtractOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(SubtractOp.class)
                    .to(result).build().exec();
        return (NDArray) result;
    }

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public NDArray addi(INDArray other) {
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
    public NDArray addi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(AddOp.class)
                    .to(result).build().exec();

        else
            new TwoArrayOps().from(this).other(other).op(AddOp.class)
                    .to(result).build().exec();
        return (NDArray) result;
    }

    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray normmax(int dimension) {
        if(isVector()) {
            return NDArray.scalar(normmax());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
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
                    INDArray arr2 = nd;
                    arr.put(i.get(),arr2.normmax(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }


    /**
     * Linear getScalar ignoring linear restrictions
     * @param i the index of the element to getScalar
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

    /**
     * Reverse division
     *
     * @param other the matrix to divide from
     * @return
     */
    @Override
    public INDArray rdiv(INDArray other) {
        return dup().rdivi(other);
    }

    /**
     * Reverse divsion (in place)
     *
     * @param other
     * @return
     */
    @Override
    public INDArray rdivi(INDArray other) {
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
    public INDArray rdiv(INDArray other, INDArray result) {
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
    public INDArray rdivi(INDArray other, INDArray result) {
        return other.divi(this, result);
    }

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public INDArray rsub(INDArray other, INDArray result) {
        return dup().rsubi(other,result);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsub(INDArray other) {
        return dup().rsubi(other);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsubi(INDArray other) {
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
    public INDArray rsubi(INDArray other, INDArray result) {
        return other.subi(this, result);
    }

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    @Override
    public INDArray assign(Number value) {
        INDArray one = reshape(new int[]{1,length});
        for(int i = 0; i < one.length(); i++)
            one.put(i,NDArrays.scalar(value.doubleValue()));
        return one;
    }

    @Override
    public int linearIndex(int i) {
        if(oldStride != null)
            return offset + i;
        int realStride = getRealStrideForLinearIndex();
        int idx = offset + i * realStride;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }

    private int getRealStrideForLinearIndex() {
        if(oldStride != null) {
            if(oldStride == null || oldStride.length < 1)
                return 1;
            if(oldStride.length == 2 && shape[0] == 1)
                return stride[1];
            if(stride().length == 2 && oldStride[1] == 1)
                return oldStride[0];
            return oldStride[0];
        }
        else {
            if(stride == null || stride().length < 1)
                return 1;
            if(stride.length == 2 && shape[0] == 1)
                return stride[1];
            if(stride().length == 2 && shape[1] == 1)
                return stride[0];
            return stride[0];
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
    @Override
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

        else{
       
        	int[] strides = Arrays.copyOfRange(stride, 1, stride.length);
        	strides[0] = shape[shape.length -1];
        	return new NDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                   strides,
                    offset + (slice * stride[0]));
        }
            

    }


    /**
     * Returns the slice of this from the specified dimension
     * @param slice the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    @Override
    public NDArray slice(int slice, int dimension) {
        if(shape.length == 2) {
            //rows
            if(dimension == 1)
                return getRow(slice);


            else if(dimension == 0)
                return getColumn(slice);

            else throw new IllegalAccessError("Illegal dimension for matrix");

        }

        if (slice == shape.length - 1)
            return slice(dimension);

        return new NDArray(data,
                ArrayUtil.removeIndex(shape,dimension),
                ArrayUtil.removeIndex(stride,dimension),
                offset + slice * stride[dimension]);
    }



    /**
     * Fetch a particular number on a multi dimensional scale.
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    @Override
    public INDArray getScalar(int... indexes) {
        int ix = offset;
        int trackStride = shape[0];
        ix += indexes[0] * stride[shape.length - 1];
        for (int i = 1; i < shape.length; i++) {
        	int firstTerm = (indexes[i]);
        	int strideVal = stride[shape.length - 1 -i];
            ix +=  firstTerm * (trackStride) ;
        	trackStride *= strideVal;
        }
        return NDArrays.scalar(data[ix]);
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


    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    public NDArray repmat(int[] shape) {
        int[] newShape = ArrayUtil.copy(shape());
        assert shape.length <= newShape.length : "Illegal shape: The passed in shape must be <= the current shape length";
        for(int i = 0; i < shape.length; i++)
            newShape[i] *= shape[i];
        INDArray result = NDArrays.create(newShape);
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

        return (NDArray) result;
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
    public NDArray putRow(int row, INDArray toPut) {
        DoubleMatrix put = (NDArray) toPut;
        putRow(row,put);
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
    public NDArray putColumn(int column, INDArray toPut) {
        DoubleMatrix put = (NDArray) toPut;
        putColumn(column, put);
        return this;
    }



    @Override
    public NDArray get(int[] indices) {
        return (NDArray) getScalar(indices);
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
        assert true == Shape.shapeEquals(arr1.shape(), arr2.shape());

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
    @Override
    public NDArray addi(double v, DoubleMatrix result) {
        new TwoArrayOps().from(this).to(NDArray.wrap(result)).scalar(NDArray.scalar(v))
                .op(AddOp.class).build().exec();
        return this;
    }




    /**
     * Add a scalar to a matrix (in-place).
     * @param other
     */
    @Override
    public NDArray addi(DoubleMatrix other) {
        new TwoArrayOps().from(this).to(this).other(NDArray.wrap(other))
                .op(AddOp.class).build().exec();
        return this;
    }

    /**
     * Subtract two matrices (in-place).
     *
     * @param other
     */
    @Override
    public NDArray subi(DoubleMatrix other) {
        NDArray ret = NDArray.wrap(other);
        new TwoArrayOps().from(this).to(this).other(ret)
                .op(SubtractOp.class).build().exec();

        return this;
    }

    /**
     * Elementwise multiplication (in-place).
     *
     * @param other
     */
    @Override
    public NDArray muli(DoubleMatrix other) {
        new TwoArrayOps().from(this).to(this).other(NDArray.wrap(other))
                .op(MultiplyOp.class).build().exec();

        return this;

    }
    /**
     * Elementwise division (in-place).
     *
     * @param other
     */
    @Override
    public NDArray divi(DoubleMatrix other) {
        new TwoArrayOps().from(this).to(this).other(NDArray.wrap(other))
                .op(DivideOp.class).build().exec();

        return this;

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
        new TwoArrayOps().from(this).to(this).scalar(NDArray.scalar(v))
                .op(AddOp.class).build().exec();

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
        new TwoArrayOps().from(this).to(this).scalar(NDArray.scalar(v))
                .op(DivideOp.class).build().exec();

        return this;
    }

    /**
     * Matrix-multiply by a scalar.
     *
     * @param v
     */
    @Override
    public NDArray mmul(double v) {
        return mul(v);
    }

    /**
     * Subtract a scalar (in place).
     *
     * @param v
     */
    @Override
    public NDArray subi(double v) {
        new TwoArrayOps().from(this).to(this).scalar(NDArray.scalar(v))
                .op(SubtractOp.class).build().exec();

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
    @Override
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


    @Override
    public void checkDimensions(INDArray other) {
        assert Arrays.equals(shape,other.shape()) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape());
        assert Arrays.equals(stride,other.stride()) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride());
        assert offset == other.offset() : "Offset of this array is " + offset + " but other was " + other.offset();

    }


    /** Elementwise multiplication with a scalar (in-place). */
    @Override
    public NDArray muli(double v, DoubleMatrix result) {
        new TwoArrayOps().from(this).to(this).scalar(NDArray.scalar(v))
                .op(MultiplyOp.class).build().exec();

        return this;
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
        return dup().subi(v);
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
        new TwoArrayOps().from(this).to(this).scalar(NDArray.scalar(v))
                .op(MultiplyOp.class).build().exec();

        return this;
    }

    @Override
    public double get(int i) {
        int idx = linearIndex(i);
        return data[idx];
    }

    @Override
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

    /**
     * Computes the sum of all elements of the matrix.
     */
    @Override
    public double sum() {
        if(isVector()) {
            double ret = 0.0;
            for(int i = 0; i < length; i++) {
                ret += get(i);
            }
            return ret;
        }
        return (double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.SUM,this).element();
    }

    /**
     * The 1-norm of the matrix as vector (sum of absolute values of elements).
     */
    @Override
    public double norm1() {
        if(isVector()) {
            double norm = 0.0;
            for (int i = 0; i < length; i++) {
                norm += Math.abs(get(i));
            }
            return norm;
        }
        return (Double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.NORM_1,this).element();

    }

    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
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

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray mean(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArray.scalar(reshape(new int[]{1,length}).mean());
        }
        else if(isVector()) {
            return NDArray.scalar(sum() / length());
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


    public double var() {
        double mean = (double) mean(Integer.MAX_VALUE).element();
        return StatUtils.variance(data(),mean);
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

    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray max(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArray.scalar(reshape(new int[]{1,length}).max());
        }

        else if(isVector()) {
            return NDArray.scalar(max());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
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
    public INDArray min(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArray.scalar(reshape(new int[]{1,length}).min());
        }

        else if(isVector()) {
            return NDArray.scalar(min());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
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
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    @Override
    public INDArray sum(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArray.scalar(reshape(new int[]{1,length}).sum());
        }

        else if(isVector()) {
            return NDArray.scalar(sum());
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

            return arr.reshape(shape);
        }
    }

    /**
     * The Euclidean norm of the matrix as vector, also the Frobenius
     * norm of the matrix.
     */
    @Override
    public double norm2() {
        if(isVector()) {
            double norm = 0.0;
            for (int i = 0; i < length; i++) {
                norm += get(i) * get(i);
            }
            return   Math.sqrt(norm);
        }
        return (double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.NORM_2,this).element();

    }

    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray norm1(int dimension) {
        if(isVector()) {
            return NDArray.scalar(norm1());
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

    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to get the std along
     * @return the standard deviation along a particular dimension
     */
    @Override
    public INDArray std(int dimension) {
        if(isVector()) {
            return NDArray.scalar(std());
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
        }
    }

    /**
     * The maximum norm of the matrix (maximal absolute value of the elements).
     */
    @Override
    public double normmax() {
        if(isVector())
            return super.normmax();
        return (double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.NORM_MAX,this).element();

    }

    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    public INDArray norm2(int dimension) {
        if(isVector()) {
            return NDArray.scalar(norm2());
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

    /**
     * Computes the product of all elements of the matrix
     */
    @Override
    public double prod() {
        if(isVector() ) {
            double ret = 0.0;
            for(int i = 0; i < length; i++) {
                ret *= get(i);
            }
            return ret;
        }

        return (double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.PROD,this).element();

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
        if(isVector() ) {
            double ret = Double.NEGATIVE_INFINITY;
            for(int i = 0; i < length; i++) {
                if(get(i) > ret)
                    ret = get(i);
            }
            return ret;
        }
        return (double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MAX,this).element();

    }

    /**
     * Returns the minimal element of the matrix.
     */
    @Override
    public double min() {
        if(isVector() ) {
            double ret = Double.NEGATIVE_INFINITY;
            for(int i = 0; i < length; i++) {
                if(get(i) > ret)
                    ret = get(i);
            }
            return ret;
        }
        return (double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MIN,this).element();

    }

    /**
     * Computes the mean value of all elements in the matrix,
     * that is, <code>x.sum() / x.length</code>.
     */
    @Override
    public double mean() {
        if(isVector() ) {
            double ret = 0.0;
            for(int i = 0; i < length; i++) {
                ret += get(i);
            }
            return ret / length();
        }
        return (double) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.MEAN,this).element();

    }

    /**
     * Returns the linear index of the maximal element of the matrix. If
     * there are more than one elements with this value, the first one
     * is returned.
     */
    @Override
    public int argmax() {
        if(isVector() ) {
            double ret = Double.NEGATIVE_INFINITY;
            int idx = 0;
            for(int i = 0; i < length; i++) {
                if(get(i) > ret) {
                    ret = get(i);
                    idx = i;
                }
            }
            return idx;
        }
        return (int) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.ARG_MAX,this).element();

    }

    /**
     * Returns the linear index of the minimal element. If there are
     * more than one elements with this value, the first one is returned.
     */
    @Override
    public int argmin() {
        if(isVector() ) {
            double ret = Double.NEGATIVE_INFINITY;
            int idx = 0;
            for(int i = 0; i < length; i++) {
                if(get(i) < ret) {
                    ret = get(i);
                    idx = i;
                }
            }
            return idx;
        }
        return (int) NDArrayUtil.doSliceWise(NDArrayUtil.ScalarOp.ARG_MIN,this).element();

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
    @Override
    public NDArray ravel() {
        return reshape(new int[]{1,length()});
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
            if(oldStride == null)
                return new NDArray(
                        data,
                        new int[]{shape[0]},
                        new int[]{stride[0]},
                        offset + c
                );
            else
                return new NDArray(
                        data,
                        new int[]{shape[0]},
                        new int[]{oldStride[0]},
                        offset + c
                );
        else
            throw new IllegalArgumentException("Unable to getFromOrigin column of non 2d matrix");
    }


    /**
     * Get whole rows from the passed indices.
     *
     * @param rindices
     */
    @Override
    public NDArray getRows(int[] rindices) {
        INDArray rows = NDArrays.create(rindices.length,columns());
        for(int i = 0; i < rindices.length; i++) {
            rows.putRow(i,getRow(rindices[i]));
        }
        return (NDArray) rows;
    }

    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    public INDArray get(NDArrayIndex... indexes) {
        if(indexes.length < shape().length) {

        }
        return null;
    }

    /**
     * Get whole columns from the passed indices.
     *
     * @param cindices
     */
    @Override
    public NDArray getColumns(int[] cindices) {
        INDArray rows = NDArrays.create(rows(),cindices.length);
        for(int i = 0; i < cindices.length; i++) {
            rows.putColumn(i,getColumn(cindices[i]));
        }
        return (NDArray) rows;
    }

    /**
     * Get a copy of a row.
     *
     * @param r
     */
    @Override
    public NDArray getRow(int r) {
        if(shape.length == 2)
            if(oldStride == null)
                return new NDArray(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset +  r * columns()
                );
            else
                return new NDArray(
                        data,
                        new int[]{shape[1]},
                        new int[]{oldStride[1]},
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
    public NDArray broadcast(int[] shape) {
        return null;
    }

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public NDArray broadcasti(int[] shape) {
        return null;
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
                shape.length == 2 && (shape[0] == 1 || shape[1] == 1) && !isScalar();
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


    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     *
     * @return the individual item in this ndarray
     */
    @Override
    public Object element() {
        if(!isScalar())
            throw new IllegalStateException("Unable to retrieve element from non scalar matrix");
        return data[0];
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