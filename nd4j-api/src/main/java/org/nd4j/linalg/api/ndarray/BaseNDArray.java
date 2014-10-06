package org.nd4j.linalg.api.ndarray;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;


import org.nd4j.linalg.ops.reduceops.Ops;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.*;


import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.nd4j.linalg.util.ArrayUtil.*;


/**
 * NDArray: (think numpy)
 *
 * A few things of note.
 *
 * An NDArray can have any number of dimensions.
 *
 * An NDArray is accessed via strides.
 *
 * Strides are how to index over
 * a contiguous block of data.
 *
 * This block of data has 2 orders(as of right now):
 * fortran and c
 *
 *
 *
 * @author Adam Gibson
 */
public abstract class BaseNDArray  implements INDArray {



    protected int[] shape;
    protected int[] stride;
    protected int offset = 0;
    protected char ordering;
    protected float[] data;
    protected int rows,columns;
    protected int length;
    protected INDArray linearView;


    public BaseNDArray() {}



    public BaseNDArray(double[][] data) {
        this(data.length, data[0].length);

        for (int r = 0; r < rows; r++) {
            assert (data[r].length == columns);
        }

        this.data = new float[length];


        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                putScalar(new int[]{r, c}, data[r][c]);
            }
        }
    }


    /**
     * Create this ndarray with the given data and shape and 0 offset
     * @param data the data to use
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape, char ordering) {
        this(data,shape,0,ordering);
    }

    /**
     *
     * @param data the data to use
     * @param shape the shape of the ndarray
     * @param offset the desired offset
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape, int offset, char ordering) {
        this(data,shape,ordering == NDArrayFactory.C ? calcStrides(shape) : calcStridesFortran(shape),offset);

    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride, int offset, char ordering) {
        this(new float[ArrayUtil.prod(shape)],shape,stride,offset,ordering);
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride, char ordering){
        this(shape,stride,0,ordering);
    }

    public BaseNDArray(int[] shape, int offset, char ordering) {
        this(shape, Nd4j.getStrides(shape, ordering),offset,ordering);
    }


    public BaseNDArray(int[] shape) {
        this(shape,0, Nd4j.order());
    }




    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseNDArray(int newRows, int newColumns, char ordering) {
        this.ordering = ordering;
        initShape(new int[]{newRows,newColumns});
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     * @param slices the slices to merge
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, char ordering) {
             this(slices,shape,Nd4j.getStrides(shape),ordering);
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     * @param slices the slices to merge
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, int[] stride, char ordering) {
        float[] ret = new float[ArrayUtil.prod(shape)];

        this.stride = stride;
        this.ordering = ordering;
        this.data = ret;

        initShape(shape);

        for(int i = 0; i < slices(); i++) {
           putSlice(i,slices.get(i));
        }






    }


    public BaseNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        this(data,shape,stride,0,ordering);
    }


    public BaseNDArray(float[] data, int[] shape, int[] stride, int offset, char ordering) {


        this.offset = offset;
        this.stride = stride;
        this.ordering = ordering;
        initShape(shape);

        if(data != null  && data.length > 0) {
            this.data = data;
            if(offset >= data.length)
                throw new IllegalArgumentException("Invalid offset: must be < data.length");


        }

        if(isVector())
            linearView = this;
        else
            this.linearView = Nd4j.create(data, new int[]{1, length}, offset());

    }

    @Override
    public INDArray linearViewColumnOrder() {
        return  Nd4j.create(data, new int[]{length,1},Nd4j.getStrides(new int[]{length,1}),offset());
    }

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    @Override
    public INDArray linearView() {
        if(isVector())
            return this;
        if(linearView == null)
            linearView = Nd4j.create(data, new int[]{1, length}, offset());
        return linearView;
    }

    /**
     * Create this ndarray with the given data and shape and 0 offset
     * @param data the data to use
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape) {
        this(data,shape,0);
    }

    public BaseNDArray(float[] data, int[] shape, int offset) {
        this(data,shape,offset, Nd4j.order());

    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public BaseNDArray(int[] shape, int[] stride, int offset) {
        this(new float[ArrayUtil.prod(shape)],shape,stride,offset, Nd4j.order());
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride){
        this(shape,stride,0);
    }

    public BaseNDArray(int[] shape, int offset) {
        this(shape,calcStrides(shape),offset);
    }


    public BaseNDArray(int[] shape, char ordering) {
        this(shape,0,ordering);
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseNDArray(int newRows, int newColumns) {
        this(newRows,newColumns, Nd4j.order());
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     * @param slices the slices to merge
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape) {
        this(slices,shape, Nd4j.order());
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     * @param slices the slices to merge
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, int[] stride) {
        this(slices,shape,stride, Nd4j.order());

    }


    public BaseNDArray(float[] data, int[] shape, int[] stride) {
        this(data,shape,stride, Nd4j.order());
    }




    public BaseNDArray(float[] data, int[] shape, int[] stride, int offset) {
        this(data,shape,stride,offset, Nd4j.order());
    }

    public BaseNDArray(float[] data) {
        this.data = data;
    }

    public BaseNDArray(float[][] data) {
        this(data.length, data[0].length);

        for (int r = 0; r < rows; r++) {
            assert (data[r].length == columns);
        }

        this.data = new float[length];


        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                putScalar(new int[]{r, c}, data[r][c]);
            }
        }
    }


    /**
     * Returns a linear float array representation of this ndarray
     *
     * @return the linear float array representation of this ndarray
     */
    @Override
    public float[] floatData() {
        return data;
    }



    public int majorStride() {
        return stride[0];
    }


    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    @Override
    public int vectorsAlongDimension(int dimension) {
        if(dimension >= shape.length)
            return length / size(shape.length - 1);
        return length / size(dimension);
    }

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    @Override
    public INDArray vectorAlongDimension(int index, int dimension) {
        assert dimension <= shape.length : "Invalid dimension " + dimension;
        if(dimension > shape().length - 1)
            dimension = shape.length - 1;
        if(ordering == NDArrayFactory.C) {

            if(dimension == shape.length - 1 && dimension != 0)
                return Nd4j.create(data,
                        new int[]{1, shape[dimension]}
                        , ArrayUtil.removeIndex(stride, 0),
                        offset + index * stride[dimension - 1]);

            else if(dimension == 0)
                return Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index);

            if(size(dimension) == 1) {

                return  Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index);
            }
            else
                return  Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * stride[0]);
        }

        else if(ordering == NDArrayFactory.FORTRAN) {

            if(dimension == shape.length - 1 && dimension != 0) {
                if(size(dimension) == 1)
                    return Nd4j.create(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index);
                return Nd4j.create(data,
                        new int[]{1, shape[dimension]}
                        , ArrayUtil.removeIndex(stride, 0),
                        offset + index * stride[dimension - 1]);

            }
            else {
                if(size(dimension) == 1)
                    return  Nd4j.create(data,
                            new int[]{shape[dimension], 1}
                            , new int[]{stride[dimension], 1},
                            offset + index);

                return  Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * stride[stride.length - 1]);
            }

        }

        throw new IllegalStateException("Illegal ordering..none declared");


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
                s += (float) getScalar(i).element();
                putScalar(i, s);
            }
        }

        else if(dimension == Integer.MAX_VALUE || dimension == shape.length - 1) {
            INDArray flattened = ravel().dup();
            float prevVal = (float) flattened.getScalar(0).element();
            for(int i = 1; i < flattened.length(); i++) {
                float d = prevVal + (float) flattened.getScalar(i).element();
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
        int idx = linearIndex(i);
        data[idx] = value.floatValue();
        return this;
    }

    @Override
    public INDArray putScalar(int[] indexes, Number value) {
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }

        data[ix] = value.floatValue();

        return this;
    }

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    public INDArray eps(Number other) {
        return dup().epsi(other);
    }

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    public INDArray epsi(Number other) {
        INDArray linearView = linearView();
        float otherVal = other.floatValue();

        for(int i = 0; i < linearView.length(); i++) {
            float val = linearView.get(i);
            float diff = Math.abs(val - otherVal);
            if(diff <= Nd4j.EPS_THRESHOLD)
                linearView.putScalar(i,1);
            else
                linearView.putScalar(i,0);

        }
        return this;
    }

    /**
     * epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    public INDArray eps(INDArray other) {
        return dup().epsi(other);
    }

    /**
     * In place epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    public INDArray epsi(INDArray other) {
        INDArray linearView = linearView();
        INDArray otherLinearView = other.linearView();
        for(int i = 0; i < linearView.length(); i++) {
            float val = linearView.get(i);
            float otherVal = otherLinearView.get(i);
            float diff = Math.abs(val - otherVal);
            if(diff <= Nd4j.EPS_THRESHOLD)
                linearView.putScalar(i,1);
            else
                linearView.putScalar(i,0);

        }
        return this;
    }

    @Override
    public INDArray lt(Number other) {
        return dup().lti(other);
    }

    @Override
    public INDArray lti(Number other) {
        return lti(Nd4j.scalar(other));
    }

    @Override
    public INDArray eq(Number other) {
        return dup().eqi(other);
    }

    @Override
    public INDArray eqi(Number other) {
        return eqi(Nd4j.scalar(other));
    }

    @Override
    public INDArray gt(Number other) {
        return dup().gti(other);
    }

    @Override
    public INDArray gti(Number other) {
        return gti(Nd4j.scalar(other));
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
    public INDArray neg() {
        return dup().negi();
    }

    /**
     * Negate each element (in-place).
     */
    @Override
    public INDArray negi() {
        return  Transforms.neg(this);
    }


    @Override
    public INDArray rdiv(Number n, INDArray result) {
        return dup().rdivi(n,result);
    }

    @Override
    public INDArray rdivi(Number n, INDArray result) {
        INDArray linear = linearView();
        INDArray resultLinear = result.linearView();
        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, n.floatValue() / linear.get(i));
        }
        return result;
    }

    @Override
    public INDArray rsub(Number n, INDArray result) {
        return dup().rsubi(n,result);
    }

    @Override
    public INDArray rsubi(Number n, INDArray result) {
        INDArray linear = linearView();
        INDArray resultLinear = result.linearView();
        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, n.floatValue() - linear.get(i));
        }
        return result;
    }

    @Override
    public INDArray div(Number n, INDArray result) {
        return dup().divi(n,result);
    }

    @Override
    public INDArray divi(Number n, INDArray result) {
        INDArray linear = linearView();
        INDArray resultLinear = result.linearView();
        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, linear.get(i) / n.floatValue());
        }
        return result;
    }

    @Override
    public INDArray mul(Number n, INDArray result) {
        return dup().muli(n,result);
    }

    @Override
    public INDArray muli(Number n, INDArray result) {
        INDArray linear = linearView();
        INDArray resultLinear = result.linearView();
        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, linear.get(i) * n.floatValue());
        }
        return result;
    }

    @Override
    public INDArray sub(Number n, INDArray result) {
        return dup().subi(n,result);
    }

    @Override
    public INDArray subi(Number n, INDArray result) {
        INDArray linear = linearView();
        INDArray resultLinear = result.linearView();
        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, linear.get(i) - n.floatValue());
        }
        return result;
    }

    @Override
    public INDArray add(Number n, INDArray result) {
        return dup().addi(n,result);
    }

    @Override
    public INDArray addi(Number n, INDArray result) {
        INDArray linear = linearView();
        INDArray resultLinear = result.linearView();
        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, linear.get(i) + n.floatValue());
        }
        return result;
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
        return getScalar(new int[]{row,column});
    }


    @Override
    public   INDArray dup() {
        INDArray ret = Nd4j.create(shape());
        INDArray linear = linearView();
        INDArray retLinear = ret.linearView();
        for(int i = 0; i < ret.length(); i++) {
            retLinear.putScalar(i,linear.get(i));
        }
        return ret;
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    @Override
    public float get(int... indices) {
        int ix = offset;
        for (int i = 0; i< indices.length; i++)
            ix += indices[i] * stride[i];


        return data[ix];

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
        for (int i = 0; i< indices.length; i++)
            ix += indices[i] * stride[i];


        data[ix] = (float) element.element();
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

        return put(i,j, Nd4j.scalar(element));
    }


    /**
     * Assigns the given matrix (put) to the specified slice
     * @param slice the slice to assign
     * @param put the slice to applyTransformToDestination
     * @return this for chainability
     */
    @Override
    public INDArray putSlice(int slice,INDArray put) {
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
            putScalar(slice, put.get(0));
        else if(put.isVector())
            for(int i = 0; i < put.length(); i++)
                view.putScalar(i, put.get(i));
        else if(put.shape().length == 2)
            for(int i = 0; i < put.rows(); i++)
                for(int j = 0; j < put.columns(); j++)
                    view.put(i,j,put.get(i, j));

        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for(int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i,view.slice(i));

        }

        return this;

    }


    protected void assertSlice(INDArray put,int slice) {
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
                && (shape[0] != 1 && shape[1] != 1));
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
    public INDArray reduce(Ops.DimensionOp op,int dimension) {
        if (isScalar())
            return this;


        if (isVector())
            return Nd4j.scalar(reduceVector(op, this));


        int[] shape = ArrayUtil.removeIndex(this.shape, dimension);

        if (dimension == 0) {
            double[] data2 = new double[ArrayUtil.prod(shape)];
            int dataIter = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for (int offset = this.offset; offset < numTimes; offset++) {
                double reduce = op(dimension, offset, op);
                data2[dataIter++] = reduce;

            }

            return Nd4j.create(data2, shape);
        } else {
            double[] data2 = new double[ArrayUtil.prod(shape)];
            int dataIter = 0;
            //want the milestone to slice[1] and beyond
            int[] sliceIndices = endsForSlices();
            int currOffset = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for (int offset = this.offset; offset < numTimes; offset++) {
                if (dataIter >= data2.length || currOffset >= sliceIndices.length)
                    break;

                //do the operation,, and look for whether it exceeded the current slice
                IterationResult pair = op(dimension, offset, op, sliceIndices[currOffset]);
                //append the result
                double reduce = pair.getResult();
                data2[dataIter++] = reduce;

                //go to next slice and iterate over that
                if (pair.isNextSlice()) {
                    //will update to next step
                    offset = sliceIndices[currOffset];
                    numTimes += sliceIndices[currOffset];
                    currOffset++;
                }

            }

            return Nd4j.create(data2, shape);

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
            INDArray ret = Nd4j.create(new int[]{shape[dimension]});

            for(int j = offset; count < this.shape[dimension]; j+= this.stride[dimension]) {
                double d = data[j];
                ret.putScalar(count++, d);
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
            double d = data[j];
            dim[count++] = d;
            if(j >= currOffsetForSlice)
                newSlice = true;
        }

        return new IterationResult(reduceVector(op, Nd4j.create(dim)),newSlice);
    }


    //getFromOrigin one result along one dimension based on the given offset
    private double op(int dimension, int offset, Ops.DimensionOp op) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op, Nd4j.create(dim));
    }

    @Override
    public int index(int row, int column) {
        if(!isMatrix()) {
            if(isColumnVector()) {
                int idx = linearIndex(row);
                return idx;
            }
            else if (isRowVector()) {
                int idx = linearIndex(column);
                return idx;
            }
            else
                throw new IllegalStateException("Unable to getFromOrigin row/column from a non matrix");
        }



        return offset + (row *  stride[0]  + column * stride[1]);
    }



    protected INDArray newShape(int[] newShape,char ordering) {
        if(Arrays.equals(newShape,this.shape()))
            return this;

        else if(Shape.isVector(newShape) && isVector()) {
            if(isRowVector() && Shape.isColumnVectorShape(newShape)) {
                return Nd4j.create(data,newShape,new int[]{stride[0],1},offset);
            }
            else if(isColumnVector() && Shape.isRowVectorShape(newShape)) {
                return Nd4j.create(data,newShape,new int[]{stride[1]},offset);

            }
        }

        INDArray newCopy = this;
        int[] newStrides = null;
        //create a new copy of the ndarray
        if(shape().length > 1 &&  ((ordering == NDArrayFactory.C && this.ordering != NDArrayFactory.C) ||
                (ordering == NDArrayFactory.FORTRAN && this.ordering != NDArrayFactory.FORTRAN))) {
            newStrides = noCopyReshape(newShape,ordering);
            if(newStrides == null) {
                newCopy = Nd4j.create(shape(),ordering);
                for(int i = 0; i < vectorsAlongDimension(0); i++) {
                    INDArray copyFrom = vectorAlongDimension(i,0);
                    INDArray copyTo = newCopy.vectorAlongDimension(i,0);
                    for(int j = 0; j < copyFrom.length(); j++) {
                        copyTo.putScalar(j,copyFrom.get(i));
                    }
                }
            }




        }

        //needed to copy data
        if(newStrides == null)
            newStrides = Nd4j.getStrides(newShape,ordering);
        if(this instanceof  IComplexNDArray)
            return Nd4j.createComplex(newCopy.data(),newShape,newStrides,offset);
        return Nd4j.create(newCopy.data(),newShape,newStrides,offset);


    }

    /**
     * Return the new strides based on the shape and ordering or null
     * if we can't do a reshape
     * @param newShape the new shape
     * @param ordering the ordering of the new shape
     * @return the new strides or null if we can't reshape
     */
    protected int[] noCopyReshape(int[] newShape,char ordering) {
        List<Integer> oldDims = new ArrayList<>();
        List<Integer> oldStrides = new ArrayList<>();
        for (int i = 0; i < shape.length; i++) {
            if (size(i) != 1) {
                oldDims.add(size(i));
                oldStrides.add(stride[i]);
            }
        }

        int np = 1;
        for (int ni = 0; ni < newShape.length; ni++) {
            np *= newShape[ni];
        }

        int op = 1;
        for (int oi = 0; oi < oldDims.size(); oi++) {
            op *= oldDims.get(oi);
        }
        if (np != op) {
        /* different total sizes; no hope */
            return null;
        }

        if (np == 0) {
        /* the current code does not handle 0-sized arrays, so give up */
            return null;
        }


          /* oi to oj and ni to nj give the axis ranges currently worked with */
        int oi = 0;
        int oj = 1;
        int ni = 0;
        int nj = 1;

        List<Integer> newStrides = new ArrayList<>();
        while (ni < newShape.length && oi < oldDims.size()) {
            np = newShape[ni];
            op = oldDims.get(oi);

            while (np != op) {
                if (np < op)
                    np *= newShape[nj++];
                else
                    op *= oldDims.get(oj++);
            }

                    /* Check whether the original axes can be combined */
            for (int ok = oi; ok < oj - 1; ok++) {
                if (ordering == NDArrayFactory.FORTRAN) {
                    if (oldStrides.get(ok + 1) != oldDims.get(ok) * oldStrides.get(ok)) {
                     /* not contiguous enough */
                        return null;
                    }
                } else {
                /* C order */
                    if (oldStrides.get(ok) != oldDims.get(ok + 1) * oldStrides.get(ok + 1)) {
                    /* not contiguous enough */
                        return null;
                    }
                }
            }


             /* Calculate new strides for all axes currently worked with */
            if (ordering == NDArrayFactory.FORTRAN) {
                newStrides.set(ni, oldStrides.get(oi));
                for (int nk = ni + 1; nk < nj; nk++) {
                    newStrides.set(nk, newStrides.get(nk - 1) * newShape[nk - 1]);
                }
            } else {
            /* C order */
                newStrides.set(nj - 1, oldStrides.get(oj - 1));

                for (int nk = nj - 1; nk > ni; nk--) {
                    newStrides.set(nk - 1, newStrides.get(nk) * newShape[nk]);
                }
            }
            ni = nj++;
            oi = oj++;
        }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
        int lastStride;
        if (ni >= 1) {
            lastStride = newStrides.get(ni - 1);
        } else {
            lastStride = length;
        }
        if (ordering == NDArrayFactory.FORTRAN) {
            lastStride *= newShape[ni - 1];
        }
        for (int nk = ni; nk < newShape.length; nk++) {
            newStrides.set(nk, lastStride);
        }


        return ArrayUtil.toArray(newStrides);


    }

    protected float reduceVector(Ops.DimensionOp op,INDArray vector) {

        switch(op) {
            case SUM:
                return (float) vector.sum(Integer.MAX_VALUE).element();
            case MEAN:
                return (float) vector.mean(Integer.MAX_VALUE).element();
            case MIN:
                return (float) vector.min(Integer.MAX_VALUE).element();
            case MAX:
                return (float) vector.max(Integer.MAX_VALUE).element();
            case NORM_1:
                return (float) vector.norm1(Integer.MAX_VALUE).element();
            case NORM_2:
                return (float) vector.norm2(Integer.MAX_VALUE).element();
            case NORM_MAX:
                return (float) vector.normmax(Integer.MAX_VALUE).element();
            default: throw new IllegalArgumentException("Illegal operation");
        }
    }



    /**
     * Returns the squared (Euclidean) distance.
     */
    @Override
    public float squaredDistance(INDArray other) {
        float sd = 0.0f;
        for (int i = 0; i < length; i++) {
            float d =  get(i) -  other.get(i);
            sd += d * d;
        }
        return sd;
    }

    /**
     * Returns the (euclidean) distance.
     */
    @Override
    public float distance2(INDArray other) {
        return  (float) Math.sqrt(squaredDistance(other));
    }

    /**
     * Returns the (1-norm) distance.
     */
    @Override
    public float distance1(INDArray other) {
        return other.sub(this).sum(Integer.MAX_VALUE).get(0);
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        INDArray get = get(indices);
        INDArray linear = get.linearView();
        if(element.isScalar()) {
            for(int i = 0; i < linear.length(); i++) {
                linear.putScalar(i,element.get(0));
            }
        }

        if(Shape.shapeEquals(element.shape(),get.shape()) || element.length() == get.length()) {
            INDArray elementLinear = element.linearView();

            for(int i = 0; i < linear.length(); i++) {
                linear.putScalar(i,elementLinear.get(i));
            }
        }


        return this;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, Number element) {
        return put(indices, Nd4j.scalar(element));
    }


    /**
     * Iterate over every column of every slice
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllColumns(SliceOp op) {
        if(isVector())
            op.operate(this);
        else if(isMatrix()) {
            for(int i = 0; i < columns(); i++) {
                op.operate(getColumn(i));
            }
        }

        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).iterateOverAllRows(op);
            }
        }
    }


    /**
     * Iterate over every row of every slice
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllRows(SliceOp op) {
        if(isVector())
            op.operate(this);
        else if(isMatrix()) {
            for(int i = 0; i < rows(); i++) {
                op.operate(getRow(i));
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
    public INDArray swapAxes(int dimension,int with) {
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




    @Override
    public float[] data() {
        return data;
    }

    @Override
    public void setData(float[] data) {
        this.data = data;
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
    public INDArray subArray(int[] offsets, int[] shape,int[] stride) {
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


        if(isVector()) {
            //offsets should be 1
            return Nd4j.create(data, shape, stride, offsets[0]);
        }


        int offset = this.offset + ArrayUtil.dotProduct(offsets, stride);

        return Nd4j.create(
                data
                , Arrays.copyOf(shape, shape.length)
                , stride
                , offset, ordering
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
                    INDArray result = (INDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = (float) result.getScalar(i).element();
                    }
                }
            }
        }

        else if(isVector()) {
            if(dimension == 0) {
                DimensionSlice slice = this.vectorForDimensionAndOffset(0,0);
                op.operate(slice);
                if(modify && slice.getIndices() != null) {
                    INDArray result = (INDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = (float) result.getScalar(i).element();
                    }
                }
            }
            else if(dimension == 1) {
                for(int i = 0; i < length; i++) {
                    DimensionSlice slice = vectorForDimensionAndOffset(dimension,i);
                    op.operate(slice);
                    if(modify && slice.getIndices() != null) {
                        INDArray result = (INDArray) slice.getResult();
                        for(int j = 0; j < slice.getIndices().length; j++) {
                            data[slice.getIndices()[j]] = (float) result.getScalar(j).element();
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




    @Override
    public void setStride(int[] stride) {
        this.stride = stride;
    }








    protected void initShape(int[] shape) {
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
                //weird case here: when its fortran contiguous, we have scencrios
                //where we need to retain the non zero stride of the ndarray.
                //this is in response to a strange scenario that happens in subArray
                //hopefully we can work out something better eventually
                if(stride != null && ordering == NDArrayFactory.FORTRAN)
                    this.stride = new int[]{ArrayUtil.nonOneStride(this.stride)};

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

        //null character
        if(this.ordering == '\u0000')
            this.ordering = Nd4j.order();

        this.length = ArrayUtil.prod(this.shape);
        if(this.stride == null) {
            if(ordering == NDArrayFactory.FORTRAN)
                this.stride = ArrayUtil.calcStridesFortran(shape);
            else
                this.stride = ArrayUtil.calcStrides(this.shape);
        }

        //recalculate stride: this should only happen with row vectors
        if(this.stride.length != this.shape.length) {
            if(ordering == NDArrayFactory.FORTRAN)
                this.stride = ArrayUtil.calcStridesFortran(this.shape);
            else
                this.stride = ArrayUtil.calcStrides(this.shape);
        }


    }



    @Override
    public INDArray getScalar(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return Nd4j.scalar(data[idx]);
    }




    protected void asserColumnVector(INDArray column) {
        assert column.isColumnVector() || column.columns() == columns() && column.rows() == 1: "Must only add a row vector";
        assert column.length() == rows() || column.columns() == columns() && column.rows() == 1: "Illegal row vector must have the same length as the number of rows in this ndarray";

    }

    /**
     * Do a row wise op (a,s,m,d)
     * a : add
     * s : subtract
     * m : multiply
     * d : divide
     * @param columnVector the column  vector
     * @param operation the operation
     * @return
     */
    protected INDArray doColumnWise(INDArray columnVector,char operation) {
        asserColumnVector(columnVector);
        INDArray columnBroadcasted = columnVector.broadcast(shape());
        INDArray columnBroadCastedLinear = columnBroadcasted.linearViewColumnOrder();


        switch(operation) {
            case 'a' : linearView().addi(columnBroadCastedLinear); break;
            case 's' : linearView().subi(columnBroadCastedLinear); break;
            case 'm' : linearView().muli(columnBroadCastedLinear); break;
            case 'd' : linearView().divi(columnBroadCastedLinear); break;
        }


        return this;



    }


    protected void assertRowVector(INDArray rowVector) {
        assert rowVector.isRowVector() || rowVector.rows() == rows() && rowVector.columns() == 1: "Must only add a row vector";
        assert rowVector.length() == columns() || rowVector.rows() == rows() && rowVector.columns() == 1: "Illegal row vector must have the same length as the number of rows in this ndarray";

    }

    /**
     * Do a row wise op (a,s,m,d)
     * a : add
     * s : subtract
     * m : multiply
     * d : divide
     * @param rowVector the row vector
     * @param operation the operation
     * @return
     */
    protected INDArray doRowWise(INDArray rowVector,char operation) {
        assertRowVector(rowVector);
        INDArray broadcasted = rowVector.broadcast(shape());
        INDArray broadCastedLinear = broadcasted.linearView();
        switch(operation) {
            case 'a' : linearView().addi(broadCastedLinear); break;
            case 's' : linearView().subi(broadCastedLinear); break;
            case 'm' : linearView().muli(broadCastedLinear); break;
            case 'd' : linearView().divi(broadCastedLinear); break;
        }

        return this;

    /*    assertRowVector(rowVector);
        if(rowVector.rows() == rows() && rowVector.columns() == 1) {
            for(int i = 0; i < rows(); i++) {
                switch(operation) {
                    case 'a' : getRow(i).addi(rowVector.get(i)); break;
                    case 's' : getRow(i).subi(rowVector.get(i)); break;
                    case 'm' : getRow(i).muli(rowVector.get(i)); break;
                    case 'd' : getRow(i).divi(rowVector.get(i)); break;
                }

            }
        }
        else {
            for(int j = 0; j< rows(); j++) {
                switch(operation) {
                    case 'a' : getRow(j).addi(rowVector); break;
                    case 's' : getRow(j).subi(rowVector); break;
                    case 'm' : getRow(j).muli(rowVector); break;
                    case 'd' : getRow(j).divi(rowVector); break;
                }
            }
        }

        return this;*/
    }


    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, INDArray element) {
        if(element == null)
            throw new IllegalArgumentException("Unable to insert null element");
        assert element.isScalar() : "Unable to insert non scalar element";
        int idx = linearIndex(i);
        data[idx] = (float) element.element();
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector,'d');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviRowVector(INDArray rowVector) {
        return doRowWise(rowVector,'d');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divRowVector(INDArray rowVector) {
        return dup().diviRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector,'m');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulColumnVector(INDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliRowVector(INDArray rowVector) {
        return doRowWise(rowVector,'m');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector,'s');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subColumnVector(INDArray columnVector) {
        return dup().subiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiRowVector(INDArray rowVector) {
        return doRowWise(rowVector,'s');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subRowVector(INDArray rowVector) {
        return dup().subiRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector,'a');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addColumnVector(INDArray columnVector) {
        return dup().addiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiRowVector(INDArray rowVector) {
        return doRowWise(rowVector,'a');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other) {
        int[] shape = {rows(),other.columns()};
        char order = Nd4j.factory().order();
        boolean switchedOrder = false;
        if(order != NDArrayFactory.FORTRAN) {
            Nd4j.factory().setOrder(NDArrayFactory.FORTRAN);
            switchedOrder = true;
        }

        INDArray result = Nd4j.create(shape);

        if(switchedOrder && order != NDArrayFactory.FORTRAN)
            Nd4j.factory().setOrder(NDArrayFactory.C);

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
    public INDArray mmul(INDArray other, INDArray result) {
        return dup().mmuli(other,result);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public INDArray div(INDArray other) {
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
    public INDArray div(INDArray other, INDArray result) {
        return dup().divi(other,result);
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public INDArray mul(INDArray other) {
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
    public INDArray mul(INDArray other, INDArray result) {
        return dup().muli(other,result);
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray sub(INDArray other) {
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
    public INDArray sub(INDArray other, INDArray result) {
        return dup().subi(other,result);
    }

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray add(INDArray other) {
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
    public INDArray add(INDArray other, INDArray result) {
        return dup().addi(other,result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other) {
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
    public INDArray mmuli(INDArray other, INDArray result) {
        INDArray otherArray =  other;
        INDArray resultArray =  result;

        LinAlgExceptions.assertMultiplies(this,other);

        if (other.isScalar()) {
            return muli(otherArray.get(0), resultArray);
        }
        if (isScalar()) {
            return otherArray.muli(get(0), resultArray);
        }

        /* check sizes and resize if necessary */
        //assertMultipliesWith(other);


        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            INDArray temp = Nd4j.create(resultArray.shape(), ArrayUtil.calcStridesFortran(resultArray.shape()));

            if (otherArray.columns() == 1) {
                Nd4j.getBlasWrapper().gemv(1.0f, this, otherArray, 0.0f, temp);
            } else {
                Nd4j.getBlasWrapper().gemm(1.0f, this, otherArray, 0.0f, temp);
            }

            Nd4j.getBlasWrapper().copy(temp, resultArray);


        } else {
            if (otherArray.columns() == 1)
                Nd4j.getBlasWrapper().gemv(1.0f, this, otherArray, 0.0f, resultArray);
            else
                Nd4j.getBlasWrapper().gemm(1.0f, this, otherArray, 0.0f, resultArray);

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
    public INDArray divi(INDArray other) {
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
    public INDArray divi(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return divi(other.get(0), result);
        }
        if (isScalar()) {
            return other.divi(get(0), result);
        }

        INDArray otherLinear = other.linearView();
        INDArray resultLinear = result.linearView();
        INDArray linearView = linearView();

        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, linearView.get(i) / otherLinear.get(i));
        }
        return result;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public INDArray muli(INDArray other) {
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
    public INDArray muli(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return muli(other.get(0), result);
        }
        if (isScalar()) {
            return other.muli(get(0), result);
        }

        INDArray otherLinear = other.linearView();
        INDArray resultLinear = result.linearView();
        INDArray linearView = linearView();

        for (int i = 0; i < length; i++) {
            resultLinear.putScalar(i, linearView.get(i) * otherLinear.get(i));
        }
        return result;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray subi(INDArray other) {
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
    public INDArray subi(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return subi(other.get(0), result);
        }
        if (isScalar()) {
            return other.rsubi(get(0), result);
        }


        if (result == this) {
            Nd4j.getBlasWrapper().axpy(-1.0f, other, result);
        } else if (result == other) {
            Nd4j.getBlasWrapper().scal(-1.0f, result);
            Nd4j.getBlasWrapper().axpy(1.0f, this, result);
        } else {
            Nd4j.getBlasWrapper().copy(this, result);
            Nd4j.getBlasWrapper().axpy(-1.0f, other, result);
        }
        return result;
    }

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray addi(INDArray other) {
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
    public INDArray addi(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return result.addi(other.get(0),result);
        }
        if (isScalar()) {
            return other.addi(get(0), result);
        }


        if (result == this) {
            Nd4j.getBlasWrapper().axpy(1.0f, other, result);
        } else if (result == other) {
            Nd4j.getBlasWrapper().axpy(1.0f, this, result);
        } else {
            /*SimpleBlas.copy(this, result);
            SimpleBlas.axpy(1.0, other, result);*/
            INDArray resultLinear = result.linearView();
            INDArray otherLinear = other.linearView();
            INDArray linear = linearView();
            for(int i = 0; i < resultLinear.length(); i++) {
                resultLinear.putScalar(i,otherLinear.get(i) + linear.get(i));
            }

        }

        return result;
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
            return Nd4j.scalar(Ops.normmax(this));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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
        INDArray one = linearView();
        for(int i = 0; i < one.length(); i++)
            one.put(i, Nd4j.scalar(value.doubleValue()));
        return this;
    }

    @Override
    public int linearIndex(int i) {
        int realStride = stride[0];
        int idx = offset + i * realStride;

        if(data != null && idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
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
    public INDArray slice(int slice) {

        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1) {
            if(size(0) == 1)
                return Nd4j.create(data, ArrayUtil.empty(), ArrayUtil.empty(), offset + slice);
            else
                return Nd4j.create(data, ArrayUtil.empty(), ArrayUtil.empty(), offset + slice * stride[0]);


        }


        //slice of a matrix is a vector
        else if (shape.length == 2) {
            if(size(0) == 1)  {
                INDArray slice2 = Nd4j.create(
                        data,
                        ArrayUtil.of(shape[1]),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset + slice, ordering
                );
                return slice2;
            }
            else {
                INDArray slice2 = Nd4j.create(
                        data,
                        ArrayUtil.of(shape[1]),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset + slice * stride[0], ordering
                );
                return slice2;
            }



        }

        else {
            int offset = this.offset + (slice * stride[0]);
            if(size(0) == 1) {

                INDArray slice2 = Nd4j.create(data,
                        Arrays.copyOfRange(shape, 1, shape.length),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset, ordering);
                return slice2;
            }

            else {
                INDArray slice2 = Nd4j.create(data,
                        Arrays.copyOfRange(shape, 1, shape.length),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset, ordering);
                return slice2;
            }




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
    public INDArray slice(int slice, int dimension) {
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

        INDArray slice2 = Nd4j.create(data,
                ArrayUtil.removeIndex(shape, dimension),
                ArrayUtil.removeIndex(stride, dimension),
                offset + slice * stride[dimension], ordering);
        return slice2;
    }



    /**
     * Fetch a particular number on a multi dimensional scale.
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    @Override
    public INDArray getScalar(int... indexes) {
        int ix = offset;


        for (int i = 0; i < indexes.length; i++) {
            ix += indexes[i] * stride[i];
        }

        if(ix >= data.length)
            throw new IllegalArgumentException("Illegal index " + Arrays.toString(indexes));

        return Nd4j.scalar(data[ix]);
    }


    @Override
    public INDArray rdiv(Number n) {
        return dup().rdivi(n);
    }

    @Override
    public INDArray rdivi(Number n) {
        return rdivi(Nd4j.valueArrayOf(shape(), n.doubleValue()));
    }

    @Override
    public INDArray rsub(Number n) {
        return dup().rsubi(n);
    }

    @Override
    public INDArray rsubi(Number n) {
        return rsubi(Nd4j.valueArrayOf(shape(), n.doubleValue()));
    }

    @Override
    public INDArray div(Number n) {
        return dup().divi(n);
    }

    @Override
    public INDArray divi(Number n) {
        return divi(n,this);
    }

    @Override
    public INDArray mul(Number n) {
        return dup().muli(n);
    }

    @Override
    public INDArray muli(Number n) {
        return muli(n,this);
    }

    @Override
    public INDArray sub(Number n) {
        return dup().subi(n);
    }

    @Override
    public INDArray subi(Number n) {
        return subi(n,this);
    }

    @Override
    public INDArray add(Number n) {
        return dup().addi(n);
    }

    @Override
    public INDArray addi(Number n) {
        return addi(n,this);
    }


    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    public INDArray repmat(int[] shape) {
        int[] newShape = ArrayUtil.copy(shape());
        assert shape.length <= newShape.length : "Illegal shape: The passed in shape must be <= the current shape length";
        for(int i = 0; i < shape.length; i++)
            newShape[i] *= shape[i];
        INDArray result = Nd4j.create(newShape);
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

        INDArray ret =  result;
        return  ret;
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
    public INDArray putRow(int row, INDArray toPut) {
        assert toPut.isVector() && toPut.length() == columns : "Illegal length for row " + toPut.length() + " should have been " + columns;
        INDArray r = getRow(row);
        for(int i = 0; i < r.length(); i++)
            r.putScalar(i,toPut.get(i));
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
    public INDArray putColumn(int column, INDArray toPut) {
        assert toPut.isVector() && toPut.length() == rows : "Illegal length for row " + toPut.length() + " should have been " + columns;
        INDArray r = getColumn(column);
        for(int i = 0; i < r.length(); i++)
            r.putScalar(i,toPut.get(i));

        return this;
    }



    @Override
    public float get(int i) {
        int idx = linearIndex(i);
        if(idx < 0)
            throw new IllegalStateException("Illegal index " + i);
        return data[idx];
    }

    @Override
    public float get(int i, int j) {
        int idx = index(i,j);
        return data[idx];
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
     * Return transposed copy of this matrix.
     */
    @Override
    public INDArray transpose() {
        if(isRowVector())
            return Nd4j.create(data, new int[]{shape[0], 1}, offset);
        else if(isColumnVector())
            return Nd4j.create(data, new int[]{shape[0]}, offset);
        if(isMatrix()) {
            INDArray reverse = Nd4j.create(new int[]{shape[1],shape[0]});
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    reverse.put(j, i, get(i, j));
                }
            }

            return reverse;
        }

        INDArray ret = permute(ArrayUtil.range(shape.length -1,-1));
        return ret;
    }


    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     * @param shape
     * @return
     */
    @Override
    public INDArray reshape(int[] shape) {
        assert ArrayUtil.prod(shape) == ArrayUtil.prod(this.shape()) : "Illegal reshape must be of same length as data";
        return newShape(shape,ordering);
    }




    @Override
    public void checkDimensions(INDArray other) {
        assert Arrays.equals(shape,other.shape()) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape());
        assert Arrays.equals(stride,other.stride()) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride());
        assert offset == other.offset() : "Offset of this array is " + offset + " but other was " + other.offset();

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
            return Nd4j.scalar(Ops.prod(this));
        }

        else if(isVector()) {
            return  prod(Integer.MAX_VALUE);
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
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
        if(dimension == Integer.MAX_VALUE || isVector()) {
            return Nd4j.scalar(Ops.mean(this));

        }
        else if(isVector()) {
            return Nd4j.scalar((double) sum(Integer.MAX_VALUE).element() / length());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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

    /**
     * Returns the overall variance of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray var(int dimension) {
        if(dimension == Integer.MAX_VALUE || isVector()) {
            return Nd4j.scalar(Ops.var(this));
        }

        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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
        if(dimension == Integer.MAX_VALUE || isVector()) {
            return Nd4j.scalar(Ops.max(this));
        }


        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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
        if(dimension == Integer.MAX_VALUE || isVector()) {
            return Nd4j.scalar(Ops.min(this));
        }


        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr =  Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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
        if(dimension == Integer.MAX_VALUE || isVector()) {
            return Nd4j.scalar(Ops.sum(this));
        }

        else if(isVector()) {
            return  sum(Integer.MAX_VALUE);
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray norm1(int dimension) {
        if(isVector() || dimension == Integer.MAX_VALUE) {
            return Nd4j.scalar(Ops.norm1(this));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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




    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    @Override
    public INDArray std(int dimension) {
        if(isVector() || dimension == Integer.MAX_VALUE) {
            return Nd4j.scalar(Ops.std(this));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    public INDArray norm2(int dimension) {
        if(isVector() || dimension == Integer.MAX_VALUE) {
            return  Nd4j.scalar(Ops.norm2(this));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final INDArray arr = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
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
     * Number of columns (shape[1]), throws an exception when
     * called when not 2d
     * @return the number of columns in the array (only 2d)
     */
    @Override
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
    @Override
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
    public INDArray ravel() {



        INDArray ret = Nd4j.create(length, ordering);

        int dimension = shape.length == 2 ? 1 : shape.length;
        int count = 0;
        for(int i = 0; i < vectorsAlongDimension(dimension); i++) {
            INDArray vec = vectorAlongDimension(i,dimension);
            for(int j = 0; j < vec.length(); j++) {
                ret.putScalar(count++,vec.get(j));
            }
        }

        return ret;

    }

    /**
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    @Override
    public void sliceVectors(List<INDArray> list) {
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
    public INDArray reshape(int newRows, int newColumns) {
        assert newRows * newColumns == length : "Illegal new shape " + newRows + " x " + newColumns;
        return reshape(new int[]{newRows,newColumns});
    }

    /**
     * Get the specified column
     *
     * @param c
     */
    @Override
    public INDArray getColumn(int c) {
        if(shape.length == 2) {
            if(ordering == NDArrayFactory.C) {
                INDArray ret = Nd4j.create(
                        data,
                        new int[]{shape[0], 1},
                        new int[]{stride[0], 1},
                        offset + c, ordering
                );

                return ret;
            }
            else {
                INDArray ret = Nd4j.create(
                        data,
                        new int[]{shape[0],1},
                        new int[]{stride[0],1},
                        offset + c * rows(), ordering
                );

                return ret;
            }

        }
        else if(isColumnVector() && c == 0)
            return this;


        else
            throw new IllegalArgumentException("Unable to get scalar column of non 2d matrix");
    }


    /**
     * Get whole rows from the passed indices.
     *
     * @param rindices
     */
    @Override
    public INDArray getRows(int[] rindices) {
        INDArray rows = Nd4j.create(rindices.length, columns());
        for(int i = 0; i < rindices.length; i++) {
            rows.putRow(i,getRow(rindices[i]));
        }
        return  rows;
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
        //fill in to match the rest of the dimensions: aka grab all the content
        //in the dimensions not filled in
        //also prune indices greater than the shape to be the shape instead

        indexes = Indices.adjustIndices(shape(),indexes);


        int[] offsets =  Indices.offsets(indexes);
        int[] shape = Indices.shape(shape(),indexes);
        int[] strides =  ArrayUtil.copy(stride());

        return subArray(offsets,shape,strides);
    }




    /**
     * Get whole columns from the passed indices.
     *
     * @param cindices
     */
    @Override
    public INDArray getColumns(int[] cindices) {
        INDArray rows = Nd4j.create(rows(), cindices.length);
        for(int i = 0; i < cindices.length; i++) {
            rows.putColumn(i,getColumn(cindices[i]));
        }
        return rows;
    }

    /**
     * Get a copy of a row.
     *
     * @param r
     */
    @Override
    public INDArray getRow(int r) {
        if(shape.length == 2) {
            if(ordering == NDArrayFactory.C) {
                INDArray ret = Nd4j.create(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + r * columns(),
                        ordering
                );

                return ret;
            }
            else {
                INDArray ret  = Nd4j.create(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + r,
                        ordering
                );
                return ret;
            }


        }

        else if(isRowVector() && r == 0)
            return this;


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
        INDArray n = null;

        if(!(o instanceof  INDArray))
            return false;

        if(n == null)
            n =  (INDArray) o;

        //epsilon equals
        if(isScalar() && n.isScalar()) {
            float val = (float) element();
            float val2 = (float) n.element();
            return Math.abs(val - val2) < 1e-6;
        }
        else if(isVector() && n.isVector()) {
            for(int i = 0; i < length; i++) {
                float curr = get(i);
                float comp = n.get(i);
                if(Math.abs(curr - comp) > 1e-3)
                    return false;
            }

            return true;

        }


        if(!Shape.shapeEquals(shape(),n.shape()))
            return false;



        if(slices() != n.slices())
            return false;

        for (int i = 0; i < slices(); i++) {
            INDArray slice = slice(i);
            INDArray nSlice = n.slice(i);

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
    @Override
    public int[] stride() {
        return stride;
    }


    @Override
    public int offset() {
        return offset;
    }

    @Override
    public char ordering() {
        return ordering;
    }

    /**
     * Returns the size of this array
     * along a particular dimension
     * @param dimension the dimension to return from
     * @return the shape of the specified dimension
     */
    @Override
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
    public INDArray broadcast(int[] shape) {
        int dims = this.shape.length;
        int targetDimensions = shape.length;


        if (targetDimensions < dims) {
            throw new IllegalArgumentException("Invalid shape to broad cast " + Arrays.toString(shape));
        }
        else if(isColumnVector()) {
            int n = shape[0];
            INDArray ret = Nd4j.create(shape);
            for(int i = 0; i < n; i++) {
                ret.putColumn(i,this);
            }
            return ret;
        }


        //edge case where column vector looks like a matrix
        else if (dims == targetDimensions && !isColumnVector()) {
            if(isVector())
                return this;
            if (Shape.shapeEquals(shape, this.shape()))
                return this;
            throw new IllegalArgumentException("Invalid shape to broad cast " + Arrays.toString(shape));
        }
        else {
            int n = shape[0];
            INDArray s = broadcast(Arrays.copyOfRange(shape, 1, targetDimensions));
            return Nd4j.repeat(s,n);
        }
    }


    /**
     * Dimshuffle: an extension of permute that adds the ability
     * to broadcast various dimensions.
     *
     * See theano for more examples.
     * This will only accept integers and xs.
     * <p/>
     * An x indicates a dimension should be broadcasted rather than permuted.
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public INDArray dimShuffle(Object[] rearrange,int[] newOrder,boolean[] broadCastable) {
        assert broadCastable.length == shape.length : "The broadcastable dimensions must be the same length as the current shape";

        boolean broadcast = false;
        Set<Object> set = new HashSet<>();
        for(int i = 0; i < rearrange.length; i++) {
            set.add(rearrange[i]);
            if(rearrange[i] instanceof Integer) {
                Integer j = (Integer) rearrange[i];
                if(j >= broadCastable.length)
                    throw new IllegalArgumentException("Illegal dimension, dimension must be < broadcastable.length (aka the real dimensions");
            }
            else if(rearrange[i] instanceof Character) {
                Character c = (Character) rearrange[i];
                if(c != 'x')
                    throw new IllegalArgumentException("Illegal input: Must be x");
                broadcast = true;

            }
            else
                throw new IllegalArgumentException("Only characters and integers allowed");
        }

        //just do permute
        if(!broadcast) {
            int[] ret = new int[rearrange.length];
            for(int i = 0; i < ret.length; i++)
                ret[i] = (Integer) rearrange[i];
            return permute(ret);
        }

        else {
            List<Integer> drop = new ArrayList<>();
            for(int i = 0; i < broadCastable.length; i++) {
                if(!set.contains(i)) {
                    if(broadCastable[i])
                        drop.add(i);
                    else
                        throw new IllegalArgumentException("We can't drop the given dimension because its not broadcastable");
                }

            }


            //list of dimensions to keep
            int[] shuffle = new int[broadCastable.length];
            int count = 0;
            for(int i = 0; i < rearrange.length; i++) {
                if(rearrange[i] instanceof Integer) {
                    shuffle[count++] = (Integer) rearrange[i];
                }
            }



            List<Integer> augment = new ArrayList<>();
            for(int i = 0; i < rearrange.length; i++) {
                if(rearrange[i] instanceof Character)
                    augment.add(i);
            }

            Integer[] augmentDims = augment.toArray(new Integer[1]);

            count = 0;

            int[] newShape = new int[shuffle.length + drop.size()];
            for(int i = 0; i < newShape.length; i++) {
                if(i < shuffle.length) {
                    newShape[count++] = shuffle[i];
                }
                else
                    newShape[count++] = drop.get(i);
            }


            INDArray ret = permute(newShape);
            List<Integer> newDims = new ArrayList<>();
            int[] shape = Arrays.copyOfRange(ret.shape(),0,shuffle.length);
            for(int i = 0; i < shape.length; i++) {
                newDims.add(shape[i]);
            }

            for(int i = 0; i <  augmentDims.length; i++) {
                newDims.add(augmentDims[i],1);
            }

            int[] toReshape = ArrayUtil.toArray(newDims);


            ret = ret.reshape(toReshape);
            return ret;

        }


    }

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.htsliceml
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public INDArray permute(int[] rearrange) {
        if(rearrange.length != shape.length)
            return dup();

        checkArrangeArray(rearrange);

        int[] newShape = doPermuteSwap(shape,rearrange);
        int[] newStride = doPermuteSwap(stride,rearrange);
        return  Nd4j.create(data, newShape, newStride, offset, ordering);



    }



    protected void copyRealTo(INDArray arr) {
        INDArray flattened = linearView();
        INDArray arrLinear = arr.linearView();
        for(int i = 0; i < flattened.length(); i++) {
            arrLinear.putScalar(i,flattened.get(i));
        }

    }

    protected int[] doPermuteSwap(int[] shape,int[] rearrange) {
        int[] ret = new int[shape.length];
        for(int i = 0; i < shape.length; i++) {
            ret[i] = shape[rearrange[i]];
        }
        return ret;
    }


    protected void checkArrangeArray(int[] arr) {
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
                sb.append(get(i));
                if(i < length - 1)
                    sb.append(" ,");
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
                    sb.append(" ,");

            }
        }
        sb.append("]\n");
        return sb.toString();
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
        return data[offset];
    }





    @Override
    public IComplexNDArray rdiv(IComplexNumber n) {
        return dup().rdivi(n);
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n) {
        return rdivi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n) {
        return dup().rsubi(n);
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n) {
        return rsubi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray div(IComplexNumber n) {
        return dup().divi(n);
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n) {
        return divi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray mul(IComplexNumber n) {
        return dup().muli(n);
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n) {
        return muli(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray sub(IComplexNumber n) {
        return dup().subi(n);
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n) {
        return subi(n,Nd4j.createComplex(shape()));
    }

    @Override
    public IComplexNDArray add(IComplexNumber n) {
        return dup().addi(n);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n) {
        return addi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray rdiv(IComplexNumber n, IComplexNDArray result) {
        return dup().rdivi(n,result);
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result) {
        return Nd4j.createComplex(this).rdivi(n,result);

    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n, IComplexNDArray result) {
        return dup().rsubi(n,result);
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result) {
        return Nd4j.createComplex(this).rsubi(n,result);
    }

    @Override
    public IComplexNDArray div(IComplexNumber n, IComplexNDArray result) {
        return dup().divi(n,result);
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n, IComplexNDArray result) {
        return Nd4j.createComplex(this).divi(n,result);

    }

    @Override
    public IComplexNDArray mul(IComplexNumber n, IComplexNDArray result) {
        return dup().muli(n,result);
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n, IComplexNDArray result) {
        return Nd4j.createComplex(this).muli(n,result);

    }

    @Override
    public IComplexNDArray sub(IComplexNumber n, IComplexNDArray result) {
        return dup().subi(n,result);
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n, IComplexNDArray result) {
        return Nd4j.createComplex(this).subi(n,result);

    }

    @Override
    public IComplexNDArray add(IComplexNumber n, IComplexNDArray result) {
        return dup().addi(n,result);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n, IComplexNDArray result) {
        return Nd4j.createComplex(this).addi(n,result);

    }
}