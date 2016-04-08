/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.complex;


import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.nd4j.linalg.api.shape.Shape;

import java.util.*;

import static org.nd4j.linalg.util.ArrayUtil.calcStrides;
import static org.nd4j.linalg.util.ArrayUtil.calcStridesFortran;


/**
 * ComplexNDArray for complex numbers.
 * <p/>
 * <p/>
 * Note that the indexing scheme for a complex ndarray is 2 * length
 * not length.
 * <p/>
 * The reason for this is the fact that imaginary components have
 * to be stored alongside realComponent components.
 *
 * @author Adam Gibson
 */
public abstract class BaseComplexNDArray extends BaseNDArray implements IComplexNDArray {

    public BaseComplexNDArray() {
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     */
    public BaseComplexNDArray(DataBuffer data, int[] shape, int[] stride) {
        this(data, shape, stride, 0, Nd4j.order());
    }

    /**
     *
     * @param data
     */
    public BaseComplexNDArray(float[] data) {
        super(data);
    }


    /**
     *
     * @param buffer
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     */
    public BaseComplexNDArray(DataBuffer buffer, int[] shape, int[] stride, int offset, char ordering) {
        super(buffer, shape, stride, offset, ordering);
    }


    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data     the data to use
     * @param shape    the shape of the ndarray
     * @param ordering
     */
    public BaseComplexNDArray(float[] data, int[] shape, char ordering) {
        this(data, shape, Nd4j.getComplexStrides(shape, ordering), 0, ordering);
    }

    /**
     *
     * @param shape
     * @param offset
     * @param ordering
     */
    public BaseComplexNDArray(int[] shape, int offset, char ordering) {
        this(Nd4j.createBuffer(ArrayUtil.prodLong(shape) * 2),
                shape, Nd4j.getComplexStrides(shape, ordering),
                offset, ordering);
    }

    /**
     *
     * @param shape
     */
    public BaseComplexNDArray(int[] shape) {
        this(Nd4j.createBuffer(ArrayUtil.prodLong(shape) * 2), shape, Nd4j.getComplexStrides(shape));
    }


    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param ordering
     */
    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        this(data, shape, stride, 0, ordering);
    }

    public BaseComplexNDArray(int[] shape, char ordering) {
        this(Nd4j.createBuffer(ArrayUtil.prodLong(shape) * 2), shape, Nd4j.getComplexStrides(shape, ordering), 0, ordering);
    }


    /**
     * Initialize the given ndarray as the real component
     *
     * @param m        the real component
     * @param stride   the stride of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public BaseComplexNDArray(INDArray m, int[] stride, char ordering) {
        this(m.shape(), stride, ordering);
        copyFromReal(m);

    }


    /**
     * Construct a complex matrix from a realComponent matrix.
     */
    public BaseComplexNDArray(INDArray m, char ordering) {
        this(m.shape(), ordering);
        copyFromReal(m);
    }


    /**
     * Construct a complex matrix from a realComponent matrix.
     */
    public BaseComplexNDArray(INDArray m) {
        this(m, Nd4j.order());
    }

    /**
     * Create with the specified ndarray as the real component
     * and the given stride
     *
     * @param m      the ndarray to use as the stride
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(INDArray m, int[] stride) {
        this(m, stride, Nd4j.order());
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices the slices of the ndarray
     * @param shape  the final shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices, int[] shape, int[] stride) {
        this(slices, shape, stride, Nd4j.order());
    }


    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices   the slices of the ndarray
     * @param shape    the final shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices, int[] shape, int[] stride, char ordering) {
        this(new float[ArrayUtil.prod(shape) * 2]);
        List<IComplexNumber> list = new ArrayList<>();
        for (int i = 0; i < slices.size(); i++) {
            IComplexNDArray flattened = slices.get(i).ravel();
            for (int j = 0; j < flattened.length(); j++)
                list.add(flattened.getComplex(j));
        }
        throw new UnsupportedOperationException();
/*

        this.ordering = ordering;
        this.data = Nd4j.createBuffer(ArrayUtil.prod(shape) * 2);
        this.stride = stride;
        init(shape);

        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            putScalar(count, list.get(i));
            count++;
        }*/
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices   the slices of the ndarray
     * @param shape    the final shape of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices, int[] shape, char ordering) {
        this(slices, shape, Nd4j.getComplexStrides(shape, ordering), ordering);
    }

    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, int offset, Character order) {
        this.data = Nd4j.createBuffer(data);
     /*   this.stride = ArrayUtil.copy(stride);
        this.offset = offset;
        this.ordering = order;
        init(shape);*/
        throw new UnsupportedOperationException();
    }

    /**
     *
     * @param data
     */
    public BaseComplexNDArray(DataBuffer data) {
        super(data);
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     */
    public BaseComplexNDArray(DataBuffer data, int[] shape, int[] stride, int offset) {
        this.data = data;
     /*   this.stride = ArrayUtil.copy(stride);
        this.offset = offset;
        this.ordering = Nd4j.order();
        init(shape);*/
        throw new UnsupportedOperationException();


    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     */
    public BaseComplexNDArray(IComplexNumber[] data, int[] shape, int[] stride, int offset, char ordering) {
        this(shape, stride, offset, ordering);
        assert data.length <= length;
        for (int i = 0; i < data.length; i++) {
            putScalar(i, data[i]);
        }
    }

    /**
     *
     * @param data
     * @param shape
     */
    public BaseComplexNDArray(DataBuffer data, int[] shape) {
        this(shape);
        this.data = data;
    }


    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     */
    public BaseComplexNDArray(IComplexNumber[] data, int[] shape, int[] stride, int offset) {
        this(data, shape, stride, offset, Nd4j.order());
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     */
    public BaseComplexNDArray(IComplexNumber[] data, int[] shape, int offset, char ordering) {
        this(data, shape, Nd4j.getComplexStrides(shape), offset, ordering);
    }

    /**
     *
     * @param buffer
     * @param shape
     * @param offset
     * @param ordering
     */
    public BaseComplexNDArray(DataBuffer buffer, int[] shape, int offset, char ordering) {
        this(buffer, shape, Nd4j.getComplexStrides(shape), offset, ordering);
    }

    /**
     *
     * @param buffer
     * @param shape
     * @param offset
     */
    public BaseComplexNDArray(DataBuffer buffer, int[] shape, int offset) {
        this(buffer, shape, Nd4j.getComplexStrides(shape), offset, Nd4j.order());
    }

    /**
     *
     * @param data
     * @param order
     */
    public BaseComplexNDArray(float[] data, Character order) {
        this(data, new int[]{1,data.length / 2}, order);
    }


    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices the slices of the ndarray
     * @param shape  the final shape of the ndarray
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices, int[] shape) {
        this(slices, shape, Nd4j.order());


    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float
     *
     * @param newData the new data for this array
     * @param shape   the shape of the ndarray
     */
    public BaseComplexNDArray(IComplexNumber[] newData, int[] shape) {
        super(new float[ArrayUtil.prod(shape) * 2]);
      /*  init(shape);
        for (int i = 0; i < length; i++)
            putScalar(i, newData[i].asDouble());
*/
        throw new UnsupportedOperationException();

    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float
     *
     * @param newData the new data for this array
     * @param shape   the shape of the ndarray
     */
    public BaseComplexNDArray(IComplexNumber[] newData, int[] shape, int[] stride) {
        super(new float[ArrayUtil.prod(shape) * 2]);
    /*    this.stride = stride;
        init(shape);
        for (int i = 0; i < length; i++)
            put(i, newData[i].asDouble());
*/
        throw new UnsupportedOperationException();

    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float
     *
     * @param newData  the new data for this array
     * @param shape    the shape of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public BaseComplexNDArray(IComplexNumber[] newData, int[] shape, char ordering) {
        super(new float[ArrayUtil.prod(shape) * 2]);
     /*   this.ordering = ordering;
        //init(shape);
        for (int i = 0; i < length; i++)
            put(i, newData[i]);*/
        throw new UnsupportedOperationException();

    }

    /**
     * Initialize with the given data,shape and stride
     *
     * @param data   the data to use
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(float[] data, int[] shape, int[] stride) {
        this(data, shape, stride, 0, Nd4j.order());
    }

    /**
     *
     * @param data
     * @param shape
     */
    public BaseComplexNDArray(float[] data, int[] shape) {
        this(data, shape, 0);
    }


    public BaseComplexNDArray(float[] data, int[] shape, int offset, char ordering) {
        this(data, shape, ordering == NDArrayFactory.C ? calcStrides(shape, 2) : calcStridesFortran(shape, 2), offset, ordering);
    }

    public BaseComplexNDArray(float[] data, int[] shape, int offset) {
        this(data, shape, offset, Nd4j.order());
    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public BaseComplexNDArray(int[] shape, int[] stride, int offset) {
        this(new float[ArrayUtil.prod(shape) * 2], shape, stride, offset);
    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering for the ndarray
     */
    public BaseComplexNDArray(int[] shape, int[] stride, int offset, char ordering) {
        this(new float[ArrayUtil.prod(shape) * 2], shape, stride, offset);
        this.shapeInformation = Shape.createShapeInformation(shape,stride,offset,stride[stride.length - 1],ordering);
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(int[] shape, int[] stride, char ordering) {
        this(shape, stride, 0, ordering);
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(int[] shape, int[] stride) {
        this(shape, stride, 0);
    }


    /**
     * @param shape
     * @param offset
     */
    public BaseComplexNDArray(int[] shape, int offset) {
        this(shape, offset, Nd4j.order());
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseComplexNDArray(int newRows, int newColumns) {
        this(new int[]{newRows, newColumns});
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering   the ordering of the ndarray
     */
    public BaseComplexNDArray(int newRows, int newColumns, char ordering) {
        this(new int[]{newRows, newColumns}, ordering);
    }

    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, int offset) {
        this(data, shape, stride, offset, Nd4j.order());
    }

    /**
     *
     * @param real
     */
    protected void copyFromReal(INDArray real) {
        if(!Shape.shapeEquals(shape(),real.shape()))
            throw new IllegalStateException("Unable to copy array. Not the same shape");
        INDArray linear = real.linearView();
        IComplexNDArray thisLinear = linearView();
        for (int i = 0; i < linear.length(); i++) {
            thisLinear.putScalar(i, Nd4j.createComplexNumber(linear.getDouble(i),0.0));
        }
    }


    @Override
    protected IComplexNDArray create(DataBuffer data, int[] shape, int[] strides) {
        return Nd4j.createComplex(data,shape,strides,offset(),ordering());
    }

    /**
     * Copy real numbers to arr
     * @param arr the arr to copy to
     */
    protected void copyRealTo(INDArray arr) {
        INDArray linear = arr.linearView();
        IComplexNDArray thisLinear = linearView();
        if(arr.isScalar())
            arr.putScalar(0,getReal(0));
        else
            for (int i = 0; i < linear.length(); i++) {
                arr.putScalar(i, thisLinear.getReal(i));
            }

    }

    /**
     * Copy imaginary numbers to the given
     * ndarray
     * @param arr the array to copy imaginary numbers to
     */
    protected void copyImagTo(INDArray arr) {
        INDArray linear = arr.linearView();
        IComplexNDArray thisLinear = linearView();
        if(arr.isScalar())
            arr.putScalar(0,getReal(0));
        else
            for (int i = 0; i < linear.length(); i++) {
                arr.putScalar(i, thisLinear.getImag(i));
            }

    }


    @Override
    public int blasOffset() {
        return  offset();
    }

    @Override
    public IComplexNDArray linearViewColumnOrder() {
        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Length can not be >= Integer.MAX_VALUE");
        return Nd4j.createComplex(data, new int[]{(int)length, 1}, offset());
    }

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    @Override
    public IComplexNDArray linearView() {
        if (isVector() || isScalar() || length() == 1 || length() == size(0))
            return this;
      return this;
    }

    @Override
    public void resetLinearView() {
    }

    @Override
    public IComplexNumber getComplex(int i, IComplexNumber result) {
        if(!isVector() || i >= length())
            throw new IllegalArgumentException("Given index >= length " + length());
        IComplexNumber d = getComplex(i);
        return result.set(d.realComponent(), d.imaginaryComponent());
    }

    @Override
    public IComplexNumber getComplex(int i, int j, IComplexNumber result) {
        IComplexNumber d = getComplex(i, j);
        return result.set(d.realComponent(), d.imaginaryComponent());
    }

    @Override
    public IComplexNDArray putScalar(int j, int i, IComplexNumber conji) {
        return putScalar(new int[]{j, i}, conji);
    }

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    public IComplexNDArray eps(Number other) {
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
    public IComplexNDArray eps(IComplexNumber other) {
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
    public IComplexNDArray epsi(IComplexNumber other) {
        IComplexNDArray linear = linearView();
        double otherVal = other.realComponent().doubleValue();
        for (int i = 0; i < linearView().length(); i++) {
            IComplexNumber n = linear.getComplex(i);
            double real = n.realComponent().doubleValue();
            double diff = Math.abs(real - otherVal);
            if (diff <= Nd4j.EPS_THRESHOLD)
                linear.putScalar(i, Nd4j.createDouble(1, 0));
            else
                linear.putScalar(i, Nd4j.createDouble(0, 0));
        }

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
    public IComplexNDArray epsi(Number other) {
        IComplexNDArray linear = linearView();
        double otherVal = other.doubleValue();
        for (int i = 0; i < linearView().length(); i++) {
            IComplexNumber n = linear.getComplex(i);
            double real = n.realComponent().doubleValue();
            double diff = Math.abs(real - otherVal);
            if (diff <= Nd4j.EPS_THRESHOLD)
                linear.putScalar(i, Nd4j.createDouble(1, 0));
            else
                linear.putScalar(i, Nd4j.createDouble(0, 0));
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
    public IComplexNDArray eps(INDArray other) {
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
    public IComplexNDArray epsi(INDArray other) {
        IComplexNDArray linear = linearView();

        if (other instanceof IComplexNDArray) {
            IComplexNDArray otherComplex = (IComplexNDArray) other;
            IComplexNDArray otherComplexLinear = otherComplex.linearView();

            for (int i = 0; i < linearView().length(); i++) {
                IComplexNumber n = linear.getComplex(i);
                IComplexNumber otherComplexNumber = otherComplexLinear.getComplex(i);
                double real = n.absoluteValue().doubleValue();
                double otherAbs = otherComplexNumber.absoluteValue().doubleValue();
                double diff = Math.abs(real - otherAbs);
                if (diff <= Nd4j.EPS_THRESHOLD)
                    linear.putScalar(i, Nd4j.createDouble(1, 0));
                else
                    linear.putScalar(i, Nd4j.createDouble(0, 0));
            }


        }

        return this;

    }

    @Override
    public IComplexNDArray lt(Number other) {
        return dup().lti(other);
    }

    @Override
    public IComplexNDArray lti(Number other) {
        IComplexNDArray linear = linearView();
        double val = other.doubleValue();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue() < val ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
        }
        return this;
    }

    @Override
    public IComplexNDArray eq(Number other) {
        return dup().eqi(other);
    }

    @Override
    public IComplexNDArray eqi(Number other) {
        IComplexNDArray linear = linearView();
        double val = other.doubleValue();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue() == val ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
        }
        return this;
    }

    @Override
    public IComplexNDArray gt(Number other) {
        return dup().gti(other);
    }

    @Override
    public IComplexNDArray gti(Number other) {
        IComplexNDArray linear = linearView();
        double val = other.doubleValue();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue() > val ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
        }
        return this;
    }

    @Override
    public IComplexNDArray lt(INDArray other) {
        return dup().lti(other);
    }

    @Override
    public IComplexNDArray lti(INDArray other) {
        if (other instanceof IComplexNDArray) {
            IComplexNDArray linear = linearView();
            IComplexNDArray otherLinear = (IComplexNDArray) other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        < otherLinear.getComplex(i).absoluteValue().doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        } else {
            IComplexNDArray linear = linearView();
            INDArray otherLinear = other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        < otherLinear.getDouble(i) ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        }

        return this;
    }

    @Override
    public IComplexNDArray eq(INDArray other) {
        return dup().eqi(other);
    }

    @Override
    public IComplexNDArray eqi(INDArray other) {
        if (other instanceof IComplexNDArray) {
            IComplexNDArray linear = linearView();
            IComplexNDArray otherLinear = (IComplexNDArray) other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        == otherLinear.getComplex(i).absoluteValue().doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        } else {
            IComplexNDArray linear = linearView();
            INDArray otherLinear = other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        == otherLinear.getDouble(i) ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        }

        return this;
    }

    @Override
    public IComplexNDArray neq(INDArray other) {
        return dup().neqi(other);
    }

    @Override
    public IComplexNDArray neq(Number other) {
        return dup().neqi(other);
    }

    @Override
    public IComplexNDArray neqi(Number other) {
        IComplexNDArray linear = linearView();
        double otherVal = other.doubleValue();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                    != otherVal ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
        }
        return this;
    }

    @Override
    public IComplexNDArray neqi(INDArray other) {
        if (other instanceof IComplexNDArray) {
            IComplexNDArray linear = linearView();
            IComplexNDArray otherLinear = (IComplexNDArray) other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        != otherLinear.getComplex(i).absoluteValue().doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        } else {
            IComplexNDArray linear = linearView();
            INDArray otherLinear = other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        != otherLinear.getDouble(i) ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        }

        return this;
    }

    @Override
    public IComplexNDArray gt(INDArray other) {
        return dup().gti(other);
    }

    @Override
    public IComplexNDArray gti(INDArray other) {
        if (other instanceof IComplexNDArray) {
            IComplexNDArray linear = linearView();
            IComplexNDArray otherLinear = (IComplexNDArray) other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        > otherLinear.getComplex(i).absoluteValue().doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        } else {
            IComplexNDArray linear = linearView();
            INDArray otherLinear = other.linearView();
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, linear.getComplex(i).absoluteValue().doubleValue()
                        > otherLinear.getDouble(i) ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0));
            }
        }
        return this;
    }


    @Override
    public IComplexNDArray rdiv(Number n, INDArray result) {
        return dup().rdivi(n, result);
    }

    @Override
    public IComplexNDArray rdivi(Number n, INDArray result) {
        return rdivi(Nd4j.createDouble(n.doubleValue(), 0), result);

    }

    @Override
    public IComplexNDArray rsub(Number n, INDArray result) {
        return dup().rsubi(n, result);
    }

    @Override
    public IComplexNDArray rsubi(Number n, INDArray result) {
        return rsubi(Nd4j.createDouble(n.doubleValue(), 0), result);

    }

    @Override
    public IComplexNDArray div(Number n, INDArray result) {
        return dup().divi(n, result);
    }

    @Override
    public IComplexNDArray divi(Number n, INDArray result) {
        return divi(Nd4j.createDouble(n.doubleValue(), 0), result);

    }

    @Override
    public IComplexNDArray mul(Number n, INDArray result) {
        return dup().muli(n, result);
    }

    @Override
    public IComplexNDArray muli(Number n, INDArray result) {
        return muli(Nd4j.createDouble(n.doubleValue(), 0), result);

    }

    @Override
    public IComplexNDArray sub(Number n, INDArray result) {
        return dup().subi(n, result);
    }

    @Override
    public IComplexNDArray subi(Number n, INDArray result) {
        return subi(Nd4j.createDouble(n.doubleValue(), 0), result);
    }

    @Override
    public IComplexNDArray add(Number n, INDArray result) {
        return dup().addi(n, result);
    }

    @Override
    public IComplexNDArray addi(Number n, INDArray result) {
        return addi(Nd4j.createDouble(n.doubleValue(), 0), result);
    }

    @Override
    public IComplexNDArray dup() {
        return (IComplexNDArray) Shape.toOffsetZeroCopy(this);
    }

    @Override
    public IComplexNDArray rsubRowVector(INDArray rowVector) {
        return dup().rsubiRowVector(rowVector);
    }

    @Override
    public IComplexNDArray rsubiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 't');
    }

    @Override
    public IComplexNDArray rsubColumnVector(INDArray columnVector) {
        return dup().rsubiColumnVector(columnVector);
    }

    @Override
    public IComplexNDArray rsubiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'h');
    }

    @Override
    public IComplexNDArray rdivRowVector(INDArray rowVector) {
        return dup().rdiviRowVector(rowVector);
    }

    @Override
    public IComplexNDArray rdiviRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 't');
    }

    @Override
    public IComplexNDArray rdivColumnVector(INDArray columnVector) {
        return dup().rdiviColumnVector(columnVector);
    }

    @Override
    public IComplexNDArray rdiviColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 't');
    }

    @Override
    protected IComplexNDArray doRowWise(INDArray rowVector, char operation) {
        return (IComplexNDArray) super.doRowWise(rowVector,operation);
    }

    @Override
    protected IComplexNDArray doColumnWise(INDArray columnVector, char operation) {
        return (IComplexNDArray) super.doColumnWise(columnVector,operation);
    }

    /**
     * Returns the squared (Euclidean) distance.
     */
    @Override
    public double squaredDistance(INDArray other) {
        double sd = 0.0;
        if (other instanceof IComplexNDArray) {
            IComplexNDArray n = (IComplexNDArray) other;
            IComplexNDArray nLinear = n.linearView();

            for (int i = 0; i < length; i++) {
                IComplexNumber diff = linearView().getComplex(i).sub(nLinear.getComplex(i));
                double d = diff.absoluteValue().doubleValue();
                sd += d * d;
            }
            return sd;
        }
        for (int i = 0; i < length; i++) {
            INDArray linear = other.linearView();
            IComplexNumber diff = linearView().getComplex(i).sub(linear.getDouble(i));
            double d = diff.absoluteValue().doubleValue();
            sd += d * d;
        }

        return sd;
    }

    /**
     * Returns the (euclidean) distance.
     */
    @Override
    public double distance2(INDArray other) {
        return Math.sqrt(squaredDistance(other));
    }

    /**
     * Returns the (1-norm) distance.
     */
    @Override
    public double distance1(INDArray other) {
        float d = 0.0f;
        if (other instanceof IComplexNDArray) {
            IComplexNDArray n2 = (IComplexNDArray) other;
            IComplexNDArray n2Linear = n2.linearView();
            for (int i = 0; i < length; i++) {
                IComplexNumber n = getComplex(i).sub(n2Linear.getComplex(i));
                d += n.absoluteValue().doubleValue();
            }
            return d;
        }

        INDArray linear = other.linearView();

        for (int i = 0; i < length; i++) {
            IComplexNumber n = linearView().getComplex(i).sub(linear.getDouble(i));
            d += n.absoluteValue().doubleValue();
        }
        return d;
    }

    @Override
    public IComplexNDArray put(INDArrayIndex[] indices, INDArray element) {
        super.put(indices,element);
        return this;
    }

    @Override
    public IComplexNDArray normmax(int...dimension) {
        return Nd4j.createComplex(super.normmax(dimension));
    }

    @Override
    public Number normmaxNumber() {
        return normmaxComplex().absoluteValue();
    }

    @Override
    public IComplexNumber normmaxComplex() {
        return normmax(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray prod(int...dimension) {
        return Nd4j.createComplex(super.prod(dimension));
    }

    @Override
    public Number prodNumber() {
        return prodComplex().absoluteValue();
    }

    @Override
    public IComplexNumber prodComplex() {
        return prod(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray mean(int...dimension) {
        return Nd4j.createComplex(super.mean(dimension));
    }

    @Override
    public Number meanNumber() {
        return meanComplex().absoluteValue();
    }

    @Override
    public IComplexNumber meanComplex() {
        return mean(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray var(int...dimension) {
        return Nd4j.createComplex(super.var(dimension));
    }

    @Override
    public Number varNumber() {
        return varComplex().absoluteValue();
    }

    @Override
    public IComplexNumber varComplex() {
        return var(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray max(int...dimension) {
        return Nd4j.createComplex(super.max(dimension));
    }

    @Override
    public Number maxNumber() {
        return maxComplex().absoluteValue();
    }

    @Override
    public IComplexNumber maxComplex() {
        return max(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray sum(int...dimension) {
        return Nd4j.createComplex(super.sum(dimension));
    }

    @Override
    public Number sumNumber() {
        return sumComplex().absoluteValue();
    }

    @Override
    public IComplexNumber sumComplex() {
        return sum(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray min(int...dimension) {
        return Nd4j.createComplex(super.min(dimension));
    }

    @Override
    public Number minNumber() {
        return minComplex().absoluteValue();
    }

    @Override
    public IComplexNumber minComplex() {
        return min(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray norm1(int...dimension) {
        return Nd4j.createComplex(super.norm1(dimension));
    }

    @Override
    public Number norm1Number() {
        return norm1Complex().absoluteValue();
    }

    @Override
    public IComplexNumber norm1Complex() {
        return norm1(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray std(int...dimension) {
        return Nd4j.createComplex(super.std(dimension));
    }

    @Override
    public Number stdNumber() {
        return stdComplex().absoluteValue();
    }

    @Override
    public IComplexNumber stdComplex() {
        return std(Integer.MAX_VALUE).getComplex(0);
    }

    @Override
    public IComplexNDArray norm2(int...dimension) {
        return Nd4j.createComplex(super.norm2(dimension));
    }

    @Override
    public Number norm2Number() {
        return norm2Complex().absoluteValue();
    }

    @Override
    public IComplexNumber norm2Complex() {
        return norm2(Integer.MAX_VALUE).getComplex(0);
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
    public IComplexNDArray put(int i, int j, Number element) {
        return (IComplexNDArray) super.put(i, j, Nd4j.scalar(element));
    }


    /**
     * @param indexes
     * @param value
     * @return
     */
    @Override
    public IComplexNDArray put(int[] indexes, double value) {
        //int ix = offset;
       /* if (indexes.length != shape.length)
            throw new IllegalArgumentException("Unable to applyTransformToDestination values: number of indices must be equal to the shape");

        for (int i = 0; i < shape.length; i++)
            ix += indexes[i] * stride[i];


        data.put(ix, value);
        return this;*/
        throw new UnsupportedOperationException();

    }

    @Override
    public IComplexNDArray put(int i, int j, IComplexNumber complex) {
        return putScalar(new int[]{i, j}, complex);
    }




    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to transform
     * @return this for chainability
     */
    @Override
    public IComplexNDArray putSlice(int slice, IComplexNDArray put) {
        if (isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            put(0, put.getScalar(0));
            return this;
        } else if (isVector()) {
            assert put.isScalar() || put.isVector() &&
                    put.length() == length() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            if (put.isScalar())
                putScalar(slice, put.getComplex(0));
            else
                for (int i = 0; i < length(); i++)
                    putScalar(i, put.getComplex(i));

            return this;
        }

        assertSlice(put, slice);

        IComplexNDArray view = slice(slice);

        if (put.length() == 1)
            putScalar(slice, put.getComplex(0));
        else if (put.isVector())
            for (int i = 0; i < put.length(); i++)
                view.putScalar(i, put.getComplex(i));
        else {

            assert Shape.shapeEquals(view.shape(),put.shape());
            IComplexNDArray linear = (IComplexNDArray)view.linearView();
            IComplexNDArray putLinearView = put.linearView();
            for(int i = 0; i < linear.length(); i++) {
                linear.putScalar(i,putLinearView.getComplex(i));
            }
        }

        return this;
    }


    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with      the one to swap it with
     * @return the swapped axes view
     */
    public IComplexNDArray swapAxes(int dimension, int with) {
        return (IComplexNDArray) super.swapAxes(dimension, with);
    }

    /**
     * Compute complex conj (in-place).
     */
    @Override
    public IComplexNDArray conji() {
        IComplexNDArray reshaped = linearView();
        IComplexDouble c = Nd4j.createDouble(0.0, 0);
        for (int i = 0; i < length; i++) {
            IComplexNumber conj = reshaped.getComplex(i, c).conj();
            reshaped.putScalar(i, conj);

        }
        return this;
    }

    @Override
    public IComplexNDArray hermitian() {
        IComplexNDArray result = Nd4j.createComplex(shape());

        IComplexDouble c = Nd4j.createDouble(0, 0);

        for (int i = 0; i < slices(); i++)
            for (int j = 0; j < columns; j++)
                result.putScalar(j, i, getComplex(i, j, c).conji());
        return result;
    }

    /**
     * Compute complex conj.
     */
    @Override
    public IComplexNDArray conj() {
        return dup().conji();
    }

    @Override
    public INDArray getReal() {
        INDArray result = Nd4j.create(shape());
        IComplexNDArray linearView = linearView();
        INDArray linearRet = result.linearView();
        for (int i = 0; i < linearView.length(); i++) {
            linearRet.putScalar(i, linearView.getReal(i));
        }
        return result;
    }

    @Override
    public double getImag(int i) {
        return getComplex(i).imaginaryComponent().doubleValue();
    }

    @Override
    public double getReal(int i) {
        return getComplex(i).realComponent().doubleValue();

    }

    @Override
    public IComplexNDArray putReal(int rowIndex, int columnIndex, double value) {
        data.put(2 * index(rowIndex, columnIndex) + offset(), value);
        return this;
    }





    @Override
    public IComplexNDArray putImag(int rowIndex, int columnIndex, double value) {
        data.put(index(rowIndex, columnIndex) + 1 + offset(), value);
        return this;
    }

    @Override
    public IComplexNDArray putReal(int i, float v) {
        super.putScalar(i,v);
        return this;
    }

    @Override
    public IComplexNDArray putImag(int i, float v) {
        /*int offset  = this.offset + Shape.offsetFor(this, i) + 1;
        data.put(offset,v);*/
        return this;
    }


    @Override
    public IComplexNumber getComplex(int i) {
        if(i >= length())
            throw new IllegalArgumentException("Index " + i + " >= " + length());
        int[] dimensions = Shape.ind2sub(this,i);
        return getComplex(dimensions);
    }

    @Override
    public IComplexNumber getComplex(int i, int j) {
        return getComplex(new int[]{i, j});
    }

    @Override
    public IComplexNumber getComplex(int... indices) {
//
//        int ix = offset;
//        for (int i = 0; i < indices.length; i++)
//            ix += indices[i] * stride[i];
//
//        return data.getComplex(ix);
        throw new UnsupportedOperationException();

    }

    /**
     * Get realComponent part of the matrix.
     */
    @Override
    public INDArray real() {
       /* INDArray ret = Nd4j.create(shape);
        copyRealTo(ret);
        return ret;*/
        throw new UnsupportedOperationException();

    }

    /**
     * Get imaginary part of the matrix.
     */
    @Override
    public INDArray imag() {
      /*  INDArray ret = Nd4j.create(shape);
        copyImagTo(ret);
        return ret;*/
        throw new UnsupportedOperationException();

    }


    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public IComplexNDArray put(int i, IComplexNDArray element) {
        if (element == null)
            throw new IllegalArgumentException("Unable to insert null element");

        assert element.isScalar() : "Unable to insert non scalar element";
        int idx = linearIndex(i);
        IComplexNumber n = element.getComplex(0);
        data.put(idx, n.realComponent().doubleValue());
        data.put(idx + 1, n.imaginaryComponent().doubleValue());
        return this;
    }


    /**
     * Fetch a particular number on a multi dimensional scale.
     *
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    @Override
    public IComplexNDArray getScalar(int... indexes) {
     /*   int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return Nd4j.scalar(Nd4j.createDouble(data.getDouble(ix), data.getDouble(ix + 1)));*/
        throw new UnsupportedOperationException();

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
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to put
     * @return this for chainability
     */
    @Override
    public IComplexNDArray putSlice(int slice, INDArray put) {
        return putSlice(slice,Nd4j.createComplex(put));
    }



    @Override
    public IComplexNDArray subArray(int[] offsets, int[] shape, int[] stride) {
        return (IComplexNDArray) super.subArray(offsets, shape, stride);
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
      /*  if (!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");

        if(isRowVector() && indices.length == 2 && indices[0] == 0) {
            int idx = linearIndex(indices[1]);
            if(element instanceof IComplexNDArray) {
                IComplexNDArray arr2 = (IComplexNDArray) element;
                data.put(idx,arr2.getComplex(0));

            }
            else
                data.put(idx,element.getDouble(0));
        }
        else {
            int ix = offset;
            if (indices.length != shape.length)
                throw new IllegalArgumentException("Unable to op values: number of indices must be equal to the shape");


            for (int i = 0; i < shape.length; i++)
                ix += indices[i] * stride[i];

            if (element instanceof IComplexNDArray) {
                IComplexNumber element2 = ((IComplexNDArray) element).getComplex(0);
                data.put(ix, element2.realComponent().doubleValue());
                data.put(ix + 1, element2.imaginaryComponent().doubleValue());
            } else {
                double element2 = element.getDouble(0);
                data.put(ix, element2);
                data.put(ix + 1, 0);
            }

        }


        return this;
*/
        throw new UnsupportedOperationException();

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
        return put(new int[]{i, j}, element);
    }

    @Override
    protected IComplexNDArray create(BaseNDArray baseNDArray) {
        return Nd4j.createComplex(baseNDArray);
    }


    protected IComplexNDArray createScalar(double d) {
        return Nd4j.createComplex(Nd4j.scalar(d));
    }

    protected INDArray createScalarForIndex(int i,boolean applyOffset) {
        return Nd4j.createComplex(data(), new int[]{1, 1}, new int[]{1, 1}, applyOffset ? offset() + i : i);
    }

    /**
     * Returns the specified slice of this matrix.
     * In matlab, this would be equivalent to (given a 2 x 2 x 2):
     * A(:,:,x) where x is the slice you want to return.
     * <p/>
     * The slice is always relative to the final dimension of the matrix.
     *
     * @param slice the slice to return
     * @return the specified slice of this matrix
     */
    @Override
    public IComplexNDArray slice(int slice) {
        return (IComplexNDArray) super.slice(slice);
    }
    @Override
    public int elementStride() {
        return 2;
    }

    @Override
    protected IComplexNDArray create(int[] shape) {
        return Nd4j.createComplex(shape, Nd4j.getComplexStrides(shape, ordering()), 0);
    }

    @Override
    protected IComplexNDArray create(int[] shape,int[] strides,int offset) {
        return Nd4j.createComplex(shape, strides, offset);
    }

    @Override
    protected int[] getStrides(int[] shape,char ordering) {
        return Nd4j.getComplexStrides(shape, ordering);
    }

    @Override
    protected IComplexNDArray create(DataBuffer data, int[] newShape, int[] newStrides, int offset, char ordering) {
        return Nd4j.createComplex(data, newShape, newStrides, offset, ordering);
    }

    /**
     * Returns the slice of this from the specified dimension
     *
     * @param slice     the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    @Override
    public IComplexNDArray slice(int slice, int dimension) {
        return (IComplexNDArray) super.slice(slice,dimension);
    }



    @Override
    protected IComplexNDArray newShape(int[] newShape, char ordering) {
        return Nd4j.createComplex(super.newShape(newShape,ordering));
    }



    @Override
    protected IComplexNDArray create(DataBuffer data, int[] newShape, int[] newStrides, int offset) {
        return Nd4j.createComplex(data, newShape, newStrides, offset);
    }



    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    public IComplexNDArray repmat(int[] shape) {
        return (IComplexNDArray) super.repmat(shape);
    }


    /**
     * Assign all of the elements in the given
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    @Override
    public IComplexNDArray assign(IComplexNDArray arr) {
        if (!arr.isScalar())
            LinAlgExceptions.assertSameShape(this, arr);
        IComplexNDArray linear = linearView();
        IComplexNDArray otherLinear = arr.linearView();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, otherLinear.getComplex(i));
        }


        return this;
    }

    @Override
    public void assign(IComplexNumber aDouble) {
        IComplexNDArray linear = linearView();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, aDouble);
        }

    }

    /**
     * Get whole rows from the passed indices.
     *
     * @param rindices
     */
    @Override
    public IComplexNDArray getRows(int[] rindices) {
        INDArray rows = Nd4j.create(rindices.length, columns());
        for (int i = 0; i < rindices.length; i++) {
            rows.putRow(i, getRow(rindices[i]));
        }
        return (IComplexNDArray) rows;
    }


    @Override
    public IComplexNDArray put(INDArrayIndex[] indices, IComplexNumber element) {
        return put(indices, Nd4j.scalar(element));
    }

    @Override
    public IComplexNDArray put(INDArrayIndex[] indices, IComplexNDArray element) {
        super.put(indices,element);
        return this;
    }


    @Override
    protected INDArray create(int[] shape, char ordering) {
        return Nd4j.createComplex(shape, ordering);
    }

    @Override
    public IComplexNDArray put(INDArrayIndex[] indices, Number element) {
        return put(indices, Nd4j.scalar(element));

    }

    @Override
    public IComplexNDArray putScalar(int i, IComplexNumber value) {
        int[] dimensions = Shape.ind2sub(this, i);
        return putScalar(dimensions,value);
    }


    @Override
    protected IComplexNDArray create(DataBuffer buffer) {
        return Nd4j.createComplex(buffer, new int[]{1, (int) buffer.length()});
    }


    @Override
    protected IComplexNDArray create(int rows, int length) {
        return create(new int[]{rows,length});
    }


    @Override
    protected IComplexNDArray create(DataBuffer data,int[] shape,int offset) {
        return Nd4j.createComplex(data, shape, offset);
    }

    protected IComplexNDArray create(INDArray baseNDArray) {
        return Nd4j.createComplex(baseNDArray);
    }


    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to get
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    @Override
    public IComplexNDArray vectorAlongDimension(int index, int dimension) {
        return (IComplexNDArray) super.vectorAlongDimension(index,dimension);
    }

    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public IComplexNDArray cumsumi(int dimension) {
        /*if (isVector()) {
            IComplexNumber s = Nd4j.createDouble(0, 0);
            for (int i = 0; i < length; i++) {
                s.addi(getComplex(i));
                putScalar(i, s);
            }
        } else if (dimension == Integer.MAX_VALUE || dimension == shape.length - 1) {
            IComplexNDArray flattened = ravel().dup();
            IComplexNumber prevVal = flattened.getComplex(0);
            for (int i = 1; i < flattened.length(); i++) {
                IComplexNumber d = prevVal.add((flattened.getComplex(i)));
                flattened.putScalar(i, d);
                prevVal = d;
            }

            return flattened;
        } else {
            for (int i = 0; i < vectorsAlongDimension(dimension); i++) {
                IComplexNDArray vec = vectorAlongDimension(i, dimension);
                vec.cumsumi(0);

            }
        }


        return this;*/
        throw new UnsupportedOperationException();

    }


    /**
     * Dimshuffle: an extension of permute that adds the ability
     * to broadcast various dimensions.
     * <p/>
     * See theano for more examples.
     * This will only accept integers and xs.
     * <p/>
     * An x indicates a dimension should be broadcasted rather than permuted.
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public IComplexNDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) {
        return (IComplexNDArray) super.dimShuffle(rearrange,newOrder,broadCastable);

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
        return assign((IComplexNDArray) arr);
    }

    @Override
    public IComplexNDArray putScalar(int i, double value) {
        return put(i, Nd4j.scalar(value));
    }

    @Override
    public INDArray putScalar(int[] i, double value) {
        super.putScalar(i, value);
        return putScalar(i, Nd4j.createComplexNumber(value, 0));
    }

    @Override
    public IComplexNDArray putScalar(int[] indexes, IComplexNumber complexNumber) {
       /* int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            if(indexes[i] >= size(i))
                throw new IllegalArgumentException("Illegal index " + i + " size at this index is " + size(i));
            ix += indexes[i] * stride[i];
        }

        data.put(ix, complexNumber.asFloat().realComponent().doubleValue());
        data.put(ix + 1, complexNumber.asFloat().imaginaryComponent().doubleValue());

        return this;*/
        throw new UnsupportedOperationException();

    }

    /**
     * Negate each element.
     */
    @Override
    public IComplexNDArray neg() {
        return dup().negi();
    }

    /**
     * Negate each element (in-place).
     */
    @Override
    public IComplexNDArray negi() {
        return (IComplexNDArray) Transforms.neg(this);
    }

    @Override
    public IComplexNDArray rdiv(Number n) {
        return rdiv(n, this);
    }

    @Override
    public IComplexNDArray rdivi(Number n) {
        return rdivi(n, this);
    }

    @Override
    public IComplexNDArray rsub(Number n) {
        return rsub(n, this);
    }

    @Override
    public IComplexNDArray rsubi(Number n) {
        return rsubi(n, this);
    }


    @Override
    public IComplexNDArray div(Number n) {
        return dup().divi(n);
    }

    @Override
    public IComplexNDArray divi(Number n) {
        return divi(Nd4j.complexScalar(n));
    }

    @Override
    public IComplexNDArray mul(Number n) {
        return dup().muli(n);
    }

    @Override
    public IComplexNDArray muli(Number n) {
        return muli(Nd4j.complexScalar(n));
    }

    @Override
    public IComplexNDArray sub(Number n) {
        return dup().subi(n);
    }

    @Override
    public IComplexNDArray subi(Number n) {
        return subi(Nd4j.complexScalar(n));
    }

    @Override
    public IComplexNDArray add(Number n) {
        return dup().addi(n);
    }

    @Override
    public IComplexNDArray addi(Number n) {
        return addi(Nd4j.complexScalar(n));
    }

    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    public IComplexNDArray get(INDArrayIndex... indexes) {
        return (IComplexNDArray) super.get(indexes);
    }


    @Override
    public IComplexNDArray cond(Condition condition) {
        return dup().condi(condition);
    }

    @Override
    public IComplexNDArray condi(Condition condition) {
        IComplexNDArray linear = linearView();
        for (int i = 0; i < length(); i++) {
            boolean met = condition.apply(linear.getComplex(i));
            IComplexNumber put = Nd4j.createComplexNumber(met ? 1 : 0, 0);
            linear.putScalar(i, put);
        }
        return this;
    }

    /**
     * Get whole columns from the passed indices.
     *
     * @param cindices
     */
    @Override
    public IComplexNDArray getColumns(int[] cindices) {
        return (IComplexNDArray) super.getColumns(cindices);
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
        return (IComplexNDArray) super.putRow(row,toPut);
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
        assert toPut.isVector() && toPut.length() == rows : "Illegal length for row " + toPut.length() + " should have been " + columns;
        IComplexNDArray r = getColumn(column);
        if (toPut instanceof IComplexNDArray) {
            IComplexNDArray putComplex = (IComplexNDArray) toPut;
            for (int i = 0; i < r.length(); i++) {
                IComplexNumber n = putComplex.getComplex(i);
                r.putScalar(i, n);
            }
        } else {
            for (int i = 0; i < r.length(); i++)
                r.putScalar(i, Nd4j.createDouble(toPut.getDouble(i), 0));

        }

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
        return getScalar(new int[]{row, column});
    }


    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public IComplexNDArray getScalar(int i) {
        return Nd4j.scalar(getComplex(i));
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
        if (element == null)
            throw new IllegalArgumentException("Unable to insert null element");
        assert element.isScalar() : "Unable to insert non scalar element";
        if (element instanceof IComplexNDArray) {
            IComplexNDArray n1 = (IComplexNDArray) element;
            IComplexNumber n = n1.getComplex(0);
            put(i, n);
        } else
            putScalar(i, Nd4j.createDouble(element.getDouble(0), 0.0));
        return this;
    }

    public void put(int i, IComplexNumber element) {
        int idx = linearIndex(i);
        data.put(idx, element.realComponent().doubleValue());
        data.put(idx + 1, element.imaginaryComponent().doubleValue());
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray diviColumnVector(INDArray columnVector) {
        for (int i = 0; i < columns(); i++) {
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
        for (int i = 0; i < rows(); i++) {
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
        for (int i = 0; i < columns(); i++) {
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
        for (int i = 0; i < rows(); i++) {
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
        for (int i = 0; i < columns(); i++) {
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
        for (int i = 0; i < rows(); i++) {
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
        for (int i = 0; i < columns(); i++) {
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
        for (int i = 0; i < rows(); i++) {
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
        return (IComplexNDArray) super.mmul(other);
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
        return dup().mmuli(other, result);
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
        return dup().divi(other, result);
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
        return dup().muli(other, result);
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
        return dup().subi(other, result);
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
        return dup().addi(other, result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public IComplexNDArray mmuli(INDArray other) {
        return mmuli(other, this);
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


        IComplexNDArray otherArray = (IComplexNDArray) other;
        IComplexNDArray resultArray = (IComplexNDArray) result;

        if (other.shape().length > 2) {
            for (int i = 0; i < other.slices(); i++) {
                resultArray.putSlice(i, slice(i).mmul(otherArray.slice(i)));
            }

            return resultArray;

        }


        LinAlgExceptions.assertMultiplies(this, other);

        if (other.isScalar()) {
            return muli(otherArray.getComplex(0), resultArray);
        }
        if (isScalar()) {
            return otherArray.muli(getComplex(0), resultArray);
        }

        /* check sizes and resize if necessary */
        //assertMultipliesWith(other);


        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            IComplexNDArray temp = Nd4j.createComplex(resultArray.shape());

            if (otherArray.columns() == 1) {
                Nd4j.getBlasWrapper().level2().gemv(BlasBufferUtil.getCharForTranspose(temp),BlasBufferUtil.getCharForTranspose(this),Nd4j.UNIT,this,otherArray,Nd4j.ZERO,temp);
            } else {
                Nd4j.getBlasWrapper().level3().gemm(BlasBufferUtil.getCharForTranspose(temp),BlasBufferUtil.getCharForTranspose(this),BlasBufferUtil.getCharForTranspose(other),Nd4j.UNIT,this,otherArray,Nd4j.ZERO,temp);

            }

            Nd4j.getBlasWrapper().copy(temp, resultArray);


        } else {
            if (otherArray.columns() == 1) {
                Nd4j.getBlasWrapper().level2().gemv(
                        BlasBufferUtil.getCharForTranspose(resultArray)
                        ,BlasBufferUtil.getCharForTranspose(this)
                        ,Nd4j.UNIT
                        ,this
                        ,otherArray
                        ,Nd4j.ZERO
                        ,resultArray);
            }


            else {
                Nd4j.getBlasWrapper().level3().gemm(
                        BlasBufferUtil.getCharForTranspose(resultArray)
                        ,BlasBufferUtil.getCharForTranspose(this)
                        ,BlasBufferUtil.getCharForTranspose(other)
                        ,Nd4j.UNIT
                        ,this
                        ,otherArray
                        ,Nd4j.ZERO
                        ,resultArray);
            }
        }
        return resultArray;


    }

    @Override
    public int secondaryStride() {
        return super.secondaryStride() / 2;
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public IComplexNDArray divi(INDArray other) {
        return divi(other, this);
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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        IComplexNDArray linear = linearView();
        IComplexNDArray cOtherLinear = cOther.linearView();
        IComplexNDArray cResultLinear = cResult.linearView();

        if (other.isScalar())
            return divi(cOther.getComplex(0), result);


        IComplexNumber c = Nd4j.createComplexNumber(0, 0);
        IComplexNumber d = Nd4j.createComplexNumber(0, 0);

        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, linear.getComplex(i, c).divi(cOtherLinear.getComplex(i, d)));
        return cResult;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray muli(INDArray other) {
        return muli(other, this);
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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        IComplexNDArray linear = linearView();
        IComplexNDArray cOtherLinear = cOther.linearView();
        IComplexNDArray cResultLinear = cResult.linearView();

        if (other.isScalar())
            return muli(cOther.getComplex(0), result);


        IComplexNumber c = Nd4j.createComplexNumber(0, 0);
        IComplexNumber d = Nd4j.createComplexNumber(0, 0);

        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, linear.getComplex(i, c).muli(cOtherLinear.getComplex(i, d)));
        return cResult;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray subi(INDArray other) {
        return subi(other, this);
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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        if (other.isScalar())
            return subi(cOther.getComplex(0), result);


        if (result == this)
            Nd4j.getBlasWrapper().axpy(Nd4j.NEG_UNIT, cOther, cResult);
        else if (result == other) {
            if (data.dataType() == (DataBuffer.Type.DOUBLE)) {
                Nd4j.getBlasWrapper().scal(Nd4j.NEG_UNIT.asDouble(), cResult);
                Nd4j.getBlasWrapper().axpy(Nd4j.UNIT, this, cResult);
            } else {
                Nd4j.getBlasWrapper().scal(Nd4j.NEG_UNIT.asFloat(), cResult);
                Nd4j.getBlasWrapper().axpy(Nd4j.UNIT, this, cResult);
            }

        } else {
            Nd4j.getBlasWrapper().copy(this, result);
            Nd4j.getBlasWrapper().axpy(Nd4j.NEG_UNIT, cOther, cResult);
        }
        return cResult;
    }

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public IComplexNDArray addi(INDArray other) {
        return addi(other, this);
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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        if (cOther.isScalar()) {
            return cResult.addi(cOther.getComplex(0), result);
        }
        if (isScalar()) {
            return cOther.addi(getComplex(0), result);
        }


        if (result == this) {

            Nd4j.getBlasWrapper().axpy(Nd4j.UNIT, cOther, cResult);
        } else if (result == other) {
            Nd4j.getBlasWrapper().axpy(Nd4j.UNIT, this, cResult);
        } else {
            INDArray resultLinear = result.linearView();
            INDArray otherLinear = other.linearView();
            INDArray linear = linearView();
            for (int i = 0; i < resultLinear.length(); i++) {
                resultLinear.putScalar(i, otherLinear.getDouble(i) + linear.getDouble(i));
            }

        }

        return (IComplexNDArray) result;
    }


    @Override
    public IComplexNDArray rdiv(IComplexNumber n, INDArray result) {
        return dup().rdivi(n, result);
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n, INDArray result) {
        IComplexNDArray cResult = (IComplexNDArray) result;
        IComplexNDArray cResultLinear = cResult.linearView();
        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, n.div(getComplex(i)));
        return cResult;
    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n, INDArray result) {
        return dup().rsubi(n, result);
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n, INDArray result) {
        IComplexNDArray cResult = (IComplexNDArray) result;
        IComplexNDArray cResultLinear = cResult.linearView();
        IComplexNDArray thiLinear = linearView();

        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, n.sub(thiLinear.getComplex(i)));
        return cResult;
    }

    @Override
    public IComplexNDArray div(IComplexNumber n, INDArray result) {
        return dup().divi(n, result);
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n, INDArray result) {
        IComplexNDArray cResult = (IComplexNDArray) result;
        IComplexNDArray cResultLinear = cResult.linearView();
        IComplexNDArray thisLinear = linearView();
        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, thisLinear.getComplex(i).div(n));
        return cResult;
    }

    @Override
    public IComplexNDArray mul(IComplexNumber n, INDArray result) {
        return dup().muli(n, result);
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n, INDArray result) {
        IComplexNDArray cResult = (IComplexNDArray) result;
        IComplexNDArray cResultLinear = cResult.linearView();
        IComplexNDArray thiLinear = linearView();
        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, thiLinear.getComplex(i).mul(n));
        return cResult;
    }

    @Override
    public IComplexNDArray sub(IComplexNumber n, INDArray result) {
        return dup().subi(n, result);
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n, INDArray result) {
        IComplexNDArray cResult = (IComplexNDArray) result;
        IComplexNDArray cResultLinear = cResult.linearView();

        IComplexNDArray linear = linearView();
        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, linear.getComplex(i).sub(n));
        return cResult;
    }

    @Override
    public IComplexNDArray add(IComplexNumber n, INDArray result) {
        return dup().addi(n, result);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n, INDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = (IComplexNDArray) result.linearView();

        for (int i = 0; i < length(); i++) {
            cResult.putScalar(i, linear.getComplex(i).add(n));
        }

        return (IComplexNDArray) result;
    }

    @Override
    public IComplexNDArray rdiv(IComplexNumber n) {
        return dup().rdivi(n, this);
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n) {
        return rdivi(n, this);
    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n) {
        return rsub(n, this);
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n) {
        return rsubi(n, this);
    }

    @Override
    public IComplexNDArray div(IComplexNumber n) {
        return div(n, this);
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n) {
        return divi(n, this);
    }

    @Override
    public IComplexNDArray mul(IComplexNumber n) {
        return dup().muli(n);
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n) {
        return muli(n, this);
    }

    @Override
    public IComplexNDArray sub(IComplexNumber n) {
        return dup().subi(n);
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n) {
        return subi(n, this);
    }

    @Override
    public IComplexNDArray add(IComplexNumber n) {
        return dup().addi(n);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n) {
        return addi(n, this);
    }


    @Override
    public IComplexNDArray putReal(int rowIndex, int columnIndex, float value) {
        return putReal(new int[]{rowIndex, columnIndex}, value);
    }

    @Override
    public IComplexNDArray putReal(int[] indices, float value) {
        return putReal(indices, (double) value);
    }

    @Override
    public IComplexNDArray putImag(int[] indices, float value) {
        return putImag(indices, (double) value);
    }

    @Override
    public IComplexNDArray putReal(int[] indices, double value) {
       /* int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];

        data.put(ix, value);
        return this;*/
        throw new UnsupportedOperationException();

    }

    @Override
    public IComplexNDArray putImag(int[] indices, double value) {
        /*int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];

        data.put(ix + 1, value);
        return this;*/
        throw new UnsupportedOperationException();

    }

    @Override
    public IComplexNDArray putImag(int rowIndex, int columnIndex, float value) {
        return putReal(new int[]{rowIndex, columnIndex}, value);
    }

    @Override
    public IComplexNDArray put(int[] indexes, float value) {
        return put(indexes, (double) value);
    }

    @Override
    public IComplexNDArray neqi(IComplexNumber other) {
        IComplexNDArray ret = linearView();
        for (int i = 0; i < length(); i++) {
            IComplexNumber num = ret.getComplex(i);
            ret.putScalar(i, num.neqc(other));
        }
        return this;
    }

    @Override
    public IComplexNDArray neq(IComplexNumber other) {
        return dup().neqi(other);
    }

    @Override
    public IComplexNDArray lt(IComplexNumber other) {
        return dup().lti(other);
    }

    @Override
    public IComplexNDArray lti(IComplexNumber other) {
        IComplexNDArray ret = linearView();
        for (int i = 0; i < length(); i++) {
            IComplexNumber num = ret.getComplex(i);
            ret.putScalar(i, num.lt(other));
        }
        return this;
    }

    @Override
    public IComplexNDArray eq(IComplexNumber other) {
        return dup().eqi(other);
    }

    @Override
    public IComplexNDArray eqi(IComplexNumber other) {
        return dup().eqi(other);
    }

    @Override
    public IComplexNDArray gt(IComplexNumber other) {
        return dup().gti(other);
    }

    @Override
    public IComplexNDArray gti(IComplexNumber other) {
        IComplexNDArray ret = linearView();
        for (int i = 0; i < length(); i++) {
            IComplexNumber num = ret.getComplex(i);
            ret.putScalar(i, num.gt(other));
        }
        return this;
    }

    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public IComplexNDArray transposei() {
        return Nd4j.createComplex(super.transposei());
    }

    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public IComplexNDArray transpose() {
        return transposei();
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for (int i = 0; i < length(); i++) {
            cResult.putScalar(i, linear.getComplex(i).addi(n));
        }

        return result;

    }


    @Override
    public IComplexNDArray subi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for (int i = 0; i < length(); i++) {
            cResult.putScalar(i, linear.getComplex(i).subi(n));
        }

        return result;

    }


    @Override
    public IComplexNDArray muli(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for (int i = 0; i < length(); i++) {
            IComplexNumber n3 = linear.getComplex(i);
            cResult.putScalar(i, n3.mul(n));
        }

        return result;

    }


    @Override
    public IComplexNDArray divi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for (int i = 0; i < length(); i++) {
            cResult.putScalar(i, linear.getComplex(i).div(n));
        }

        return result;

    }


    @Override
    public IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for (int i = 0; i < length(); i++) {
            cResult.putScalar(i, n.sub(linear.getComplex(i)));
        }

        return result;

    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for (int i = 0; i < length(); i++) {
            cResult.putScalar(i, n.div(linear.getComplex(i)));
        }

        return result;

    }


    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     *
     * @param shape
     * @return
     */
    @Override
    public IComplexNDArray reshape(int...shape) {
        return (IComplexNDArray) super.reshape(shape);
    }

    @Override
    public IComplexNDArray reshape(char order, int... newShape) {
        return (IComplexNDArray) super.reshape(order, newShape);
    }

    @Override
    public IComplexNDArray reshape(char order, int rows, int columns) {
        return (IComplexNDArray) super.reshape(order, rows, columns);
    }

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    @Override
    public IComplexNDArray assign(Number value) {
        IComplexNDArray one = linearView();
        for (int i = 0; i < one.length(); i++)
            one.putScalar(i, Nd4j.createDouble(value.doubleValue(), 0));
        return this;
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
        return rdivi(other, this);
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
        return dup().rdivi(other, result);
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
        return dup().rsubi(other, result);
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
        return rsubi(other, this);
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
        IComplexNumber max = reshape.getComplex(0);

        for (int i = 1; i < reshape.length(); i++) {
            IComplexNumber curr = reshape.getComplex(i);
            double val = curr.realComponent().doubleValue();
            if (val > curr.realComponent().doubleValue())
                max = curr;

        }
        return max;
    }


    public IComplexNumber min() {
        IComplexNDArray reshape = ravel();
        IComplexNumber min = reshape.getComplex(0);
        for (int i = 1; i < reshape.length(); i++) {
            IComplexNumber curr = reshape.getComplex(i);
            double val = curr.realComponent().doubleValue();
            if (val < curr.realComponent().doubleValue())
                min = curr;

        }
        return min;
    }



    /**
     * Reshape the matrix. Number of elements must not change.
     *
     * @param newRows
     * @param newColumns
     */
    @Override
    public IComplexNDArray reshape(int newRows, int newColumns) {
        return reshape(new int[]{newRows, newColumns});
    }


    /**
     * Get the specified column
     *
     * @param c
     */
    @Override
    public IComplexNDArray getColumn(int c) {
        return (IComplexNDArray) super.getColumn(c);

    }


    /**
     * Get a copy of a row.
     *
     * @param r
     */
    @Override
    public IComplexNDArray getRow(int r) {
        return (IComplexNDArray) super.getRow(r);


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
        IComplexNDArray n = null;
        if (!(o instanceof IComplexNDArray))
            return false;

        if (n == null)
            n = (IComplexNDArray) o;

        //epsilon equals
        if (isScalar() && n.isScalar()) {
            IComplexNumber c = n.getComplex(0);
            return FastMath.abs(getComplex(0).sub(c).realComponent().doubleValue()) < Nd4j.EPS_THRESHOLD;
        }
        else if (isVector() && n.isVector()) {
            for (int i = 0; i < length; i++) {
                IComplexNumber nComplex = n.getComplex(i);
                IComplexNumber thisComplex = getComplex(i);
                if(!nComplex.equals(thisComplex))
                    return false;
            }

            return true;

        }

        if (!Shape.shapeEquals(shape(), n.shape()))
            return false;
        //epsilon equals
        if (isScalar()) {
            IComplexNumber c = n.getComplex(0);
            return getComplex(0).sub(c).absoluteValue().doubleValue() < Nd4j.EPS_THRESHOLD;
        }
        else if (isVector()) {
            for (int i = 0; i < length; i++) {
                IComplexNumber curr = getComplex(i);
                IComplexNumber comp = n.getComplex(i);
                if(!curr.equals(comp))
                    return false;
            }

            return true;


        }

        for (int i = 0; i < slices(); i++) {
            IComplexNDArray sliceI = slice(i);
            IComplexNDArray nSliceI = n.slice(i);
            if (!sliceI.equals(nSliceI))
                return false;
        }

        return true;


    }


    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public IComplexNDArray broadcast(int[] shape) {
        return (IComplexNDArray) super.broadcast(shape);
    }


    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     *
     * @return the individual item in this ndarray
     */
    @Override
    public Object element() {
        if (!isScalar())
            throw new IllegalStateException("Unable to getScalar the element of a non scalar");
        int idx = linearIndex(0);
        return Nd4j.createDouble(data.getDouble(idx), data.getDouble(idx + 1));
    }


    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public IComplexNDArray permute(int[] rearrange) {
        return (IComplexNDArray) super.permute(rearrange);
    }


    /**
     * Flattens the array for linear indexing
     *
     * @return the flattened version of this array
     */
    @Override
    public IComplexNDArray ravel() {

        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Length can not be >= Integer.MAX_VALUE");
        IComplexNDArray ret = Nd4j.createComplex((int)length, ordering());
        IComplexNDArray linear = linearView();
        for(int i = 0; i < length(); i++) {
            ret.putScalar(i,linear.getComplex(i));
        }

        return ret;


    }


    /**
     * Generate string representation of the matrix.
     */
    @Override
    public String toString() {
        if (isScalar()) {
            return element().toString();
        } else if (isVector()) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            if(length >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Length can not be >= Integer.MAX_VALUE");
            long numElementsToPrint = Nd4j.MAX_ELEMENTS_PER_SLICE < 0 ? length : Nd4j.MAX_ELEMENTS_PER_SLICE;
            for (int i = 0; i < length; i++) {
                sb.append(getComplex(i));
                if (i < length - 1)
                    sb.append(" ,");
                if (i >= numElementsToPrint) {
                    long numElementsLeft = length - i;
                    //set towards the end of the buffer
                    if (numElementsLeft > numElementsToPrint) {
                        i += numElementsLeft - numElementsToPrint - 1;
                        sb.append(" ,... ,");
                    }
                }

            }
            sb.append("]\n");
            return sb.toString();
        }


       /* StringBuilder sb = new StringBuilder();
        int length = shape[0];
        sb.append("[");
        if (length > 0) {
            sb.append(slice(0).toString());
            int slices = Nd4j.MAX_SLICES_TO_PRINT > 0 ? Nd4j.MAX_SLICES_TO_PRINT : slices();
            if (slices > slices())
                slices = slices();
            for (int i = 1; i < slices; i++) {
                sb.append(slice(i).toString());
                if (i < length - 1)
                    sb.append(" ,");

            }


        }
        sb.append("]\n");
        return sb.toString();*/
        throw new UnsupportedOperationException();

    }


}
