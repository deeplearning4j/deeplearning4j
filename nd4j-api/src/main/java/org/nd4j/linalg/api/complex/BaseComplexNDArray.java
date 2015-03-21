/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.complex;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.nd4j.linalg.util.Shape;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

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

    public BaseComplexNDArray(DataBuffer data, int[] shape, int[] stride) {
        this(data, shape, stride, 0, Nd4j.order());
    }

    public BaseComplexNDArray(float[] data) {
        super(data);
    }


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

    public BaseComplexNDArray(int[] shape, int offset, char ordering) {
        this(Nd4j.createBuffer(ArrayUtil.prod(shape) * 2),
                shape, Nd4j.getComplexStrides(shape, ordering),
                offset, ordering);
    }

    public BaseComplexNDArray(int[] shape) {
        this(Nd4j.createBuffer(ArrayUtil.prod(shape) * 2), shape, Nd4j.getComplexStrides(shape));
    }


    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        this(data, shape, stride, 0, ordering);
    }

    public BaseComplexNDArray(int[] shape, char ordering) {
        this(Nd4j.createBuffer(ArrayUtil.prod(shape) * 2), shape, Nd4j.getComplexStrides(shape, ordering), 0, ordering);
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


        this.ordering = ordering;
        this.data = Nd4j.createBuffer(ArrayUtil.prod(shape) * 2);
        this.stride = stride;
        init(shape);

        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            putScalar(count, list.get(i));
            count++;
        }
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
        this(slices, shape, ordering == NDArrayFactory.C ? ArrayUtil.calcStrides(shape, 2) : ArrayUtil.calcStridesFortran(shape, 2), ordering);


    }

    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, int offset, Character order) {
        this.data = Nd4j.createBuffer(data);
        this.stride = stride;
        this.offset = offset;
        this.ordering = order;
        init(shape);
    }

    public BaseComplexNDArray(DataBuffer data) {
        super(data);
    }

    public BaseComplexNDArray(DataBuffer data, int[] shape, int[] stride, int offset) {
        this.data = data;
        this.stride = stride;
        this.offset = offset;
        this.ordering = Nd4j.order();
        init(shape);

    }

    public BaseComplexNDArray(IComplexNumber[] data, int[] shape, int[] stride, int offset, char ordering) {
        this(shape, stride, offset, ordering);
        assert data.length <= length;
        for (int i = 0; i < data.length; i++) {
            putScalar(i, data[i]);
        }
    }

    public BaseComplexNDArray(DataBuffer data, int[] shape) {
        this(shape);
        this.data = data;
    }

    public BaseComplexNDArray(IComplexNumber[] data, int[] shape, int[] stride, int offset) {
        this(data, shape, stride, offset, Nd4j.order());
    }

    public BaseComplexNDArray(IComplexNumber[] data, int[] shape, int offset, char ordering) {
        this(data, shape, Nd4j.getComplexStrides(shape), offset, ordering);
    }

    public BaseComplexNDArray(DataBuffer buffer, int[] shape, int offset, char ordering) {
        this(buffer, shape, Nd4j.getComplexStrides(shape), offset, ordering);
    }

    public BaseComplexNDArray(DataBuffer buffer, int[] shape, int offset) {
        this(buffer, shape, Nd4j.getComplexStrides(shape), offset, Nd4j.order());
    }

    public BaseComplexNDArray(float[] data, Character order) {
        this(data, new int[]{data.length / 2}, order);
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
        init(shape);
        for (int i = 0; i < length; i++)
            put(i, newData[i].asDouble());

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
        this.stride = stride;
        init(shape);
        for (int i = 0; i < length; i++)
            put(i, newData[i].asDouble());

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
        this.ordering = ordering;
        init(shape);
        for (int i = 0; i < length; i++)
            put(i, newData[i]);

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
        this.ordering = ordering;
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

    protected void copyFromReal(INDArray real) {
        INDArray linear = real.linearView();
        IComplexNDArray thisLinear = linearView();
        for (int i = 0; i < linear.length(); i++) {
            thisLinear.putScalar(i, Nd4j.createComplexNumber(linear.getDouble(i), 0));
        }
    }

    protected void copyRealTo(INDArray arr) {
        INDArray linear = arr.linearView();
        IComplexNDArray thisLinear = linearView();
        for (int i = 0; i < linear.length(); i++) {
            arr.putScalar(i, thisLinear.getReal(i));
        }

    }

    @Override
    public int blasOffset() {
        return offset > 0 ? offset() / 2 : offset();
    }

    @Override
    public IComplexNDArray linearViewColumnOrder() {
        return Nd4j.createComplex(data, new int[]{length, 1}, offset());
    }

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    @Override
    public IComplexNDArray linearView() {
        if (isVector())
            return this;
        if (linearView == null)
            resetLinearView();
        return (IComplexNDArray) linearView;
    }

    @Override
    public void resetLinearView() {
        linearView = Nd4j.createComplex(data, new int[]{1, length}, stride(), offset());
    }

    @Override
    public IComplexNumber getComplex(int i, IComplexNumber result) {
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
        int idx = index(j, i);
        data.put(idx, conji.realComponent().doubleValue());
        data.put(idx + 1, conji.imaginaryComponent().doubleValue());
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
        IComplexNDArray ret = Nd4j.createComplex(shape());
        IComplexNDArray linear = linearView();
        IComplexNDArray retLinear = ret.linearView();
        for (int i = 0; i < ret.length(); i++) {
            retLinear.putScalar(i, linear.getComplex(i));
        }
        return ret;
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

        if (columns() == 1 && rowVector.isScalar()) {
            if (rowVector instanceof IComplexNDArray) {
                IComplexNDArray rowVectorComplex = (IComplexNDArray) rowVector;
                switch (operation) {
                    case 'a':
                        addi(rowVectorComplex.getComplex(0));
                        break;
                    case 's':
                        subi(rowVectorComplex.getComplex(0));
                        break;
                    case 'm':
                        muli(rowVectorComplex.getComplex(0));
                        break;
                    case 'd':
                        divi(rowVectorComplex.getComplex(0));
                        break;
                    case 'h':
                        rsubi(rowVectorComplex.getComplex(0));
                        break;
                    case 't':
                        rdivi(rowVectorComplex.getComplex(0));
                        break;
                }

            } else {
                switch (operation) {
                    case 'a':
                        addi(rowVector.getDouble(0));
                        break;
                    case 's':
                        subi(rowVector.getDouble(0));
                        break;
                    case 'm':
                        muli(rowVector.getDouble(0));
                        break;
                    case 'd':
                        divi(rowVector.getDouble(0));
                        break;
                    case 'h':
                        rsubi(rowVector.getDouble(0));
                        break;
                    case 't':
                        rdivi(rowVector.getDouble(0));
                        break;
                }

            }

            return this;
        }

        assertRowVector(rowVector);
        for (int i = 0; i < rows(); i++) {
            switch (operation) {

                case 'a':
                    getRow(i).addi(rowVector);
                    break;
                case 's':
                    getRow(i).subi(rowVector);
                    break;
                case 'm':
                    getRow(i).muli(rowVector);
                    break;
                case 'd':
                    getRow(i).divi(rowVector);
                    break;
                case 'h':
                    getRow(i).rsubi(rowVector);
                    break;
                case 't':
                    getRow(i).rdivi(rowVector);
                    break;
            }
        }


        return this;
    }

    @Override
    protected IComplexNDArray doColumnWise(INDArray columnVector, char operation) {
        if (rows() == 1 && columnVector.isScalar()) {
            if (columnVector instanceof IComplexNDArray) {
                IComplexNDArray columnVectorComplex = (IComplexNDArray) columnVector;
                switch (operation) {
                    case 'a':
                        addi(columnVectorComplex.getComplex(0));
                        break;
                    case 's':
                        subi(columnVectorComplex.getComplex(0));
                        break;
                    case 'm':
                        muli(columnVectorComplex.getComplex(0));
                        break;
                    case 'd':
                        divi(columnVectorComplex.getComplex(0));
                        break;
                    case 'h':
                        rsubi(columnVectorComplex.getComplex(0));
                        break;
                    case 't':
                        rdivi(columnVectorComplex.getComplex(0));
                        break;
                }

            } else {
                switch (operation) {
                    case 'a':
                        addi(columnVector.getDouble(0));
                        break;
                    case 's':
                        subi(columnVector.getDouble(0));
                        break;
                    case 'm':
                        muli(columnVector.getDouble(0));
                        break;
                    case 'd':
                        divi(columnVector.getDouble(0));
                        break;
                    case 'h':
                        rsubi(columnVector.getDouble(0));
                        break;
                    case 't':
                        rdivi(columnVector.getDouble(0));
                        break;
                }

            }

            return this;
        }

        assertColumnVector(columnVector);
        for (int i = 0; i < columns(); i++) {
            IComplexNDArray slice = slice(i, 0);
            switch (operation) {

                case 'a':
                    slice.addi(columnVector);
                    break;
                case 's':
                    slice.subi(columnVector);
                    break;
                case 'm':
                    slice.muli(columnVector);
                    break;
                case 'd':
                    slice.divi(columnVector);
                    break;
                case 'h':
                    slice.rsubi(columnVector);
                    break;
                case 't':
                    slice.rdivi(columnVector);
                    break;
            }
        }

        return this;
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
    public IComplexNDArray put(NDArrayIndex[] indices, INDArray element) {
        if (Indices.isContiguous(indices)) {
            IComplexNDArray get = get(indices);
            IComplexNDArray linear = get.linearView();
            IComplexNDArray imag = element instanceof IComplexNDArray ? (IComplexNDArray) element : Nd4j.createComplex(element);
            IComplexNDArray elementLinear = imag.linearView();
            if (element.isScalar()) {
                for (int i = 0; i < linear.length(); i++) {
                    linear.putScalar(i, elementLinear.getComplex(0));
                }
            }

            if (Shape.shapeEquals(element.shape(), get.shape()) || element.length() <= get.length()) {
                for (int i = 0; i < elementLinear.length(); i++) {
                    linear.putScalar(i, elementLinear.getComplex(i));
                }
            }


        } else {
            if (isVector()) {
                assert indices.length == 1 : "Indices must only be of length 1.";
                assert element.isScalar() || element.isVector() : "Unable to assign elements. Element is not a vector.";
                assert indices[0].length() == element.length() : "Number of specified elements in index does not match length of element.";
                int[] assign = indices[0].indices();
                IComplexNDArray imag = element instanceof IComplexNDArray ? (IComplexNDArray) element : Nd4j.createComplex(element);
                IComplexNDArray elementLinear = imag.linearView();

                for (int i = 0; i < element.length(); i++) {
                    putScalar(assign[i], elementLinear.getComplex(i));
                }

                return this;

            }

            if (element.isVector())
                slice(indices[0].indices()[0]).put(Arrays.copyOfRange(indices, 1, indices.length), element);


            else {
                for (int i = 0; i < element.slices(); i++) {
                    INDArray slice = slice(indices[0].indices()[i]);
                    slice.put(Arrays.copyOfRange(indices, 1, indices.length), element.slice(i));
                }
            }

        }

        return this;
    }

    @Override
    public IComplexNDArray normmax(int dimension) {
        return Nd4j.createComplex(super.normmax(dimension));
    }

    @Override
    public IComplexNDArray prod(int dimension) {
        return Nd4j.createComplex(super.prod(dimension));
    }

    @Override
    public IComplexNDArray mean(int dimension) {
        return Nd4j.createComplex(super.mean(dimension));
    }

    @Override
    public IComplexNDArray var(int dimension) {
        return Nd4j.createComplex(super.var(dimension));
    }

    @Override
    public IComplexNDArray max(int dimension) {
        return Nd4j.createComplex(super.max(dimension));
    }

    @Override
    public IComplexNDArray sum(int dimension) {
        return Nd4j.createComplex(super.sum(dimension));
    }

    @Override
    public IComplexNDArray min(int dimension) {
        return Nd4j.createComplex(super.min(dimension));
    }

    @Override
    public IComplexNDArray norm1(int dimension) {
        return Nd4j.createComplex(super.norm1(dimension));
    }

    @Override
    public IComplexNDArray std(int dimension) {
        return Nd4j.createComplex(super.std(dimension));
    }

    @Override
    public IComplexNDArray norm2(int dimension) {
        return Nd4j.createComplex(super.norm2(dimension));
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
        return put(i, j, Nd4j.scalar(element));
    }


    /**
     * @param indexes
     * @param value
     * @return
     */
    @Override
    public IComplexNDArray put(int[] indexes, double value) {
        int ix = offset;
        if (indexes.length != shape.length)
            throw new IllegalArgumentException("Unable to applyTransformToDestination values: number of indices must be equal to the shape");

        for (int i = 0; i < shape.length; i++)
            ix += indexes[i] * stride[i];


        data.put(ix, value);
        return this;
    }


    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to applyTransformToDestination
     * @return this for chainability
     */
    @Override
    public IComplexNDArray putSlice(int slice, IComplexNDArray put) {
        if (isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            putScalar(0, put.getDouble(0));
            return this;
        } else if (isVector()) {
            assert put.isScalar() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            putScalar(slice, put.getDouble(0));
            return this;
        }


        assertSlice(put, slice);


        IComplexNDArray view = slice(slice);

        if (put.isScalar())
            putScalar(slice, put.getDouble(0));
        else if (put.isVector())
            for (int i = 0; i < put.length(); i++)
                view.putScalar(i, put.getComplex(i));
        else if (put.shape().length == 2)
            for (int i = 0; i < put.rows(); i++)
                for (int j = 0; j < put.columns(); j++)
                    view.put(i, j, put.getDouble(i, j));

        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for (int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i, view.slice(i));

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
        int[] shape = ArrayUtil.range(0, shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
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
        data.put(2 * index(rowIndex, columnIndex) + offset, value);
        return this;
    }


    @Override
    public int linearIndex(int i) {
        int realStride = majorStride();
        int idx = offset + (i * realStride);
        if (idx >= data.length())
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }


    @Override
    public IComplexNDArray putImag(int rowIndex, int columnIndex, double value) {
        data.put(index(rowIndex, columnIndex) + 1 + offset, value);
        return this;
    }

    @Override
    public IComplexNDArray putReal(int i, float v) {
        int idx = linearIndex(i);
        data.put(idx, v);
        return this;
    }

    @Override
    public IComplexNDArray putImag(int i, float v) {
        int idx = linearIndex(i);
        data.put(idx * 2 + 1, v);
        return this;
    }


    @Override
    public IComplexNumber getComplex(int i) {
        int idx = linearIndex(i);
        return Nd4j.createDouble(data.getDouble(idx), data.getDouble(idx + 1));
    }

    @Override
    public IComplexNumber getComplex(int i, int j) {
        int idx = index(i, j);
        return Nd4j.createDouble(data.getDouble(idx), data.getDouble(idx + 1));

    }

    /**
     * Get realComponent part of the matrix.
     */
    @Override
    public INDArray real() {
        INDArray ret = Nd4j.create(shape);
        copyRealTo(ret);
        return ret;
    }

    /**
     * Get imaginary part of the matrix.
     */
    @Override
    public INDArray imag() {
        INDArray ret = Nd4j.create(shape);
        Nd4j.getBlasWrapper().dcopy(length, data.asFloat(), 1, 2, ret.data().asFloat(), 0, 1);
        return ret;
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
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return Nd4j.scalar(Nd4j.createDouble(data.getDouble(ix), data.getDouble(ix + 1)));
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
     *
     * @return the off sets for the beginning of each slice
     */
    @Override
    public int[] endsForSlices() {
        int[] ret = new int[slices()];
        int currOffset = offset;
        for (int i = 0; i < slices(); i++) {
            ret[i] = (currOffset);
            currOffset += stride[0];
        }
        return ret;
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
        if (isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            put(0, put.getScalar(0));
            return this;
        } else if (isVector()) {
            assert put.isScalar() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            put(slice, put.getScalar(0));
            return this;
        }


        assertSlice(put, slice);


        IComplexNDArray view = slice(slice);

        if (put.isScalar())
            put(slice, put.getScalar(0));
        else if (put.isVector())
            for (int i = 0; i < put.length(); i++)
                view.put(i, put.getScalar(i));
        else if (put.shape().length == 2) {
            if (put instanceof IComplexNDArray) {
                IComplexNDArray complexPut = (IComplexNDArray) put;
                for (int i = 0; i < put.rows(); i++)
                    for (int j = 0; j < put.columns(); j++)
                        view.putScalar(i, j, complexPut.getComplex(i, j));


            }

            for (int i = 0; i < put.rows(); i++)
                for (int j = 0; j < put.columns(); j++)
                    view.putScalar(i, j, Nd4j.createDouble(put.getDouble(i, j), 0.0));


        } else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for (int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i, view.slice(i));

        }

        return this;
    }

    @Override
    public IComplexNDArray subArray(int[] offsets, int[] shape, int[] stride) {
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

        return Nd4j.createComplex(
                data
                , Arrays.copyOf(shape, shape.length)
                , stride
                , offset + ArrayUtil.dotProduct(offsets, stride)
        );
    }


    @Override
    public int majorStride() {
        if (offset == 0)
            return super.majorStride();
        else {
            //stride already taken in to account once by offset
            return super.majorStride();
        }
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
        if (!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
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
        return put(new int[]{i, j}, element);
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
        int offset = this.offset + (slice * majorStride());

        IComplexNDArray ret;
        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d ComplexNDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            ret = Nd4j.createComplex(
                    data,
                    ArrayUtil.empty(),
                    ArrayUtil.empty(),
                    offset, ordering);


            //slice of a matrix is a vector
        else if (shape.length == 2) {
            ret = Nd4j.createComplex(
                    data,
                    ArrayUtil.of(shape[1]),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset, ordering

            );

        } else {
            if (offset >= data.length())
                throw new IllegalArgumentException("Offset index is > data.length");
            ret = Nd4j.createComplex(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset, ordering);
        }
        return ret;
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
        int offset = this.offset + dimension * stride[slice];
        if (this.offset == 0)
            offset *= 2;
        IComplexNDArray ret;
        if (shape.length == 2) {
            int st = stride[1];
            if (st == 1) {
                return Nd4j.createComplex(
                        data,
                        new int[]{shape[dimension]},
                        offset, ordering);
            } else {
                return Nd4j.createComplex(
                        data,
                        new int[]{shape[dimension]},
                        new int[]{st},
                        offset);
            }


        }

        if (slice == 0)
            return slice(dimension);


        return Nd4j.createComplex(
                data,
                ArrayUtil.removeIndex(shape, dimension),
                ArrayUtil.removeIndex(stride, dimension),
                offset, ordering
        );
    }


    @Override
    protected void init(int[] shape) {
        this.shape = shape;

        if (this.shape.length == 1) {
            rows = 1;
            columns = this.shape[0];
        } else if (this.shape().length == 2) {
            if (shape[0] == 1) {
                this.shape = new int[1];
                this.shape[0] = shape[1];
                rows = 1;
                columns = shape[1];
            } else {
                rows = shape[0];
                columns = shape[1];
            }


        }

        //default row vector
        else if (this.shape.length == 1) {
            columns = this.shape[0];
            rows = 1;
        }

        //null character
        if (this.ordering == '\u0000')
            this.ordering = Nd4j.order();

        this.length = ArrayUtil.prod(this.shape);
        if (this.stride == null) {
            if (ordering == NDArrayFactory.FORTRAN)
                this.stride = ArrayUtil.calcStridesFortran(this.shape, 2);
            else
                this.stride = ArrayUtil.calcStrides(this.shape, 2);
        }

        //recalculate stride: this should only happen with row vectors
        if (this.stride.length != this.shape.length) {
            if (ordering == NDArrayFactory.FORTRAN)
                this.stride = ArrayUtil.calcStridesFortran(this.shape, 2);
            else
                this.stride = ArrayUtil.calcStrides(this.shape, 2);
        }
    }


    protected INDArray newShape(int[] newShape, char ordering) {
        if (Arrays.equals(newShape, this.shape()))
            return this;

        else if (Shape.isVector(newShape) && isVector()) {
            if (isRowVector() && Shape.isColumnVectorShape(newShape)) {
                return Nd4j.createComplex(data, newShape, new int[]{stride[0], 1}, offset);
            } else if (isColumnVector() && Shape.isRowVectorShape(newShape)) {
                return Nd4j.createComplex(data, newShape, new int[]{stride[1]}, offset);

            }
        }

        IComplexNDArray newCopy = this;
        int[] newStrides = null;
        //create a new copy of the ndarray
        if (shape().length > 1 && ((ordering == NDArrayFactory.C && this.ordering != NDArrayFactory.C) ||
                (ordering == NDArrayFactory.FORTRAN && this.ordering != NDArrayFactory.FORTRAN))) {
            newStrides = noCopyReshape(newShape, ordering);
            if (newStrides == null) {
                newCopy = Nd4j.createComplex(shape(), ordering);
                for (int i = 0; i < vectorsAlongDimension(0); i++) {
                    IComplexNDArray copyFrom = vectorAlongDimension(i, 0);
                    IComplexNDArray copyTo = newCopy.vectorAlongDimension(i, 0);
                    for (int j = 0; j < copyFrom.length(); j++) {
                        copyTo.putScalar(j, copyFrom.getDouble(i));
                    }
                }
            }


        }

        //needed to copy data
        if (newStrides == null)
            newStrides = Nd4j.getComplexStrides(newShape, ordering);

        return Nd4j.createComplex(newCopy.data(), newShape, newStrides, offset);


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
        for (int i = 0; i < shape.length; i++)
            newShape[i] *= shape[i];
        IComplexNDArray result = Nd4j.createComplex(newShape);
        //nd copy
        if (isScalar()) {
            for (int i = 0; i < result.length(); i++) {
                result.put(i, getScalar(0));

            }
        } else if (isMatrix()) {

            for (int c = 0; c < shape()[1]; c++) {
                for (int r = 0; r < shape()[0]; r++) {
                    for (int i = 0; i < rows(); i++) {
                        for (int j = 0; j < columns(); j++) {
                            result.put(r * rows() + i, c * columns() + j, getScalar(i, j));
                        }
                    }
                }
            }

        } else {
            int[] sliceRepmat = ArrayUtil.removeIndex(shape, 0);
            for (int i = 0; i < result.slices(); i++) {
                result.putSlice(i, repmat(sliceRepmat));
            }
        }

        return result;
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
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNumber element) {
        return put(indices, Nd4j.scalar(element));
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNDArray element) {
        if (Indices.isContiguous(indices)) {
            IComplexNDArray get = get(indices);
            IComplexNDArray linear = get.linearView();
            if (element.isScalar()) {
                for (int i = 0; i < linear.length(); i++) {
                    linear.putScalar(i, element.getComplex(0));
                }
            }

            if (Shape.shapeEquals(element.shape(), get.shape()) || element.length() <= get.length()) {
                IComplexNDArray elementLinear = element.linearView();

                for (int i = 0; i < elementLinear.length(); i++) {
                    linear.putScalar(i, elementLinear.getComplex(i));
                }
            }


        } else {
            if (isVector()) {
                assert indices.length == 1 : "Indices must only be of length 1.";
                assert element.isScalar() || element.isVector() : "Unable to assign elements. Element is not a vector.";
                assert indices[0].length() == element.length() : "Number of specified elements in index does not match length of element.";
                int[] assign = indices[0].indices();
                for (int i = 0; i < element.length(); i++) {
                    putScalar(assign[i], element.getComplex(i));
                }

                return this;

            }

            if (element.isVector())
                slice(indices[0].indices()[0]).put(Arrays.copyOfRange(indices, 1, indices.length), element);


            else {
                for (int i = 0; i < element.slices(); i++) {
                    INDArray slice = slice(indices[0].indices()[i]);
                    slice.put(Arrays.copyOfRange(indices, 1, indices.length), element.slice(i));
                }
            }

        }

        return this;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, Number element) {
        return put(indices, Nd4j.scalar(element));

    }

    @Override
    public IComplexNDArray putScalar(int i, IComplexNumber value) {
        int idx = linearIndex(i);
        data.put(idx, value.realComponent().doubleValue());
        data.put(idx + 1, value.imaginaryComponent().doubleValue());
        return this;
    }


    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    @Override
    public IComplexNDArray vectorAlongDimension(int index, int dimension) {
        assert dimension <= shape.length : "Invalid dimension " + dimension;
        if (ordering == NDArrayFactory.C) {

            if (dimension == shape.length - 1 && dimension != 0) {
                if (size(dimension) == 1)
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * 2);
                else
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * stride[0]);
            } else if (dimension == 0)
                return Nd4j.createComplex(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * 2);


            if (size(dimension) == 0)
                return Nd4j.createComplex(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * 2);

            return Nd4j.createComplex(data,
                    new int[]{shape[dimension], 1}
                    , new int[]{stride[dimension], 1},
                    offset + index * 2 * stride[0]);
        } else if (ordering == NDArrayFactory.FORTRAN) {

            if (dimension == shape.length - 1 && dimension != 0) {
                if (size(dimension) == 1) {
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * 2);
                } else
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * 2 * stride[0]);
            }

            if (size(dimension) == 1) {
                return Nd4j.createComplex(data,
                        new int[]{1, shape[dimension]}
                        , ArrayUtil.removeIndex(stride, 0),
                        offset + index * 2);
            } else
                return Nd4j.createComplex(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * 2 * stride[0]);
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
    public IComplexNDArray cumsumi(int dimension) {
        if (isVector()) {
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


        return this;
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
        assert broadCastable.length == shape.length : "The broadcastable dimensions must be the same length as the current shape";

        boolean broadcast = false;
        Set<Object> set = new HashSet<>();
        for (int i = 0; i < rearrange.length; i++) {
            set.add(rearrange[i]);
            if (rearrange[i] instanceof Integer) {
                Integer j = (Integer) rearrange[i];
                if (j >= broadCastable.length)
                    throw new IllegalArgumentException("Illegal dimension, dimension must be < broadcastable.length (aka the real dimensions");
            } else if (rearrange[i] instanceof Character) {
                Character c = (Character) rearrange[i];
                if (c != 'x')
                    throw new IllegalArgumentException("Illegal input: Must be x");
                broadcast = true;

            } else
                throw new IllegalArgumentException("Only characters and integers allowed");
        }

        //just do permute
        if (!broadcast) {
            int[] ret = new int[rearrange.length];
            for (int i = 0; i < ret.length; i++)
                ret[i] = (Integer) rearrange[i];
            return permute(ret);
        } else {
            List<Integer> drop = new ArrayList<>();
            for (int i = 0; i < broadCastable.length; i++) {
                if (!set.contains(i)) {
                    if (broadCastable[i])
                        drop.add(i);
                    else
                        throw new IllegalArgumentException("We can't drop the given dimension because its not broadcastable");
                }

            }


            //list of dimensions to keep
            int[] shuffle = new int[broadCastable.length];
            int count = 0;
            for (int i = 0; i < rearrange.length; i++) {
                if (rearrange[i] instanceof Integer) {
                    shuffle[count++] = (Integer) rearrange[i];
                }
            }


            List<Integer> augment = new ArrayList<>();
            for (int i = 0; i < rearrange.length; i++) {
                if (rearrange[i] instanceof Character)
                    augment.add(i);
            }

            Integer[] augmentDims = augment.toArray(new Integer[1]);

            count = 0;

            int[] newShape = new int[shuffle.length + drop.size()];
            for (int i = 0; i < newShape.length; i++) {
                if (i < shuffle.length) {
                    newShape[count++] = shuffle[i];
                } else
                    newShape[count++] = drop.get(i);
            }


            IComplexNDArray ret = permute(newShape);
            List<Integer> newDims = new ArrayList<>();
            int[] shape = Arrays.copyOfRange(ret.shape(), 0, shuffle.length);
            for (int i = 0; i < shape.length; i++) {
                newDims.add(shape[i]);
            }

            for (int i = 0; i < augmentDims.length; i++) {
                newDims.add(augmentDims[i], 1);
            }

            int[] toReshape = ArrayUtil.toArray(newDims);


            ret = ret.reshape(toReshape);
            return ret;

        }


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
    public INDArray putScalar(int[] indexes, IComplexNumber complexNumber) {
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }

        data.put(ix, complexNumber.asFloat().realComponent().doubleValue());
        data.put(ix + 1, complexNumber.asFloat().imaginaryComponent().doubleValue());

        return this;
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
    public IComplexNDArray get(NDArrayIndex... indexes) {
        //fill in to match the rest of the dimensions: aka grab all the content
        //in the dimensions not filled in
        //also prune indices greater than the shape to be the shape instead

        indexes = Indices.adjustIndices(shape(), indexes);


        int[] offsets = Indices.offsets(indexes);
        int[] shape = Indices.shape(shape(), indexes);

        if (ArrayUtil.prod(shape) > length())
            return this;

        //no stride will help here, need to do manually

        if (!Indices.isContiguous(indexes)) {
            IComplexNDArray ret = Nd4j.createComplex(shape);
            if (ret.isVector() && isVector()) {
                int[] indices = indexes[0].indices();
                for (int i = 0; i < ret.length(); i++) {
                    ret.putScalar(i, getComplex(indices[i]));
                }

                return ret;
            }
            for (int i = 0; i < ret.slices(); i++) {
                IComplexNDArray putSlice = slice(i).get(Arrays.copyOfRange(indexes, 1, indexes.length));
                ret.putSlice(i, putSlice);

            }

            return ret;
        }

        int[] strides = ordering == 'f' ? ArrayUtil.calcStridesFortran(shape, 2) : ArrayUtil.copy(stride());

        if (offsets.length != shape.length)
            offsets = Arrays.copyOfRange(offsets, 0, shape.length);

        if (strides.length != shape.length)
            strides = Arrays.copyOfRange(offsets, 0, shape.length);


        return subArray(offsets, shape, strides);
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
        IComplexNDArray rows = Nd4j.createComplex(rows(), cindices.length);
        for (int i = 0; i < cindices.length; i++) {
            rows.putColumn(i, getColumn(cindices[i]));
        }
        return rows;
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
        assert toPut.isVector() && toPut.length() == columns : "Illegal length for row " + toPut.length() + " should have been " + columns;
        IComplexNDArray r = getRow(row);
        if (toPut instanceof IComplexNDArray) {
            IComplexNDArray putComplex = (IComplexNDArray) toPut;
            for (int i = 0; i < r.length(); i++)
                r.putScalar(i, putComplex.getComplex(i));
        } else {
            for (int i = 0; i < r.length(); i++)
                r.putScalar(i, Nd4j.createDouble(toPut.getDouble(i), 0));
        }

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
        int idx = linearIndex(i);
        return Nd4j.scalar(Nd4j.createDouble(data.getDouble(idx), data.getDouble(idx + 1)));
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

    private void put(int i, float element) {
        int idx = linearIndex(i);
        data.put(idx, element);
        data.put(idx + 1, 0.0);
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
        int[] shape = {rows(), other.columns()};
        return mmuli(other, Nd4j.createComplex(shape));
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
            return muli(otherArray.getDouble(0), resultArray);
        }
        if (isScalar()) {
            return otherArray.muli(getDouble(0), resultArray);
        }

        /* check sizes and resize if necessary */
        //assertMultipliesWith(other);


        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            IComplexNDArray temp = Nd4j.createComplex(resultArray.shape(), ArrayUtil.calcStridesFortran(resultArray.shape(), 2));

            if (otherArray.columns() == 1) {
                if (data.dataType() == (DataBuffer.DOUBLE))
                    Nd4j.getBlasWrapper().gemv(Nd4j.UNIT.asDouble(), this, otherArray, Nd4j.ZERO.asDouble(), temp);
                else
                    Nd4j.getBlasWrapper().gemv(Nd4j.UNIT.asFloat(), this, otherArray, Nd4j.ZERO.asFloat(), temp);
            } else {
                if (data.dataType() == (DataBuffer.DOUBLE))
                    Nd4j.getBlasWrapper().gemm(Nd4j.UNIT.asDouble(), this, otherArray, Nd4j.ZERO.asDouble(), temp);
                else
                    Nd4j.getBlasWrapper().gemm(Nd4j.UNIT.asFloat(), this, otherArray, Nd4j.ZERO.asFloat(), temp);

            }

            Nd4j.getBlasWrapper().copy(temp, resultArray);


        } else {
            if (otherArray.columns() == 1)
                if (data.dataType() == (DataBuffer.DOUBLE))
                    Nd4j.getBlasWrapper().gemv(Nd4j.UNIT.asDouble(), this, otherArray, Nd4j.ZERO.asDouble(), resultArray);
                else
                    Nd4j.getBlasWrapper().gemv(Nd4j.UNIT.asFloat(), this, otherArray, Nd4j.ZERO.asFloat(), resultArray);
            else if (data.dataType() == (DataBuffer.FLOAT))
                Nd4j.getBlasWrapper().gemm(Nd4j.UNIT.asFloat(), this, otherArray, Nd4j.ZERO.asFloat(), resultArray);
            else
                Nd4j.getBlasWrapper().gemm(Nd4j.UNIT.asDouble(), this, otherArray, Nd4j.ZERO.asDouble(), resultArray);

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
            if (data.dataType() == (DataBuffer.DOUBLE)) {
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
            /*SimpleBlas.copy(this, result);
            SimpleBlas.axpy(1.0, other, result);*/
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
        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, n.sub(getComplex(i)));
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
        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, getComplex(i).div(n));
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
        for (int i = 0; i < length; i++)
            cResultLinear.putScalar(i, getComplex(i).mul(n));
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
        return addi(n, this);
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
        int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];

        data.put(ix, value);
        return this;
    }

    @Override
    public IComplexNDArray putImag(int[] indices, double value) {
        int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];

        data.put(ix + 1, value);
        return this;
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
        if (isRowVector())
            return Nd4j.createComplex(data, new int[]{shape[0], 1}, offset);
        else if (isColumnVector())
            return Nd4j.createComplex(data, new int[]{shape[0]}, offset);
        if (ordering() == NDArrayFactory.FORTRAN && isMatrix()) {
            IComplexNDArray reverse = Nd4j.createComplex(ArrayUtil.reverseCopy(shape));

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    reverse.putScalar(new int[]{j, i}, getComplex(i, j));
                }
            }

            return reverse;
        }


        IComplexNDArray ret = permute(ArrayUtil.range(shape.length - 1, -1));
        return ret;

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
            IComplexNumber num = n3.mul(n);
            cResult.putScalar(i, linear.getComplex(i).mul(n));
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
    public IComplexNDArray reshape(int[] shape) {
        return (IComplexNDArray) super.reshape(shape);
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
     * Converts the matrix to a one-dimensional array of doubles.
     */
    @Override
    public IComplexNumber[] toArray() {
        length = ArrayUtil.prod(shape);
        IComplexNumber[] ret = new IComplexNumber[length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = getComplex(i);
        return ret;
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
        if (shape.length == 2) {
            if (ordering == NDArrayFactory.C) {
                IComplexNDArray ret = Nd4j.createComplex(
                        data,
                        new int[]{shape[0]},
                        new int[]{stride[0]},
                        offset + (c * 2), ordering
                );

                return ret;
            } else {
                IComplexNDArray ret = Nd4j.createComplex(
                        data,
                        new int[]{shape[0]},
                        new int[]{stride[0]},
                        offset + (c * 2), ordering
                );

                return ret;
            }

        } else if (isColumnVector() && c == 0)
            return this;

        else
            throw new IllegalArgumentException("Unable to getFromOrigin column of non 2d matrix");

    }


    /**
     * Get a copy of a row.
     *
     * @param r
     */
    @Override
    public IComplexNDArray getRow(int r) {
        if (shape.length == 2) {
            if (ordering == NDArrayFactory.C) {
                IComplexNDArray ret = Nd4j.createComplex(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + (r * 2) * columns(),
                        ordering
                );
                return ret;
            } else {
                IComplexNDArray ret = Nd4j.createComplex(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + (r * 2),
                        ordering
                );
                return ret;
            }


        } else if (isRowVector() && r == 0)
            return this;


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
        IComplexNDArray n = null;
        if (!(o instanceof IComplexNDArray))
            return false;

        if (n == null)
            n = (IComplexNDArray) o;

        //epsilon equals
        if (isScalar() && n.isScalar()) {
            IComplexNumber c = n.getComplex(0);
            return Math.abs(getComplex(0).sub(c).realComponent().doubleValue()) < 1e-6;
        } else if (isVector() && n.isVector()) {
            for (int i = 0; i < length; i++) {
                double curr = getComplex(i).realComponent().doubleValue();
                double comp = n.getComplex(i).realComponent().doubleValue();
                double currImag = getComplex(i).imaginaryComponent().doubleValue();
                double compImag = n.getComplex(i).imaginaryComponent().doubleValue();
                if (Math.abs(curr - comp) > 1e-3 || Math.abs(currImag - compImag) > 1e-3)
                    return false;
            }

            return true;

        }

        if (!Shape.shapeEquals(shape(), n.shape()))
            return false;
        //epsilon equals
        if (isScalar()) {
            IComplexNumber c = n.getComplex(0);
            return getComplex(0).sub(c).absoluteValue().doubleValue() < 1e-6;
        } else if (isVector()) {
            for (int i = 0; i < length; i++) {
                IComplexNumber curr = getComplex(i);
                IComplexNumber comp = n.getComplex(i);
                if (curr.sub(comp).absoluteValue().doubleValue() > 1e-6)
                    return false;
            }

            return true;


        }

        for (int i = 0; i < slices(); i++) {
            if (!(slice(i).equals(n.slice(i))))
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
        if (Shape.shapeEquals(shape, shape()))
            return this;
        boolean compatible = true;
        int count = shape.length - 1;
        int thisCount = this.shape.length - 1;
        for (int i = shape.length - 1; i > 0; i--) {
            if (count < 0 || thisCount < 0)
                break;
            if (shape[count] != shape()[thisCount] && shape[count] != 1 && shape()[thisCount] != 1) {
                compatible = false;
                break;
            }

            count--;
            thisCount--;
        }

        if (!compatible)
            throw new IllegalArgumentException("Incompatible broadcast from " + Arrays.toString(shape()) + " to " + Arrays.toString(shape));
        //broadcast to shape
        if (isScalar()) {
            IComplexNDArray ret = Nd4j.createComplex(Nd4j.valueArrayOf(shape, getDouble(0)));
            return ret;
        } else if (isColumnVector() && Shape.isMatrix(shape)) {
            IComplexNDArray ret = Nd4j.createComplex(shape);
            for (int i = 0; i < ret.columns(); i++)
                ret.putColumn(i, this.dup());


            return ret;
        }


        int[] retShape = new int[shape.length];

        for (int i = 0; i < retShape.length; i++) {
            if (i < shape().length)
                retShape[i] = Math.max(shape[i], shape()[i]);
            else
                retShape[i] = shape[i];
        }

        IComplexNDArray ret = Nd4j.createComplex(retShape);
        IComplexNDArray linear = ret.linearView();
        IComplexNDArray thisLinear = linearView();
        int bufferIdx = 0;
        for (int i = 0; i < ret.length(); i++) {
            linear.putScalar(i, thisLinear.getComplex(bufferIdx));
            bufferIdx++;
            if (bufferIdx >= length())
                bufferIdx = 0;
        }

        return ret;
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
        if (rearrange.length < shape.length)
            return dup();

        checkArrangeArray(rearrange);
        int[] newDims = doPermuteSwap(shape, rearrange);
        int[] newStrides = doPermuteSwap(stride, rearrange);

        IComplexNDArray ret = Nd4j.createComplex(data, newDims, newStrides, offset, ordering);
        return ret;
    }


    /**
     * Flattens the array for linear indexing
     *
     * @return the flattened version of this array
     */
    @Override
    public IComplexNDArray ravel() {
        final IComplexNDArray ret = Nd4j.createComplex(length, ordering);
        final AtomicInteger counter = new AtomicInteger(0);

        SliceOp op = new SliceOp() {
            @Override
            public void operate(INDArray nd) {
                IComplexNDArray nd1 = (IComplexNDArray) nd;
                for (int i = 0; i < nd.length(); i++) {
                    int element = counter.getAndIncrement();
                    ret.putScalar(element, nd1.getComplex(i));


                }
            }
        };
        //row order
        if (ordering == NDArrayFactory.C) {
            iterateOverAllRows(op);
        }
        //column order
        else if (ordering == NDArrayFactory.FORTRAN) {
            iterateOverAllColumns(op);
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
            int numElementsToPrint = Nd4j.MAX_ELEMENTS_PER_SLICE < 0 ? length : Nd4j.MAX_ELEMENTS_PER_SLICE;
            for (int i = 0; i < length; i++) {
                sb.append(getComplex(i));
                if (i < length - 1)
                    sb.append(" ,");
                if (i >= numElementsToPrint) {
                    int numElementsLeft = length - i;
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


        StringBuilder sb = new StringBuilder();
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
        return sb.toString();
    }


}
