package org.nd4j.linalg.api.complex;


import static  org.nd4j.linalg.util.ArrayUtil.calcStrides;
import static org.nd4j.linalg.util.ArrayUtil.calcStridesFortran;
import static  org.nd4j.linalg.util.ArrayUtil.reverseCopy;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.DimensionSlice;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.nd4j.linalg.ops.reduceops.Ops;
import org.nd4j.linalg.ops.reduceops.complex.ComplexOps;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.*;


import java.util.*;
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
public abstract class BaseComplexNDArray extends BaseNDArray implements IComplexNDArray {

    public BaseComplexNDArray() {}


    public BaseComplexNDArray(float[] data) {
        super(data);
    }

    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data     the data to use
     * @param shape    the shape of the ndarray
     * @param ordering
     */
    public BaseComplexNDArray(float[] data, int[] shape, char ordering) {
        this(data,shape, Nd4j.getComplexStrides(shape, ordering),0,ordering);
    }

    public BaseComplexNDArray(int[] shape, int offset, char ordering) {
        this(new float[ArrayUtil.prod(shape) * 2],
                shape, Nd4j.getComplexStrides(shape, ordering),
                offset,ordering);
    }

    public BaseComplexNDArray(int[] shape) {
        this(new float[ArrayUtil.prod(shape) * 2],shape, Nd4j.getComplexStrides(shape));
    }



    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        this(data,shape,stride,0,ordering);
    }

    public BaseComplexNDArray(int[] shape, char ordering) {
        this(new float[ArrayUtil.prod(shape) * 2],shape, Nd4j.getComplexStrides(shape, ordering),ordering);
    }


    /**
     * Initialize the given ndarray as the real component
     * @param m the real component
     * @param stride the stride of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public BaseComplexNDArray(INDArray m,int[] stride,char ordering) {
        this(m.shape(),stride,ordering);
        copyFromReal(m);

    }


    /** Construct a complex matrix from a realComponent matrix. */
    public BaseComplexNDArray(INDArray m,char ordering) {
        this(m.shape(),ordering);
        copyFromReal(m);
    }


    /** Construct a complex matrix from a realComponent matrix. */
    public BaseComplexNDArray(INDArray m) {
        this(m, Nd4j.order());
    }

    /**
     * Create with the specified ndarray as the real component
     * and the given stride
     * @param m the ndarray to use as the stride
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(INDArray m,int[] stride) {
        this(m,stride, Nd4j.order());
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     * @param slices the slices of the ndarray
     * @param shape the final shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices,int[] shape,int[] stride) {
        this(slices,shape,stride, Nd4j.order());
    }



    /**
     * Create an ndarray from the specified slices
     * and the given shape
     * @param slices the slices of the ndarray
     * @param shape the final shape of the ndarray
     * @param stride the stride of the ndarray
     * @param ordering the ordering for the ndarray
     *
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices,int[] shape,int[] stride,char ordering) {
        this(new float[ArrayUtil.prod(shape) * 2]);
        List<IComplexNumber> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++) {
            IComplexNDArray flattened = slices.get(i).ravel();
            for(int j = 0; j < flattened.length(); j++)
                list.add(flattened.getComplex(j));
        }


        this.ordering = ordering;
        this.data = new float[ArrayUtil.prod(shape) * 2 ];
        this.stride = stride;
        initShape(shape);

        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            putScalar(count,list.get(i));
            count ++;
        }

        this.linearView = Nd4j.createComplex(data, new int[]{1, length}, offset());


    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     * @param slices the slices of the ndarray
     * @param shape the final shape of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices,int[] shape,char ordering) {
        this(slices,shape,ordering == NDArrayFactory.C ? ArrayUtil.calcStrides(shape,2) : ArrayUtil.calcStridesFortran(shape,2),ordering);


    }

    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, int offset, Character order) {
        this.data = data;
        this.stride = stride;
        this.offset = offset;
        this.ordering = order;
        initShape(shape);
    }



    protected void copyFromReal(INDArray real) {
        INDArray linear = real.linearView();
        IComplexNDArray thisLinear = linearView();
        for(int i = 0; i < linear.length(); i++) {
            thisLinear.putScalar(i, Nd4j.createComplexNumber(linear.get(i),0));
        }
    }

    protected void copyRealTo(INDArray arr) {
        INDArray linear = arr.linearView();
        IComplexNDArray thisLinear = linearView();
        for(int i = 0; i < linear.length(); i++) {
            arr.putScalar(i, thisLinear.getReal(i));
        }

    }

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    @Override
    public IComplexNDArray linearView() {
        if(linearView == null)
            linearView = Nd4j.createComplex(data, new int[]{1, length}, offset());
        return (IComplexNDArray) linearView;
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     * @param slices the slices of the ndarray
     * @param shape the final shape of the ndarray
     */
    public BaseComplexNDArray(List<IComplexNDArray> slices,int[] shape) {
        this(slices,shape, Nd4j.order());


    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     */
    public BaseComplexNDArray(IComplexNumber[] newData,int[] shape) {
        super(new float[ArrayUtil.prod(shape) * 2]);
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i, newData[i].asDouble());

    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     */
    public BaseComplexNDArray(IComplexNumber[] newData,int[] shape,int[] stride) {
        super(new float[ArrayUtil.prod(shape) * 2]);
        this.stride = stride;
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i,newData[i].asDouble());

    }





    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float
     * @param newData the new data for this array
     * @param shape the shape of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public BaseComplexNDArray(IComplexNumber[] newData,int[] shape,char ordering) {
        super(new float[ArrayUtil.prod(shape) * 2]);
        this.ordering = ordering;
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i,newData[i]);

    }

    /**
     * Initialize with the given data,shape and stride
     * @param data the data to use
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(float[] data,int[] shape,int[] stride) {
        this(data,shape,stride,0, Nd4j.order());
    }









    public BaseComplexNDArray(float[] data,int[] shape) {
        this(data,shape,0);
    }


    public BaseComplexNDArray(float[] data,int[] shape,int offset,char ordering) {
        this(data,shape,ordering == NDArrayFactory.C ? calcStrides(shape,2) : calcStridesFortran(shape,2),offset,ordering);
    }

    public BaseComplexNDArray(float[] data,int[] shape,int offset) {
        this(data,shape,offset, Nd4j.order());
    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public BaseComplexNDArray(int[] shape,int[] stride,int offset) {
        this(new float[ArrayUtil.prod(shape) * 2],shape,stride,offset);
    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     * @param ordering the ordering for the ndarray
     */
    public BaseComplexNDArray(int[] shape,int[] stride,int offset,char ordering) {
        this(new float[ArrayUtil.prod(shape) * 2],shape,stride,offset);
        this.ordering = ordering;
    }



    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(int[] shape,int[] stride,char ordering){
        this(shape,stride,0,ordering);
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseComplexNDArray(int[] shape,int[] stride){
        this(shape,stride,0);
    }

    /**
     *
     * @param shape
     * @param offset
     */
    public BaseComplexNDArray(int[] shape,int offset) {
        this(shape,offset, Nd4j.order());
    }




    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseComplexNDArray(int newRows, int newColumns) {
        this(new int[]{newRows,newColumns});
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering the ordering of the ndarray
     */
    public BaseComplexNDArray(int newRows, int newColumns,char ordering) {
        this(new int[]{newRows,newColumns},ordering);
    }




    public BaseComplexNDArray(float[] data, int[] shape, int[] stride, int offset) {
        this(data,shape,stride,offset, Nd4j.order());
    }






    @Override
    public IComplexNumber getComplex(int i, IComplexNumber result) {
        IComplexNumber d = getComplex(i);
        return result.set(d.realComponent(),d.imaginaryComponent());
    }

    @Override
    public IComplexNumber getComplex(int i, int j, IComplexNumber result) {
        IComplexNumber d = getComplex(i,j);
        return result.set(d.realComponent(),d.imaginaryComponent());
    }

    @Override
    public IComplexNDArray putScalar(int j, int i, IComplexNumber conji) {
        int idx = index(j,i);
        data[idx] = conji.realComponent().floatValue();
        data[idx + 1] = conji.imaginaryComponent().floatValue();
        return this;
    }

    @Override
    public IComplexNDArray lt(Number other) {
        return Nd4j.createComplex(super.lt(other));
    }

    @Override
    public IComplexNDArray lti(Number other) {
        return Nd4j.createComplex(super.lti(other));
    }

    @Override
    public IComplexNDArray eq(Number other) {
        return Nd4j.createComplex(super.eq(other));
    }

    @Override
    public IComplexNDArray eqi(Number other) {
        return Nd4j.createComplex(super.eqi(other));
    }

    @Override
    public IComplexNDArray gt(Number other) {
        return Nd4j.createComplex(super.gt(other));
    }

    @Override
    public IComplexNDArray gti(Number other) {
        return Nd4j.createComplex(super.gti(other));
    }

    @Override
    public IComplexNDArray lt(INDArray other) {
        return Nd4j.createComplex(super.lt(other));
    }

    @Override
    public IComplexNDArray lti(INDArray other) {
        return Nd4j.createComplex(super.lti(other));
    }

    @Override
    public IComplexNDArray eq(INDArray other) {
        return Nd4j.createComplex(super.eq(other));
    }

    @Override
    public IComplexNDArray eqi(INDArray other) {
        return Nd4j.createComplex(super.eqi(other));
    }

    @Override
    public IComplexNDArray gt(INDArray other) {
        return Nd4j.createComplex(super.gt(other));
    }

    @Override
    public IComplexNDArray gti(INDArray other) {
        return Nd4j.createComplex(super.gti(other));
    }


    @Override
    public IComplexNDArray rdiv(Number n, INDArray result) {
        return dup().rdivi(n,result);
    }

    @Override
    public IComplexNDArray rdivi(Number n, INDArray result) {
        return rdivi(Nd4j.createFloat(n.floatValue(), 0), result);

    }

    @Override
    public IComplexNDArray rsub(Number n, INDArray result) {
        return dup().rsubi(n,result);
    }

    @Override
    public IComplexNDArray rsubi(Number n, INDArray result) {
        return rsubi(Nd4j.createFloat(n.floatValue(), 0), result);

    }

    @Override
    public IComplexNDArray div(Number n, INDArray result) {
        return dup().divi(n,result);
    }

    @Override
    public IComplexNDArray divi(Number n, INDArray result) {
        return divi(Nd4j.createFloat(n.floatValue(), 0), result);

    }

    @Override
    public IComplexNDArray mul(Number n, INDArray result) {
        return dup().muli(n,result);
    }

    @Override
    public IComplexNDArray muli(Number n, INDArray result) {
        return muli(Nd4j.createFloat(n.floatValue(), 0), result);

    }

    @Override
    public IComplexNDArray sub(Number n, INDArray result) {
        return dup().subi(n,result);
    }

    @Override
    public IComplexNDArray subi(Number n, INDArray result) {
        return subi(Nd4j.createFloat(n.floatValue(), 0), result);
    }

    @Override
    public IComplexNDArray add(Number n, INDArray result) {
        return dup().addi(n,result);
    }

    @Override
    public IComplexNDArray addi(Number n, INDArray result) {
        return addi(Nd4j.createFloat(n.floatValue(), 0),result);
    }

    @Override
    public IComplexNDArray dup() {
        float[] dupData = new float[data.length];
        System.arraycopy(data,0,dupData,0,dupData.length);
        IComplexNDArray ret = Nd4j.createComplex(dupData, shape, stride, offset, ordering);
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
    public IComplexNDArray put(int i, int j, Number element) {
        return put(i,j, Nd4j.scalar(element));
    }


    /**
     * @param indexes
     * @param value
     * @return
     */
    @Override
    public IComplexNDArray put(int[] indexes, float value) {
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
    @Override
    public IComplexNDArray putSlice(int slice, IComplexNDArray put) {
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


        IComplexNDArray view = slice(slice);

        if(put.isScalar())
            put(slice,put.get(0));
        else if(put.isVector())
            for(int i = 0; i < put.length(); i++)
                view.putScalar(i, put.getComplex(i));
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





    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     * @param dimension the dimension to swap
     * @param with the one to swap it with
     * @return the swapped axes view
     */
    public IComplexNDArray swapAxes(int dimension,int with) {
        int[] shape = ArrayUtil.range(0,shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
    }





    /**
     * Compute complex conj (in-place).
     */
    @Override
    public IComplexNDArray conji() {
        IComplexNDArray reshaped = reshape(1,length);
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
        for(int i = 0; i < linearView.length(); i++) {
            linearRet.putScalar(i,linearView.getImag(i));
        }
        return result;
    }

    @Override
    public double getImag(int i) {
        int linear = linearIndex(i);
        return data[linear + 1];
    }

    @Override
    public double getReal(int i) {
        int linear = linearIndex(i);
        return data[linear];
    }

    @Override
    public IComplexNDArray putReal(int rowIndex, int columnIndex, float value) {
        data[2 * index(rowIndex, columnIndex) + offset] = value;
        return this;
    }




    @Override
    public int linearIndex(int i) {
        int realStride = stride[0];
        int idx = offset + (i * realStride);
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }



    @Override
    public IComplexNDArray putImag(int rowIndex, int columnIndex, float value) {
        data[index(rowIndex, columnIndex) + 1 + offset] = value;
        return this;
    }

    @Override
    public IComplexNDArray putReal(int i, float v) {
        int idx = linearIndex(i);
        data[idx] = v;
        return this;
    }

    @Override
    public IComplexNDArray putImag(int i, float v) {
        int idx = linearIndex(i);
        data[idx * 2 + 1] = v;
        return this;
    }


    @Override
    public IComplexNumber getComplex(int i) {
        int idx = linearIndex(i);
        return Nd4j.createDouble(data[idx], data[idx + 1]);
    }

    @Override
    public IComplexNumber getComplex(int i, int j) {
        int idx = index(i,j);
        return Nd4j.createDouble(data[idx], data[idx + 1]);

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
        Nd4j.getBlasWrapper().dcopy(length, data, 1, 2, ret.data(), 0, 1);
        return ret;
    }





    /**
     * Iterate along a dimension.
     * This encapsulates the process of sum, mean, and other processes
     * take when iterating over a dimension.
     * @param dimension the dimension to iterate over
     * @param op the operation to apply
     * @param modify whether to modify this array or not based on the results
     */
    @Override
    public void iterateOverDimension(int dimension,SliceOp op,boolean modify) {
        if(isScalar()) {
            if(dimension > 1)
                throw new IllegalArgumentException("Dimension must be 0 for a scalar");
            else {
                DimensionSlice slice = this.vectorForDimensionAndOffset(0,0);
                op.operate(slice);

                if(modify && slice.getIndices() != null) {
                    IComplexNDArray result = (IComplexNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.getComplex(i).realComponent().floatValue();
                        data[slice.getIndices()[i] + 1] = result.getComplex(i).imaginaryComponent().floatValue();
                    }
                }
            }
        }



        else if(isVector()) {
            if(dimension == 0) {
                DimensionSlice slice = vectorForDimensionAndOffset(0,0);
                op.operate(slice);
                if(modify && slice.getIndices() != null) {
                    IComplexNDArray result = (IComplexNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.getComplex(i).realComponent().floatValue();
                        data[slice.getIndices()[i] + 1] = result.getComplex(i).imaginaryComponent().floatValue();
                    }
                }
            }
            else if(dimension == 1) {
                for(int i = 0; i < length; i++) {
                    DimensionSlice slice = vectorForDimensionAndOffset(dimension,i);
                    op.operate(slice);
                    if(modify && slice.getIndices() != null) {
                        IComplexNDArray result = (IComplexNDArray) slice.getResult();
                        for(int j = 0; j < slice.getIndices().length; j++) {
                            data[slice.getIndices()[j]] = result.getComplex(j).realComponent().floatValue();
                            data[slice.getIndices()[j] + 1] = result.getComplex(j).imaginaryComponent().floatValue();
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
                        IComplexNDArray result = (IComplexNDArray) vector.getResult();
                        for(int i = 0; i < vector.getIndices().length; i++) {
                            data[vector.getIndices()[i]] = result.getComplex(i).realComponent().floatValue();
                            data[vector.getIndices()[i] + 1] = result.getComplex(i).imaginaryComponent().floatValue();
                        }
                    }

                }

            }

            else {
                //needs to be 2 * shape: this is due to both realComponent and imaginary components
                float[] data2 = new float[ArrayUtil.prod(shape) ];
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
                        IComplexNDArray result = (IComplexNDArray) pair.getResult();
                        for(int i = 0; i < pair.getIndices().length; i++) {
                            data[pair.getIndices()[i]] = result.getComplex(i).realComponent().floatValue();
                            data[pair.getIndices()[i] + 1] = result.getComplex(i).imaginaryComponent().floatValue();
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
        IComplexNDArray ret = Nd4j.createComplex(new int[]{shape[dimension]});
        boolean newSlice = false;
        List<Integer> indices = new ArrayList<>();
        for(int j = offset; count < this.shape[dimension]; j+= this.stride[dimension] ) {
            IComplexDouble d = Nd4j.createDouble(data[j], data[j + 1]);
            indices.add(j);
            ret.putScalar(count++, d);
            if(j >= currOffsetForSlice)
                newSlice = true;

        }

        return new DimensionSlice(newSlice,ret,ArrayUtil.toArray(indices));
    }


    //getFromOrigin one result along one dimension based on the given offset
    public DimensionSlice vectorForDimensionAndOffset(int dimension, int offset) {
        if(isScalar() && dimension == 0 && offset == 0)
            return new DimensionSlice(false, Nd4j.complexScalar(get(offset)),new int[]{offset});


            //need whole vector
        else  if (isVector()) {
            if(dimension == 0) {
                int[] indices = new int[length];
                for(int i = 0; i < indices.length; i++)
                    indices[i] = i;
                return new DimensionSlice(false,dup(),indices);
            }

            else if(dimension == 1) {
                return new DimensionSlice(false, Nd4j.complexScalar(get(offset)),new int[]{offset});
            }

            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);
        }


        else {
            int count = 0;
            IComplexNDArray ret = Nd4j.createComplex(new int[]{shape[dimension]});
            List<Integer> indices = new ArrayList<>();
            for (int j = offset; count < this.shape[dimension]; j += this.stride[dimension] ) {
                IComplexDouble d = Nd4j.createDouble(data[j], data[j + 1]);
                ret.putScalar(count++, d);
                indices.add(j);
            }

            return new DimensionSlice(false, ret, ArrayUtil.toArray(indices));
        }

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
        if(element == null)
            throw new IllegalArgumentException("Unable to insert null element");
        assert element.isScalar() : "Unable to insert non scalar element";
        int idx = linearIndex(i);
        IComplexNumber n = element.getComplex(0);
        data[idx] = n.realComponent().floatValue();
        data[idx + 1] = n.imaginaryComponent().floatValue();
        return this;
    }




    //getFromOrigin one result along one dimension based on the given offset
    private ComplexIterationResult op(int dimension, int offset, Ops.DimensionOp op,int currOffsetForSlice) {
        float[] dim = new float[this.shape[dimension]];
        int count = 0;
        boolean newSlice = false;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            float d = data[j];
            dim[count++] = d;
            if(j >= currOffsetForSlice)
                newSlice = true;
        }

        IComplexNDArray r = Nd4j.createComplex(dim);
        IComplexDouble r2 = reduceVector(op,r);
        return new ComplexIterationResult(newSlice,r2);
    }


    //getFromOrigin one result along one dimension based on the given offset
    private IComplexNumber op(int dimension, int offset, Ops.DimensionOp op) {
        float[] dim = new float[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            float d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op, Nd4j.createComplex(dim));
    }





    private IComplexDouble reduceVector(Ops.DimensionOp op,IComplexNDArray vector) {

        switch(op) {
            case SUM:
                return (IComplexDouble) vector.sum(Integer.MAX_VALUE).element();
            case MEAN:
                return (IComplexDouble) vector.mean(Integer.MAX_VALUE).element();
            case NORM_1:
                return Nd4j.createDouble((double) vector.norm1(Integer.MAX_VALUE).element(), 0);
            case NORM_2:
                return Nd4j.createDouble((double) vector.norm2(Integer.MAX_VALUE).element(), 0);
            case NORM_MAX:
                return Nd4j.createDouble((double) vector.normmax(Integer.MAX_VALUE).element(), 0);
            case FFT:
            default: throw new IllegalArgumentException("Illegal operation");
        }
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
        return Nd4j.scalar(Nd4j.createDouble(data[ix], data[ix + 1]));
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
    @Override
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
            return Nd4j.scalar(reduceVector(op, this));


        int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

        if(dimension == 0) {
            float[] data2 = new float[ArrayUtil.prod(shape) * 2];
            int dataIter = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for(int offset = this.offset; offset < numTimes; offset++) {
                IComplexNumber reduce = op(dimension, offset, op);
                data2[dataIter++] = reduce.realComponent().floatValue();
                data2[dataIter++] = reduce.imaginaryComponent().floatValue();


            }

            return Nd4j.createComplex(data2, shape);
        }

        else {
            float[] data2 = new float[ArrayUtil.prod(shape)];
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
                data2[dataIter++] = reduce.realComponent().floatValue();
                data2[dataIter++] = reduce.imaginaryComponent().floatValue();
                //go to next slice and iterate over that
                if(pair.isNextIteration()) {
                    //will update to next step
                    offset = sliceIndices[currOffset];
                    numTimes +=  sliceIndices[currOffset];
                    currOffset++;
                }

            }

            return Nd4j.createComplex(data2, shape);
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
                    view.put(i,j, Nd4j.scalar((IComplexNumber) put.getScalar(i, j).element()));

                }

        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for(int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i,view.slice(i));

        }

        return this;
    }

    @Override
    public IComplexNDArray subArray(int[] offsets, int[] shape,int[] stride) {
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
            data[ix] = (float) element2.realComponent();
            data[ix + 1]= (float) element2.imaginaryComponent();
        }
        else {
            float element2 = (float) element.element();
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
    public IComplexNDArray slice(int slice) {
        int offset = this.offset + (slice * stride[0]);

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

        }

        else {
            if(offset >= data.length)
                throw new IllegalArgumentException("Offset index is > data.length");
            ret = Nd4j.createComplex(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset, ordering);
        }
        ret.toString();
        return ret;
    }


    /**
     * Returns the slice of this from the specified dimension
     * @param slice the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    @Override
    public IComplexNDArray slice(int slice, int dimension) {
        int offset = this.offset + dimension * stride[slice];
        if(this.offset == 0)
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
                this.stride = ArrayUtil.calcStridesFortran(this.shape,2);
            else
                this.stride = ArrayUtil.calcStrides(this.shape,2);
        }

        //recalculate stride: this should only happen with row vectors
        if(this.stride.length != this.shape.length) {
            if(ordering == NDArrayFactory.FORTRAN)
                this.stride = ArrayUtil.calcStridesFortran(this.shape,2);
            else
                this.stride = ArrayUtil.calcStrides(this.shape,2);
        }
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
        IComplexNDArray result = Nd4j.createComplex(newShape);
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
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    @Override
    public IComplexNDArray assign(IComplexNDArray arr) {
        if(!arr.isScalar())
            LinAlgExceptions.assertSameShape(this, arr);
        System.arraycopy(arr.data(), arr.offset(), data(), offset(), arr.length());


        IComplexNDArray linear = linearView();
        for(int i = 0; i < linear.length(); i++) {
            linear.putScalar(i,arr.getComplex(0));
        }


        return this;
    }
    /**
     * Get whole rows from the passed indices.
     *
     * @param rindices
     */
    @Override
    public IComplexNDArray getRows(int[] rindices) {
        INDArray rows = Nd4j.create(rindices.length, columns());
        for(int i = 0; i < rindices.length; i++) {
            rows.putRow(i,getRow(rindices[i]));
        }
        return (IComplexNDArray) rows;
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
        int idx = linearIndex(i);
        data[idx] = value.realComponent().floatValue();
        data[idx + 1] = value.imaginaryComponent().floatValue();
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
        if(ordering == NDArrayFactory.C) {

            if(dimension == shape.length - 1 && dimension != 0) {
                if (size(dimension) == 1)
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * 2);
                else
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * 2 * stride[dimension - 1]);
            }

            else if(dimension == 0)
                return Nd4j.createComplex(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * 2);


            if(size(dimension) == 0)
                return  Nd4j.createComplex(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * 2);

            return  Nd4j.createComplex(data,
                    new int[]{shape[dimension], 1}
                    , new int[]{stride[dimension], 1},
                    offset + index * 2 * stride[0]);
        }

        else if(ordering == NDArrayFactory.FORTRAN) {

            if(dimension == shape.length - 1 && dimension != 0) {
                if(size(dimension) == 1) {
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * 2);
                }

                else
                    return Nd4j.createComplex(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index * 2 * stride[0]);
            }

            if(size(dimension) == 1) {
                return Nd4j.createComplex(data,
                        new int[]{1, shape[dimension]}
                        , ArrayUtil.removeIndex(stride, 0),
                        offset + index * 2);
            }
            else
                return  Nd4j.createComplex(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * 2 * stride[0]);
        }

        throw new IllegalStateException("Illegal ordering..none declared");}

    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public IComplexNDArray cumsumi(int dimension) {
        if(isVector()) {
            IComplexNumber s = Nd4j.createDouble(0, 0);
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
    public IComplexNDArray dimShuffle(Object[] rearrange,int[] newOrder,boolean[] broadCastable) {
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


            IComplexNDArray ret = permute(newShape);
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
    public IComplexNDArray putScalar(int i, Number value) {
        return put(i, Nd4j.scalar(value));
    }

    @Override
    public INDArray putScalar(int[] i, Number value) {
        super.putScalar(i,value);
        return putScalar(i, Nd4j.createComplexNumber(value.floatValue(), 0));
    }

    @Override
    public INDArray putScalar(int[] indexes, IComplexNumber complexNumber) {
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }

        data[ix] = complexNumber.asFloat().realComponent();
        data[ix + 1] = complexNumber.asFloat().imaginaryComponent();

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
        return  Transforms.neg(this);
    }

    @Override
    public IComplexNDArray rdiv(Number n) {
        return rdiv(n,this);
    }

    @Override
    public IComplexNDArray rdivi(Number n) {
        return rdivi(n,this);
    }

    @Override
    public IComplexNDArray rsub(Number n) {
        return rsub(n,this);
    }

    @Override
    public IComplexNDArray rsubi(Number n) {
        return rsubi(n,this);
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


        int[] offsets =  Indices.offsets(indexes);
        int[] shape = Indices.shape(shape(),indexes);
        int[] strides = ordering == 'f' ? ArrayUtil.calcStridesFortran(shape,2) :  ArrayUtil.copy(stride());

        return subArray(offsets,shape,strides);
    }

    /**
     * Get whole columns from the passed indices.
     *
     * @param cindices
     */
    @Override
    public IComplexNDArray getColumns(int[] cindices) {
        IComplexNDArray rows = Nd4j.createComplex(rows(), cindices.length);
        for(int i = 0; i < cindices.length; i++) {
            rows.putColumn(i,getColumn(cindices[i]));
        }
        return  rows;
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
        super.putRow(row, toPut);
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
        super.putColumn(column, toPut);
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
        return  getScalar(new int[]{row, column});
    }




    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public IComplexNDArray getScalar(int i) {
        int idx  = linearIndex(i);
        return Nd4j.scalar(Nd4j.createDouble(data[idx], data[idx + 1]));
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
            IComplexNDArray n1 = (IComplexNDArray) element;
            IComplexNumber n = n1.getComplex(0);
            put(i,n);
        }
        else
            put(i,element.get(0));
        return this;
    }

    private void put(int i, float element) {
        int idx = linearIndex(i);
        data[idx] = element;
        data[idx + 1] = 0.0f;
    }

    public void put(int i, IComplexNumber element) {
        int idx = linearIndex(i);
        data[idx] = element.realComponent().floatValue();
        data[idx + 1] = element.imaginaryComponent().floatValue();
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


        LinAlgExceptions.assertMultiplies(this,other);


        IComplexNDArray otherArray = Nd4j.createComplex(other);
        IComplexNDArray resultArray = Nd4j.createComplex(result);





        if (result == this || result == other) {
			/* actually, blas cannot do multiplications in-place. Therefore, we will fake by
			 * allocating a temporary object on the side and copy the result later.
			 */

            IComplexNDArray temp = Nd4j.createComplex(resultArray.shape(), ArrayUtil.calcStrides(resultArray.shape()));
            Nd4j.getBlasWrapper().gemm(Nd4j.createFloat(1, 0), this, otherArray, Nd4j.createFloat(0, 0), temp);

            Nd4j.getBlasWrapper().copy(temp, resultArray);

        }
        else {
            IComplexNDArray thisInput =  this;
            Nd4j.getBlasWrapper().gemm(Nd4j.createFloat(1, 0), thisInput, otherArray, Nd4j.createFloat(0, 0), resultArray);
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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        IComplexNDArray linear = linearView();
        IComplexNDArray cOtherLinear = cOther.linearView();
        IComplexNDArray cResultLinear = cResult.linearView();

        if (other.isScalar())
            return divi(cOther.getComplex(0), result);


        IComplexNumber c = Nd4j.createComplexNumber(0, 0);
        IComplexNumber d =  Nd4j.createComplexNumber(0, 0);

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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        IComplexNDArray linear = linearView();
        IComplexNDArray cOtherLinear = cOther.linearView();
        IComplexNDArray cResultLinear = cResult.linearView();

        if (other.isScalar())
            return muli(cOther.getComplex(0), result);


        IComplexNumber c = Nd4j.createComplexNumber(0, 0);
        IComplexNumber d =  Nd4j.createComplexNumber(0, 0);

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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        if (other.isScalar())
            return subi(cOther.getComplex(0), result);


        if (result == this)
            Nd4j.getBlasWrapper().axpy(Nd4j.NEG_UNIT, cOther, cResult);
        else if (result == other) {
            Nd4j.getBlasWrapper().scal(Nd4j.NEG_UNIT, cResult);
            Nd4j.getBlasWrapper().axpy(Nd4j.UNIT, this, cResult);
        }
        else {
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
        IComplexNDArray cOther = (IComplexNDArray) other;
        IComplexNDArray cResult = (IComplexNDArray) result;

        if (cOther.isScalar()) {
            return cResult.addi(cOther.getComplex(0),result);
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
            for(int i = 0; i < resultLinear.length(); i++) {
                resultLinear.putScalar(i,otherLinear.get(i) + linear.get(i));
            }

        }

        return (IComplexNDArray) result;
    }




    @Override
    public IComplexNDArray rdiv(IComplexNumber n, INDArray result) {
        return dup().rdivi(n,result);
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
        return dup().rsubi(n,result);
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
        return dup().divi(n,result);
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
        return dup().muli(n,result);
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
        return dup().subi(n,result);
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
        return dup().addi(n,result);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n, INDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = (IComplexNDArray) result.linearView();
        for(int i = 0; i < length(); i++) {
            cResult.putScalar(i,linear.getComplex(i).add(n));
        }

        return (IComplexNDArray) result;
    }

    @Override
    public IComplexNDArray rdiv(IComplexNumber n) {
        return dup().rdivi(n,this);
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n) {
        return rdivi(n,this);
    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n) {
        return rsub(n, this);
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n) {
        return rsubi(n,this);
    }

    @Override
    public IComplexNDArray div(IComplexNumber n) {
        return div(n,this);
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n) {
        return divi(n,this);
    }

    @Override
    public IComplexNDArray mul(IComplexNumber n) {
        return dup().muli(n);
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n) {
        return muli(n,this);
    }

    @Override
    public IComplexNDArray sub(IComplexNumber n) {
        return dup().subi(n);
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n) {
        return subi(n,this);
    }

    @Override
    public IComplexNDArray add(IComplexNumber n) {
        return addi(n,this);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n) {
        return addi(n,this);
    }








    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public IComplexNDArray transpose() {
        //transpose of row vector is column vector
        if(isRowVector())
            return Nd4j.createComplex(data, new int[]{shape[0], 1}, offset);
            //transpose of a column vector is row vector
        else if(isColumnVector())
            return Nd4j.createComplex(data, new int[]{shape[0]}, offset);

        IComplexNDArray n = Nd4j.createComplex(data, reverseCopy(shape), reverseCopy(stride), offset);
        return n;

    }

    @Override
    public IComplexNDArray addi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult =  result.linearView();
        for(int i = 0; i < length(); i++) {
            cResult.putScalar(i,linear.getComplex(i).addi(n));
        }

        return  result;

    }


    @Override
    public IComplexNDArray subi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for(int i = 0; i < length(); i++) {
            cResult.putScalar(i,linear.getComplex(i).subi(n));
        }

        return  result;

    }


    @Override
    public IComplexNDArray muli(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for(int i = 0; i < length(); i++) {
            IComplexNumber n3 = linear.getComplex(i);
            IComplexNumber num = n3.mul(n);
            cResult.putScalar(i,linear.getComplex(i).mul(n));
        }

        return  result;

    }



    @Override
    public IComplexNDArray divi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for(int i = 0; i < length(); i++) {
            cResult.putScalar(i,linear.getComplex(i).div(n));
        }

        return result;

    }


    @Override
    public IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for(int i = 0; i < length(); i++) {
            cResult.putScalar(i,n.sub(linear.getComplex(i)));
        }

        return  result;

    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result) {
        IComplexNDArray linear = linearView();
        IComplexNDArray cResult = result.linearView();
        for(int i = 0; i < length(); i++) {
            cResult.putScalar(i,n.div(linear.getComplex(i)));
        }

        return  result;

    }

    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     * @param shape
     * @return
     */
    @Override
    public IComplexNDArray reshape(int[] shape) {
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

        if(Shape.shapeEquals(shape(),shape))
            return this;

        //row to column vector
        if(isRowVector()) {
            if(Shape.isColumnVectorShape(shape)) {
                return Nd4j.createComplex(data, shape, Nd4j.getComplexStrides(shape, ordering), offset, ordering);
            }
        }
        //column to row vector
        if(isColumnVector()) {
            if(Shape.isRowVectorShape(shape)) {
                return Nd4j.createComplex(data, shape, Nd4j.getComplexStrides(shape, ordering), offset, ordering);

            }
        }



        //returns strides for reshape or null if data needs to be copied
        int[] newStrides = newStridesReshape(shape);

        if(newStrides != null) {
            IComplexNDArray ndArray = Nd4j.createComplex(data, shape, newStrides, offset, ordering);
            return ndArray;


        }

        //need to copy data
        else {
            IComplexNDArray create = Nd4j.createComplex(shape, Nd4j.getComplexStrides(shape, ordering));
            final IComplexNDArray flattened = ravel();
            //current position in the vector
            final AtomicInteger vectorCounter = new AtomicInteger(0);
            //row order
            if(ordering == NDArrayFactory.C) {
                create.iterateOverAllRows(new SliceOp() {
                    @Override
                    public void operate(DimensionSlice nd) {
                        IComplexNDArray nd1 = (IComplexNDArray) nd.getResult();
                        for (int i = 0; i < nd1.length(); i++) {
                            int element = vectorCounter.getAndIncrement();
                            nd1.put(i, flattened.getScalar(element));
                        }
                    }

                    @Override
                    public void operate(INDArray nd) {
                        for (int i = 0; i < nd.length(); i++) {
                            int element = vectorCounter.getAndIncrement();
                            nd.put(i, flattened.getScalar(element));

                        }
                    }
                });
            }
            //column order
            else if(ordering == NDArrayFactory.FORTRAN) {
                create.iterateOverAllColumns(new SliceOp() {
                    @Override
                    public void operate(DimensionSlice nd) {
                        IComplexNDArray nd1 = (IComplexNDArray) nd.getResult();

                        for(int i = 0; i < nd1.length(); i++) {
                            int element = vectorCounter.getAndIncrement();
                            nd1.put(i, flattened.getScalar(element));

                        }
                    }

                    @Override
                    public void operate(INDArray nd) {
                        for(int i = 0; i < nd.length(); i++) {
                            int element = vectorCounter.getAndIncrement();
                            nd.put(i, flattened.getScalar(element));

                        }
                    }
                });
            }

            return create;

        }



    }

    protected int[] newStridesReshape(int[] shape) {

        int[][] oldShapeAndStride = getNonOneStridesAndShape();
        int[] oldShape = oldShapeAndStride[0];
        int[] oldStride = oldShapeAndStride[1];
         /* oi to oj and ni to nj give the axis ranges currently worked with */
        int newNd = shape.length;
        int oldNd = oldShapeAndStride[0].length;
        int np, op;
        int nk;


        //must be same length
        if (ArrayUtil.prod(shape) != ArrayUtil.prod(oldShape))
            return null;
        //no 0 length arr
        if (ArrayUtil.prod(shape) == 0)
            return null;

        int[] newStrides = new int[oldStride.length];


         /* oi to oj and ni to nj give the axis ranges currently worked with */
        int ni = 0,
                oi = 0,
                nj = 1,
                oj = 1;

        for (; ni < newNd && oi < oldNd; ni = nj++, oi = oj++) {
            np = shape[ni];
            op = oldShape[oi];

            while (np != op) {
                if (np < op)
                /* Misses trailing 1s, these are handled later */
                    np *= shape[nj++];

                else
                    op *= oldShape[oj++];

            }

             /* Check whether the original axes can be combined */
            for (int ok = oi; ok < oj - 1; ok++) {
                if (ordering == NDArrayFactory.FORTRAN) {
                    if (oldStride[ok + 1] != oldStride[ok] * oldStride[ok])
                     /* not contiguous enough */
                        return null;

                } else {
                /* C order */
                    if (oldStride[ok] != oldShape[ok + 1] * oldStride[ok + 1])
                    /* not contiguous enough */
                        return null;

                }
            }


        }
        return Nd4j.getComplexStrides(shape, ordering);

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




    /**
     * Returns a copy of
     * all of the data in this array in order
     * @return all of the data in order
     */
    @Override
    public float[] data() {
        return data;
    }











    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    @Override
    public IComplexNDArray prod(int dimension) {
        return Nd4j.createComplex(super.prod(dimension));

    }

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public IComplexNDArray mean(int dimension) {
        return Nd4j.createComplex(super.mean(dimension));

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
        for(int i = 0; i < one.length(); i++)
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
        return Nd4j.createComplex(super.max(dimension));

    }

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public IComplexNDArray min(int dimension) {
        return Nd4j.createComplex(super.min(dimension));

    }


    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public IComplexNDArray normmax(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return Nd4j.scalar(ComplexOps.normmax(this));
        }

        else if(isVector()) {
            return  sum(Integer.MAX_VALUE);
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = Nd4j.createComplex(new int[]{ArrayUtil.prod(shape)});
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

        if(dimension == Integer.MAX_VALUE) {
            return Nd4j.scalar(ComplexOps.sum(this));
        }

        else if(isVector()) {
            return  sum(Integer.MAX_VALUE);
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = Nd4j.createComplex(new int[]{ArrayUtil.prod(shape)});
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
    public IComplexNDArray norm1(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return Nd4j.scalar(ComplexOps.norm1(this));
        }

        else if(isVector()) {
            return  norm1(Integer.MAX_VALUE);
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = Nd4j.createComplex(new int[]{ArrayUtil.prod(shape)});
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

    public IComplexDouble std() {
        StandardDeviation dev = new StandardDeviation();
        INDArray real = getReal();
        INDArray imag = imag();
        double std = dev.evaluate(ArrayUtil.doubleCopyOf(real.data()));
        double std2 = dev.evaluate(ArrayUtil.doubleCopyOf(imag.data()));
        return Nd4j.createDouble(std, std2);
    }

    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    @Override
    public INDArray std(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return Nd4j.scalar(ComplexOps.std(this));
        }

        else if(isVector()) {
            return  std(Integer.MAX_VALUE);
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = Nd4j.createComplex(new int[]{ArrayUtil.prod(shape)});
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
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    public IComplexNDArray norm2(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return Nd4j.scalar(ComplexOps.norm2(this));
        }

        else if(isVector()) {
            return  sum(Integer.MAX_VALUE);
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = Nd4j.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
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
     * Converts the matrix to a one-dimensional array of doubles.
     */
    @Override
    public IComplexNumber[] toArray() {
        length = ArrayUtil.prod(shape);
        IComplexNumber[] ret = new IComplexNumber[length];
        for(int i = 0; i < ret.length; i++)
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
        return reshape(new int[]{newRows,newColumns});
    }




    /**
     * Get the specified column
     *
     * @param c
     */
    @Override
    public IComplexNDArray getColumn(int c) {
        if(shape.length == 2) {
            if(ordering == NDArrayFactory.C) {
                IComplexNDArray ret = Nd4j.createComplex(
                        data,
                        new int[]{shape[0]},
                        new int[]{stride[0]},
                        offset + (c * 2), ordering
                );

                return ret;
            }
            else {
                IComplexNDArray ret = Nd4j.createComplex(
                        data,
                        new int[]{shape[0]},
                        new int[]{stride[1]},
                        offset + (c * 2), ordering
                );

                return ret;
            }

        }

        else if(isColumnVector() && c == 0)
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
        if(shape.length == 2) {
            if(ordering == NDArrayFactory.C) {
                IComplexNDArray ret = Nd4j.createComplex(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + (r * 2) * columns(),
                        ordering
                );
                return ret;
            }
            else {
                IComplexNDArray ret  = Nd4j.createComplex(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + (r * 2),
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
     * ComplexDoubleMatrix which has the same size and the maximal absolute
     * difference in matrix elements is smaller than 1e-6.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        IComplexNDArray n = null;
        if(!(o instanceof  IComplexNDArray))
            return false;

        if(n == null)
            n = (IComplexNDArray) o;

        //epsilon equals
        if(isScalar() && n.isScalar()) {
            IComplexNumber c = n.getComplex(0);
            return Math.abs(getComplex(0).sub(c).realComponent().floatValue()) < 1e-6;
        }
        else if(isVector() && n.isVector()) {
            for(int i = 0; i < length; i++) {
                float curr = getComplex(i).realComponent().floatValue();
                float comp = n.getComplex(i).realComponent().floatValue();
                float currImag = getComplex(i).imaginaryComponent().floatValue();
                float compImag = n.getComplex(i).imaginaryComponent().floatValue();
                if(Math.abs(curr - comp) > 1e-3 || Math.abs(currImag - compImag) > 1e-3)
                    return false;
            }

            return true;

        }

        if(!Shape.shapeEquals(shape(),n.shape()))
            return false;
        //epsilon equals
        if(isScalar()) {
            IComplexNumber c = n.getComplex(0);
            return getComplex(0).sub(c).absoluteValue().doubleValue() < 1e-6;
        }
        else if(isVector()) {
            for(int i = 0; i < length; i++) {
                IComplexNumber curr = getComplex(i);
                IComplexNumber comp = n.getComplex(i);
                if(curr.sub(comp).absoluteValue().doubleValue() > 1e-6)
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
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public IComplexNDArray broadcast(int[] shape) {

        int dims = this.shape.length;
        int targetDimensions = shape.length;
        if (targetDimensions < dims) {
            throw new IllegalArgumentException("Invalid shape to broad cast " + Arrays.toString(shape));
        }
        else if (dims == targetDimensions) {
            if (Shape.shapeEquals(shape, this.shape()))
                return this;
            throw new IllegalArgumentException("Invalid shape to broad cast " + Arrays.toString(shape));
        }
        else {
            int n= shape[0];
            IComplexNDArray s = broadcast(Arrays.copyOfRange(shape, 1, targetDimensions));
            return Nd4j.repeat(s,n);
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
            throw new IllegalStateException("Unable to getScalar the element of a non scalar");
        int idx = linearIndex(0);
        return Nd4j.createDouble(data[idx], data[idx + 1]);
    }


    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public IComplexNDArray permute(int[] rearrange) {
        if(rearrange.length < shape.length)
            return dup();

        checkArrangeArray(rearrange);
        int[] newDims = doPermuteSwap(shape,rearrange);
        int[] newStrides = doPermuteSwap(stride,rearrange);

        IComplexNDArray ret = Nd4j.createComplex(data, newDims, newStrides, offset, ordering);
        return ret;
    }





    /**
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    @Override
    public IComplexNDArray ravel() {
        final IComplexNDArray ret = Nd4j.createComplex(length, ordering);
        final AtomicInteger counter = new AtomicInteger(0);

        SliceOp op = new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                IComplexNDArray nd1 = (IComplexNDArray) nd.getResult();
                for (int i = 0; i < nd1.length(); i++) {
                    int element = counter.getAndIncrement();
                    ret.putScalar(element,nd1.getComplex(i));
                }
            }

            @Override
            public void operate(INDArray nd) {
                IComplexNDArray nd1 = (IComplexNDArray) nd;
                for (int i = 0; i < nd.length(); i++) {
                    int element = counter.getAndIncrement();
                    ret.putScalar(element,nd1.getComplex(i));


                }
            }
        };
        //row order
        if(ordering == NDArrayFactory.C) {
            iterateOverAllRows(op);
        }
        //column order
        else if(ordering == NDArrayFactory.FORTRAN) {
            iterateOverAllColumns(op);
        }

        return ret;

    }



    /** Generate string representation of the matrix. */
    @Override
    public String toString() {
        if (isScalar()) {
            return element().toString();
        }
        else if(isMatrix()) {
            StringBuilder sb = new StringBuilder();
            sb.append('[');
            for(int i = 0; i < rows; i++) {
                sb.append('[');
                for(int j = 0; j < columns; j++) {
                    sb.append(getComplex(i,j));
                    if(j < columns - 1)
                        sb.append(',');
                }
                sb.append(']');

            }

            sb.append("]\n");
            return sb.toString();
        }


        else if(isVector()) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for(int i = 0; i < length; i++) {
                sb.append(getComplex(i));
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




}
