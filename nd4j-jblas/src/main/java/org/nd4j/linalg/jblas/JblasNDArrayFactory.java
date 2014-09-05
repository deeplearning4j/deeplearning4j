package org.nd4j.linalg.jblas;


import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jblas.complex.ComplexDouble;
import org.nd4j.linalg.jblas.complex.ComplexFloat;
import org.nd4j.linalg.jblas.complex.ComplexNDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;

/**
 * Jblas NDArray Factory
 * @author Adam Gibson
 */
public class JblasNDArrayFactory extends BaseNDArrayFactory {

    public JblasNDArrayFactory(String dtype,Character order) {
        super(dtype,order);
    }

    /**
     * Create float
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    @Override
    public IComplexFloat createFloat(float real, float imag) {
        return new ComplexFloat(real,imag);
    }

    /**
     * Create an instance of a complex double
     *
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    @Override
    public IComplexDouble createDouble(double real, double imag) {
        return new ComplexDouble(real,imag);
    }

    /**
     * Create an ndarray with the given data layout
     *
     * @param data the data to create the ndarray with
     * @return the ndarray with the given data layout
     */
    @Override
    public INDArray create(double[][] data) {
        return new NDArray(data);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(INDArray arr) {
        return new ComplexNDArray(arr);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param data  the data to wrap
     * @param shape
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape) {
        if(order == FORTRAN)
            return new ComplexNDArray(data,shape,ArrayUtil.calcStridesFortran(shape));
        else
            return new ComplexNDArray(data,shape);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arrs  the arr to wrap
     * @param shape
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape) {
        if(order == FORTRAN)
            return new ComplexNDArray(arrs,shape,ArrayUtil.calcStridesFortran(shape));

        return new ComplexNDArray(arrs,shape);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset) {
        return new ComplexNDArray(data,shape,stride,offset);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset) {
        return new NDArray(data,shape,stride,offset);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset) {
        return new ComplexNDArray(ArrayUtil.floatCopyOf(data),shape,stride,offset);
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, int offset) {
        return new NDArray(data,shape,stride,offset);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param list
     * @param shape the shape of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
        if(order == FORTRAN)
            return new NDArray(list,shape, ArrayUtil.calcStridesFortran(shape));
        else
            return new NDArray(list,shape);
    }

    /**
     * Create a complex ndarray with the given data
     *
     * @param data     the data to use with tne ndarray
     * @param shape    the shape of the ndarray
     * @param stride   the stride for the ndarray
     * @param offset   the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created complex ndarray
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new ComplexNDArray(ArrayUtil.floatCopyOf(data),shape,stride,offset,ordering);
    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset, char ordering) {
        return new ComplexNDArray(ArrayUtil.floatCopyOf(data),shape,offset,ordering);
    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset) {
         return new ComplexNDArray(ArrayUtil.floatCopyOf(data),shape,offset);
    }

    /**
     * Create a complex ndarray with the given data
     *
     * @param data     the data to use with tne ndarray
     * @param shape    the shape of the ndarray
     * @param stride   the stride for the ndarray
     * @param offset   the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created complex ndarray
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new ComplexNDArray(data,shape,stride,offset,ordering);
    }

    @Override
    public INDArray create(float[][] floats) {
        return new NDArray(floats);

    }

    @Override
    public IComplexNDArray createComplex(float[] dim) {
        return new ComplexNDArray(dim);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new NDArray(data,shape,stride,offset,ordering);
    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset, char ordering) {
        return new ComplexNDArray(data,shape, Nd4j.getComplexStrides(shape, ordering),offset,ordering);

    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset) {
        return new ComplexNDArray(data,shape,offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int offset) {
        return new NDArray(data,shape,offset);
    }
}
