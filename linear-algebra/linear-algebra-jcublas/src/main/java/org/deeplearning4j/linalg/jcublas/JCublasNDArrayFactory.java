package org.deeplearning4j.linalg.jcublas;

import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexDouble;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexFloat;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexFloat;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.BaseNDArrayFactory;

import org.deeplearning4j.linalg.util.ArrayUtil;

import java.util.List;

/**
 * Created by mjk on 8/21/14.
 */
public class JCublasNDArrayFactory extends BaseNDArrayFactory {
    public JCublasNDArrayFactory(String dtype,Character order) {
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
        return new JCublasComplexFloat(real,imag);
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
        return new JCublasComplexDouble(real,imag);
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
        return new JCublasComplexNDArray(arr);
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
            return new JCublasComplexNDArray(data,shape, ArrayUtil.calcStridesFortran(shape));
        else
            return new JCublasComplexNDArray(data,shape);
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
            return new JCublasComplexNDArray(arrs,shape,ArrayUtil.calcStridesFortran(shape));

        return new JCublasComplexNDArray(arrs,shape);
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
        return new JCublasComplexNDArray(data,shape,stride,offset);
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
        return new JCublasNDArray(data,shape,stride,offset);
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
        return new JCublasComplexNDArray(data,shape,stride,offset);
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
        return new JCublasNDArray(data,shape,stride,offset);
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
            return new JCublasNDArray(list,shape, ArrayUtil.calcStridesFortran(shape));
        else
            return new JCublasNDArray(list,shape);
    }
}
